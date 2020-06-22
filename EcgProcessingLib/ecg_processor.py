from typing import Optional, Dict, Tuple

import EcgProcessingLib.utils as utils
import EcgProcessingLib.signal as signal
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal as ss
from NilsPodLib import Dataset
from tqdm.notebook import tqdm


class EcgProcessor:

    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, dataset: Optional[Dataset] = None,
                 df: Optional[pd.DataFrame] = None,
                 sampling_rate: Optional[float] = 256.0):
        if all([i is None for i in [dataset, df, data_dict]]):
            raise ValueError("Either 'dataset', 'df', or 'data_dict' must be specified as parameter!")

        self.sampling_rate: int = int(sampling_rate)

        if data_dict:
            self.data_dict = data_dict
        elif dataset:
            df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utils.utc).tz_convert(tz=utils.tz)
            self.sampling_rate = int(dataset.info.sampling_rate_hz)
            self.data_dict: Dict = {
                'Data': df
            }
        else:
            df = df.tz_localize(tz=utils.utc).tz_convert(tz=utils.tz)
            self.data_dict: Dict = {
                'Data': df
            }
        self.ecg_result: Dict[str, pd.DataFrame] = {}
        self.heart_rate: Dict[str, pd.DataFrame] = {}
        self.rpeak_loc: Dict[str, pd.DataFrame] = {}

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        return {k: pd.DataFrame(v['ECG_Clean']) for k, v in self.ecg_result.items()}

    @property
    def hr_result(self) -> Dict[str, pd.DataFrame]:
        return self.heart_rate

    def ecg_process(self, quality_thres: Optional[float] = 0.4, title: Optional[str] = None,
                    method: Optional[str] = "neurokit") -> None:
        for name, df in tqdm(self.data_dict.items(), desc=title):
            ecg_result, rpeak_idx = self._ecg_process(df, method=method)
            ecg_result['R_Peak_Outlier'] = 0.0
            rpeaks = ecg_result.loc[ecg_result['ECG_R_Peaks'] == 1.0, ['ECG_Quality']]
            rpeaks['R_Peak_Idx'] = rpeak_idx
            # ecg_result.drop('ECG_Rate', axis=1, inplace=True)
            rpeaks = self.correct_outlier(ecg_result, rpeaks, self.sampling_rate, quality_thres)
            heart_rate = pd.DataFrame({'ECG_Rate': 60 / rpeaks['RR_Interval']})
            heart_rate_interpolated = nk.signal_interpolate(rpeaks['R_Peak_Idx'], heart_rate['ECG_Rate'],
                                                            desired_length=len(ecg_result['ECG_Clean']))
            ecg_result['ECG_Rate'] = heart_rate_interpolated
            self.ecg_result[name] = ecg_result
            self.heart_rate[name] = heart_rate
            self.rpeak_loc[name] = rpeaks.drop('RR_Interval', axis=1)

    def _ecg_process(self, data: pd.DataFrame, method: Optional[str] = "neurokit") -> Tuple[pd.DataFrame, np.array]:
        ecg_signal = data['ecg'].values
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate, method=method)
        instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate, method=method)
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks['ECG_R_Peaks'], sampling_rate=self.sampling_rate)

        signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Quality": quality},
                               index=data.index)
        instant_peaks.index = data.index
        return pd.concat([signals, instant_peaks], axis=1), rpeaks['ECG_R_Peaks']

    @classmethod
    def correct_outlier(cls, ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256,
                        quality_thres: Optional[float] = 0.4, corr_thres: Optional[float] = 0.3,
                        hr_thres: Optional[Tuple[int, int]] = (45, 200)) -> pd.DataFrame:
        # signal outlier: copy dataframe to mark removed beats later
        rpeaks_copy = rpeaks.copy()
        # segment individual heart beats
        heartbeats = nk.ecg_segment(ecg_signal['ECG_Clean'], rpeaks['R_Peak_Idx'], sampling_rate)
        heartbeats = nk.epochs_to_df(heartbeats)
        heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')
        heartbeats = heartbeats.set_index('Index')
        heartbeats = heartbeats.loc[heartbeats.index.intersection(rpeaks['R_Peak_Idx'])].sort_values(by="Label")
        heartbeats = heartbeats[~heartbeats.index.duplicated()]
        heartbeats_pivoted.columns = heartbeats.index

        # get the last index because it will get lost when computing the RR interval
        last_idx = rpeaks.iloc[-1]
        # compute the average over all heart beats and compute the correlation coefficient between all beats and
        # the average
        mean_beat = heartbeats_pivoted.mean(axis=1)
        heartbeats_pivoted['mean'] = mean_beat
        corr_coeff = heartbeats_pivoted.corr()['mean'].abs().sort_values(ascending=True)
        corr_coeff.drop('mean', inplace=True)
        # compute RR intervals (in seconds) from R Peak Locations
        rpeaks['RR_Interval'] = np.ediff1d(rpeaks['R_Peak_Idx'], to_end=0) / sampling_rate
        rpeaks['R_Peak_Outlier'] = 0.0

        # signal outlier: drop all beats that are below a correlation coefficient threshold and below the signal
        # quality threshold
        rpeaks[rpeaks['R_Peak_Idx'].isin(corr_coeff[corr_coeff < corr_thres].index)] = None
        rpeaks.loc[rpeaks['ECG_Quality'] < quality_thres] = None

        # statistical outlier: remove the x% highest and lowest beats (1.96 std = 5% outlier, 2.576 std = 1% outlier)
        z_score = (rpeaks['RR_Interval'] - rpeaks['RR_Interval'].mean()) / rpeaks['RR_Interval'].std()
        rpeaks.loc[np.abs(z_score) > 2.576] = None

        # physiological outlier: minimum/maximum heart rate threshold
        rpeaks.loc[(rpeaks['RR_Interval'] > (60 / hr_thres[0])) | (rpeaks['RR_Interval'] < (60 / hr_thres[1]))] = None

        # mark all removed beats as outlier in the ECG dataframe
        removed_beats = rpeaks_copy['R_Peak_Idx'][rpeaks['R_Peak_Idx'].isna()]
        rpeaks.fillna({'R_Peak_Outlier': 1.0}, inplace=True)
        ecg_signal.loc[removed_beats.index, 'R_Peak_Outlier'] = 1.0

        rpeaks.drop('ECG_Quality', axis=1, inplace=True)
        # interpolate the removed beats
        rpeaks.interpolate(method='linear', limit_direction='both', inplace=True)
        rpeaks['R_Peak_Idx'] = rpeaks['R_Peak_Idx'].astype(int)
        rpeaks.loc[rpeaks.index[-1]] = [last_idx['R_Peak_Idx'], rpeaks['RR_Interval'].mean(), 0.0]
        rpeaks.drop_duplicates(subset='R_Peak_Idx', inplace=True)

        # fill missing RR intervals with interpolated R Peak Locations
        rpeaks_corrected = (rpeaks['RR_Interval'].cumsum() * sampling_rate).astype(int)
        rpeaks_corrected = np.append(rpeaks['R_Peak_Idx'].iloc[0], rpeaks_corrected[:-1] + rpeaks['R_Peak_Idx'].iloc[0])
        artifacts, rpeaks_corrected = nk.signal_fixpeaks(rpeaks_corrected, sampling_rate, iterative=True)
        rpeaks.loc[:, 'R_Peak_Idx_Corrected'] = rpeaks_corrected
        return rpeaks

    @classmethod
    def hrv_process(cls, rpeaks: pd.DataFrame, index: Optional[str] = None, index_name: Optional[str] = None,
                    sampling_rate: Optional[int] = 256) -> pd.DataFrame:
        hrv_time = nk.hrv_time(rpeaks['R_Peak_Idx_Corrected'], sampling_rate=sampling_rate)
        hrv_nonlinear = nk.hrv_nonlinear(rpeaks['R_Peak_Idx_Corrected'], sampling_rate=sampling_rate)

        rpeaks = pd.concat([hrv_time, hrv_nonlinear], axis=1)
        if index:
            rpeaks.index = [index]
            rpeaks.index.name = index_name
        return rpeaks

    @classmethod
    def ecg_extract_edr(cls, ecg_signals: pd.DataFrame, rpeaks: pd.DataFrame,
                        sampling_rate: Optional[int] = 256,
                        edr_type: Optional[str] = 'peak_trough_mean') -> pd.DataFrame:
        """Extract ECG-derived respiration."""

        if edr_type not in _edr_methods:
            raise ValueError(
                "`edr_type` must be one of {}, not {}".format(list(_edr_methods.keys()), edr_type))
        edr_func = _edr_methods[edr_type]

        peaks = np.squeeze(rpeaks['R_Peak_Idx'].values)

        troughs = signal.find_extrema_in_radius(ecg_signals['ECG_Clean'], peaks, radius=(int(0.1 * sampling_rate), 0))
        outlier_mask = rpeaks['R_Peak_Outlier'] == 1

        edr_signal_raw = edr_func(ecg_signals['ECG_Clean'], peaks, troughs)
        edr_signal = _remove_outlier_and_interpolate(edr_signal_raw, outlier_mask, peaks, len(ecg_signals))
        edr_signal = nk.signal_filter(edr_signal, sampling_rate=sampling_rate, lowcut=0.1, highcut=0.5, order=10)

        return pd.DataFrame(edr_signal, index=ecg_signals.index, columns=["ECG_Resp"])

    @classmethod
    def rsp_compute_rate(cls, rsp_signal: pd.DataFrame, sampling_rate: Optional[int] = 256):
        # find peaks: minimal distance between peaks: 1 seconds
        rsp_signal = signal.sanitize_input(rsp_signal)
        edr_maxima = ss.find_peaks(rsp_signal, height=0, distance=sampling_rate)[0]
        edr_minima = ss.find_peaks(-1 * rsp_signal, height=0, distance=sampling_rate)[0]
        # threshold: 0.2 * Q3 (= 75th percentile)
        max_threshold = 0.2 * np.percentile(rsp_signal[edr_maxima], 75)
        edr_maxima = edr_maxima[rsp_signal[edr_maxima] > max_threshold]

        rsp_cycles_start_end = np.vstack([edr_maxima[:-1], edr_maxima[1:]]).T

        valid_resp_phases_mask = np.apply_along_axis(_check_contains_trough, 1, rsp_cycles_start_end, edr_minima)
        rsp_cycles_start_end = rsp_cycles_start_end[valid_resp_phases_mask]

        edr_signal_split: list = np.split(rsp_signal, rsp_cycles_start_end.flatten())[1::2]
        rsp_cycles_start_end = rsp_cycles_start_end.T
        rsp_peaks = rsp_cycles_start_end[0]
        rsp_troughs = np.array(list(map(lambda arr: np.argmin(arr), edr_signal_split))) + rsp_cycles_start_end[0]

        rsp_rate_peaks = _rsp_rate(rsp_peaks, sampling_rate, len(rsp_signal))
        rsp_rate_troughs = _rsp_rate(rsp_troughs, sampling_rate, len(rsp_signal))
        return np.concatenate([rsp_rate_peaks, rsp_rate_troughs]).mean()

    @classmethod
    def rsa_process(cls, ecg_signal: pd.DataFrame, rsp_signal: pd.DataFrame, sampling_rate: Optional[int] = 256) -> \
            Dict[str, float]:
        rsp_signal = signal.sanitize_input(rsp_signal)
        rsp_output = nk.rsp_process(rsp_signal, sampling_rate)[0]
        rsp_output.index = ecg_signal.index
        return nk.ecg_rsa(ecg_signal, rsp_output, sampling_rate=sampling_rate)

    @classmethod
    def rsp_rsa_process(cls, ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame,
                        index: Optional[str] = None, index_name: Optional[str] = None,
                        sampling_rate: Optional[int] = 256, return_mean: Optional[bool] = True):
        rsp_rate = dict.fromkeys(_edr_methods.keys())
        rsa = dict.fromkeys(_edr_methods.keys())
        for method in _edr_methods.keys():
            rsp_signal = cls.ecg_extract_edr(ecg_signal, rpeaks, sampling_rate, edr_type=method)
            rsp_rate[method] = cls.rsp_compute_rate(rsp_signal, sampling_rate)
            rsa[method] = cls.rsa_process(ecg_signal, rsp_signal, sampling_rate)

        if return_mean:
            mean_resp_rate = np.mean(list(rsp_rate.values()))
            rsa = list(rsa.values())
            mean_rsa = {k: np.mean([t[k] for t in rsa]) for k in rsa[0]}
            mean_rsa['RSP_Rate'] = mean_resp_rate
            if not index:
                index = "0"
                index_name = "Index"
            df_rsa = pd.DataFrame(mean_rsa, index=[index])
            df_rsa.index.name = index_name
            return df_rsa
        else:
            df_rsa = pd.DataFrame(rsa).T
            df_rsa['RSP_Rate'] = rsp_rate.values()
            df_rsa.index.name = "Method"
            if index:
                return pd.concat([df_rsa], keys=[index], names=[index_name])
            return df_rsa


def _edr_peak_trough_mean(ecg: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray):
    peak_vals = np.array(ecg.iloc[peaks])
    trough_vals = np.array(ecg.iloc[troughs])
    return np.mean([peak_vals, trough_vals], axis=0)


def _edr_peak_trough_diff(ecg: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray):
    peak_vals = np.array(ecg.iloc[peaks])
    trough_vals = np.array(ecg.iloc[troughs])
    return peak_vals - trough_vals


def _edr_peak_peak_interval(ecg: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray):
    peak_interval = np.ediff1d(peaks, to_begin=0)
    peak_interval[0] = peak_interval.mean()
    return peak_interval


def _remove_outlier_and_interpolate(data: np.ndarray, outlier_mask: np.ndarray, x_old: np.ndarray, desired_length: int):
    data[outlier_mask] = np.nan
    data = pd.Series(data).interpolate(limit_direction='both').values
    return nk.signal_interpolate(x_old, data, desired_length=desired_length, method='linear')


def _check_contains_trough(start_end: np.ndarray, minima: np.ndarray):
    start, end = start_end
    return minima[(minima > start) & (minima < end)].shape[0] == 1


def _rsp_rate(extrema: np.ndarray, sampling_rate: int, desired_length: int) -> np.ndarray:
    rsp_rate_raw = (sampling_rate * 60) / np.ediff1d(extrema)
    return nk.signal_interpolate(extrema[:-1], rsp_rate_raw, desired_length, method='linear')


_edr_methods = {
    'peak_trough_mean': _edr_peak_trough_mean,
    'peak_trough_diff': _edr_peak_trough_diff,
    'peak_peak_interval': _edr_peak_peak_interval
}
