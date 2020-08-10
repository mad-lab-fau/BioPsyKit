from typing import Optional, Dict, Tuple, Union, Sequence, Callable

import EcgProcessingLib.utils as utils
import EcgProcessingLib.signal as signal
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal as ss
from NilsPodLib import Dataset
from tqdm.notebook import tqdm
import pytz


class EcgProcessor:
    """
    Class for ECG processing pipeline. Simply pass a pandas dataframe, a dictionary of pandas dataframes,
    or a Dataset object from NilsPodLib and start processing ECG data.

    Each instance has four important attributes:
        self.data_dict: Dictionary with raw data, split into the specified subphases.
        If data is not split the dictionary only has one entry, accessible by the key 'Data'

        self.ecg_result: Dictionary with ECG processing results
        Columns:

        * ECG_Raw: Raw ECG signal
        * ECG_Clean: Cleaned (filtered) ECG signal
        * ECG_Quality: Quality indicator [0,1] for ECG signal quality
        * ECG_R_Peaks: 1.0 where R peak was detected in the ECG signal, 0.0 else
        * R_Peak_Outlier: 1.0 when a detected R peak was classified as outlier, 0.0 else
        * ECG_Rate: Computed Heart rate interpolated to signal length

        self.heart_rate: Dictionary with heart rate data derived from the ECG signal
        Columns:

        * ECG_Rate: Computed heart rate for each detected R peak

        self.rpeaks: Dictionary with R peak location indices derived from the ECG signal
        Columns:

        * R_Peak_Quality: Quality indicator [0,1] for signal quality
        * R_Peak_Idx: Index of detected R peak in the raw ECG signal
        * RR_Interval: Interval between the current and the successive R peak in seconds
        * R_Peak_Outlier: 1.0 when a detected R peak was classified as outlier, 0.0 else
    """

    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, dataset: Optional[Dataset] = None,
                 df: Optional[pd.DataFrame] = None,
                 sampling_rate: Optional[float] = 256.0, timezone: Optional[Union[pytz.timezone, str]] = utils.tz):
        """
        Initializes an `EcgProcessor` instance that can be used for ECG processing.

        :param data_dict: Dictionary with pandas dataframes containing NilsPod data
        :param dataset: NilsPodLib.Dataset
        :param df: pandas dataframe with NilsPod data
        :param sampling_rate: sampling rate of the data (not necessary if `dataset` is passed,
        then it is inferred from the dataset header)
        :param timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')
        """

        if all([i is None for i in [dataset, df, data_dict]]):
            raise ValueError("Either 'dataset', 'df', or 'data_dict' must be specified as parameter!")

        self.sampling_rate: int = int(sampling_rate)
        if isinstance(timezone, str):
            # convert to pytz object
            timezone = pytz.timezone(timezone)

        if data_dict:
            self.data_dict = data_dict
        elif dataset:
            # convert dataset to dataframe and localize timestamp
            df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utils.utc).tz_convert(tz=timezone)
            self.sampling_rate = int(dataset.info.sampling_rate_hz)
            self.data_dict: Dict = {
                'Data': df
            }
        else:
            # localize dataframe
            df = df.tz_localize(tz=utils.utc).tz_convert(tz=timezone)
            self.data_dict: Dict = {
                'Data': df
            }
        self.ecg_result: Dict[str, pd.DataFrame] = {}
        self.heart_rate: Dict[str, pd.DataFrame] = {}
        self.rpeaks: Dict[str, pd.DataFrame] = {}

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the filtered ECG signal.

        :return: Dictionary with filtered ECG signal per sub-phase.
        """
        return {k: pd.DataFrame(v['ECG_Clean']) for k, v in self.ecg_result.items()}

    @property
    def hr_result(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the heart rate computation result from ECG R peak detection.

        :return: Dictionary with heart rate per sub-phase.
        """
        return self.heart_rate

    @classmethod
    def outlier_corrections(cls) -> Sequence[str]:
        """
        Returns the keys of all possible outlier correction methods.

        Currently available outlier correction methods are:

        * `correlation`: Computes the cross-correlation coefficient between every single beat and the average of all detected beats. Marks beats as outlier if cross-correlation coefficient is below a certain threshold
        * `quality`: Uses the 'ECG_Quality' indicator from neurokit to assess signal quality. Marks beats as outlier if quality indicator of beat is below a certain threshold
        * `artifact`: Artifact detection based on work from `Berntson et al. (1990), Psychophysiology`
        * `physiological`: Physiological outlier removal. Marks beats as outlier if their heart rate is above or below a threshold that can not be achieved physiologically
        * `statistical`: Statistical outlier removal. Marks beats as outlier if they are within the xx% highest or lowest heart rates. Values are removed based on the z-score (e.g. 1.96 => 5%, 2.5% highest, 2.5% lowest values)

        :return: List containing the keys of all possible outlier correction methods
        """
        return list(_outlier_correction_methods.keys())

    @classmethod
    def outlier_params_default(cls) -> Dict[str, Union[float, Sequence[float]]]:
        """
        Returns all default parameters for outlier correction methods.

        **NOTE:** outlier correction method `artifact` has no threshold, but '0.0' is default parameter to provide
        a homogenous interface

        :return: Dictionary containing the default parameters for the different outlier correction methods
        """
        return _outlier_correction_params_default

    def ecg_process(self, outlier_correction: Optional[Union[str, None, Sequence[str]]] = 'all',
                    outlier_params: Optional[Union[str, Dict[str, Union[float, Sequence[float]]]]] = 'default',
                    title: Optional[str] = None, method: Optional[str] = "neurokit") -> None:
        """
        Processes the ECG signal and optionally performs outlier correction (see `correct_outlier`).

        :param outlier_correction: List containing the outlier correction methods to be applied.
        Pass 'None' to not apply any outlier correction, 'all' to apply all available outlier correction methods.
        See `EcgProcessor.outlier_corrections` to get a list of possible outlier correction. Default: 'all'
        :param outlier_params: Dictionary of parameters to be passed to the outlier correction methods.
        See `EcgProcessor.outlier_params_default` for the default parameters. Default: 'default'
        :param title: optional title of the bar showing processing progress in Jupyter Notebooks
        :param method: method for cleaning the ECG signal and R peak detection as defined by 'neurokit'.
        Default: 'neurokit'
        """

        for name, df in tqdm(self.data_dict.items(), desc=title):
            ecg_result, rpeaks = self._ecg_process(df, method=method)
            ecg_result, rpeaks = self.correct_outlier(ecg_result, rpeaks, self.sampling_rate, outlier_correction,
                                                      outlier_params)
            heart_rate = pd.DataFrame({'ECG_Rate': 60 / rpeaks['RR_Interval']})
            heart_rate_interpolated = nk.signal_interpolate(rpeaks['R_Peak_Idx'], heart_rate['ECG_Rate'],
                                                            desired_length=len(ecg_result['ECG_Clean']))
            ecg_result['ECG_Rate'] = heart_rate_interpolated
            self.ecg_result[name] = ecg_result
            self.heart_rate[name] = heart_rate
            self.rpeaks[name] = rpeaks

    def _ecg_process(self, data: pd.DataFrame, method: Optional[str] = "neurokit") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Private method to perform the actual ECG processing.

        :param data: ECG data as pandas dataframe. Needs to have one column named 'ecg'
        :param method: method for cleaning the ECG signal and R peak detection as defined by 'neurokit'.
        Default: 'neurokit'
        :return: A tuple of two dataframes: one containing the processed ECG signal, one containing the detected R peaks
        """
        # get numpy
        ecg_signal = data['ecg'].values
        # clean (i.e. filter) the ECG signal using the specified method
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate, method=method)

        # find peaks using the specified method
        # instant_peaks: array indicating where detected R peaks are in the raw ECG signal
        # rpeak_index array containing the indices of detected R peaks
        instant_peaks, rpeak_idx = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate, method=method)
        rpeak_idx = rpeak_idx['ECG_R_Peaks']
        instant_peaks = np.squeeze(instant_peaks.values)

        # compute quality indicator
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeak_idx, sampling_rate=self.sampling_rate)

        # construct new dataframe
        ecg_signal_output = pd.DataFrame(
            {"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Quality": quality, "ECG_R_Peaks": instant_peaks,
             'R_Peak_Outlier': np.zeros(len(data))},
            index=data.index)

        # copy new dataframe consisting of R peaks indices (and their respective quality indicator)
        rpeaks = ecg_signal_output.loc[ecg_signal_output['ECG_R_Peaks'] == 1.0, ['ECG_Quality']]
        rpeaks.rename(columns={'ECG_Quality': 'R_Peak_Quality'}, inplace=True)
        rpeaks.loc[:, 'R_Peak_Idx'] = rpeak_idx
        # compute RR interval
        rpeaks['RR_Interval'] = np.ediff1d(rpeaks['R_Peak_Idx'], to_end=0) / self.sampling_rate
        # ensure equal length by filling the last value with the average RR interval
        rpeaks.loc[rpeaks.index[-1], 'RR_Interval'] = rpeaks['RR_Interval'].mean()

        return ecg_signal_output, rpeaks

    @classmethod
    def correct_outlier(cls, ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame,
                        sampling_rate: Optional[int] = 256,
                        outlier_correction: Optional[Union[str, None, Sequence[str]]] = 'all',
                        outlier_params: Optional[Union[str, Dict[str, Union[float, Sequence[float]]]]] = 'default'
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs outlier correction of the detected R peaks.

        Different methods for outlier detection are available (see `EcgProcessor.outlier_corrections()` to get a list
        of possible outlier correction methods). All outlier methods work independently on the detected R peaks,
        the results will be combined by a logical 'or'. RR intervals classified as outlier will be removed and imputed
        using linear interpolation.

        :param ecg_signal: dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
        :param rpeaks: dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
        :param sampling_rate: Sampling rate of recorded data
        :param outlier_correction: List containing the outlier correction methods to be applied.
        Pass 'None' to not apply any outlier correction, 'all' to apply all available outlier correction methods.
        See `EcgProcessor.outlier_corrections` to get a list of possible outlier correction methods. Default: 'all'
        :param outlier_params: Dictionary of parameters to be passed to the outlier correction methods.
        `EcgProcessor.outlier_params_default` to see the default parameters. Default: 'default'

        :return: A tuple of two dataframes: one containing the processed ECG signal, one containing the detected R peaks

        """

        if outlier_correction == 'all':
            outlier_correction = list(_outlier_correction_methods.keys())
        elif outlier_correction in ['None', None]:
            outlier_correction = list()

        try:
            outlier_funcs: Dict[str, Callable] = {key: _outlier_correction_methods[key] for key in outlier_correction}
        except KeyError:
            raise ValueError(
                "`outlier_correction` may only contain values from {}, None or `all`, not `{}`.".format(
                    list(_outlier_correction_methods.keys()), outlier_correction))

        if outlier_params in ['default', None]:
            outlier_params = {key: _outlier_correction_params_default[key] for key in outlier_funcs}

        # get outlier params (values not passed as arguments will be filled with default arguments)
        outlier_params = {
            key: outlier_params[key] if key in outlier_params else _outlier_correction_params_default[key] for key
            in outlier_funcs}

        # copy dataframe to mark removed beats later
        rpeaks_copy = rpeaks.copy()
        # get the last index because it will get lost when computing the RR interval
        last_idx = rpeaks.iloc[-1]

        # initialize bool mask to mask outlier and add outlier column to rpeaks dataframe
        bool_mask = np.full(rpeaks.shape[0], False)
        rpeaks['R_Peak_Outlier'] = 0.0

        for key in outlier_funcs:
            bool_mask = outlier_funcs[key](ecg_signal, rpeaks, sampling_rate, bool_mask, outlier_params[key])

        # mark all removed beats as outlier in the ECG dataframe
        rpeaks[bool_mask] = None
        removed_beats = rpeaks_copy['R_Peak_Idx'][rpeaks['R_Peak_Idx'].isna()]
        # mark all outlier with 1.0 in the column R_Peak_Outlier column
        rpeaks.fillna({'R_Peak_Outlier': 1.0}, inplace=True)
        # also mark outlier in the ECG signal dataframe
        ecg_signal.loc[removed_beats.index, 'R_Peak_Outlier'] = 1.0

        # interpolate the removed beats
        rpeaks.loc[rpeaks.index[-1]] = [rpeaks['R_Peak_Quality'].mean(), last_idx['R_Peak_Idx'],
                                        rpeaks['RR_Interval'].mean(), 0.0]
        rpeaks.interpolate(method='linear', limit_direction='both', inplace=True)
        # drop duplicate R peaks (can happen during outlier correction at edge cases)
        rpeaks.drop_duplicates(subset='R_Peak_Idx', inplace=True)

        return ecg_signal, rpeaks

    @classmethod
    def correct_rpeaks(cls, ecg_processor: Optional['EcgProcessor'] = None, key: Optional[str] = None,
                       ecg_signal: Optional[pd.DataFrame] = None,
                       rpeaks: Optional[pd.DataFrame] = None,
                       sampling_rate: Optional[int] = 256) -> pd.DataFrame:
        """
        Performs an R peak correction algorithms to get less noisy HRV parameters. R peak detection comes from Neurokit
        and is based on an algorithm by `Lipponen et al. (2019), Journal of medical engineering & technology`.

        To use this function, either simply pass an `EcgProcessor` object together with a key indicating which subphase


        **NOTE** This algorithm might add additional R peaks or remove certain R peaks, so results of this function
        might not match with the R peaks of `EcgProcessor.rpeaks` or might not be used in combination with
        `EcgProcessor.ecg` since indices won't match.


        In this library it is **not** generally applied to the detected R peaks but only used right before passing
        R peaks to `EcgProcessor.hrv_process()`.

        :param ecg_processor: `EcgProcessor` object. If this argument is passed, the `key` argument needs to be
        supplied as well
        :param key: Dictionary key of the subphase to process. Needed when `ecg_processor` is passed as argument
        :param ecg_signal: dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
        :param rpeaks: dataframe with R peaks. Outout of `EcgProcessor.ecg_process()`
        :param sampling_rate: Sampling rate of the recording


        :return: dataframe containing corrected R peak indices
        """

        utils.check_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            rpeaks = ecg_processor.rpeaks[key]
            ecg_signal = ecg_processor.ecg_result[key]
            sampling_rate = ecg_processor.sampling_rate

        # fill missing RR intervals with interpolated R Peak Locations
        rpeaks_corrected = (rpeaks['RR_Interval'].cumsum() * sampling_rate).astype(int)
        rpeaks_corrected = np.append(rpeaks['R_Peak_Idx'].iloc[0], rpeaks_corrected[:-1] + rpeaks['R_Peak_Idx'].iloc[0])
        artifacts, rpeaks_corrected = nk.signal_fixpeaks(rpeaks_corrected, sampling_rate, iterative=True)
        rpeaks_corrected = rpeaks_corrected.astype(int)
        return pd.DataFrame(rpeaks_corrected, columns=['R_Peak_Idx'])

    @classmethod
    def hrv_process(cls, ecg_processor: Optional['EcgProcessor'] = None, key: Optional[str] = None,
                    ecg_signal: Optional[pd.DataFrame] = None, rpeaks: Optional[pd.DataFrame] = None,
                    index: Optional[str] = None, index_name: Optional[str] = None,
                    sampling_rate: Optional[int] = 256) -> pd.DataFrame:

        utils.check_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        rpeaks = cls.correct_rpeaks(ecg_signal=ecg_signal, rpeaks=rpeaks, sampling_rate=sampling_rate)
        hrv_time = nk.hrv_time(rpeaks['R_Peak_Idx'], sampling_rate=sampling_rate)
        hrv_nonlinear = nk.hrv_nonlinear(rpeaks['R_Peak_Idx'], sampling_rate=sampling_rate)

        rpeaks = pd.concat([hrv_time, hrv_nonlinear], axis=1)
        if index:
            rpeaks.index = [index]
            rpeaks.index.name = index_name
        return rpeaks

    @classmethod
    def ecg_extract_edr(cls, ecg_processor: Optional['EcgProcessor'] = None, key: Optional[str] = None,
                        ecg_signal: Optional[pd.DataFrame] = None, rpeaks: Optional[pd.DataFrame] = None,
                        sampling_rate: Optional[int] = 256,
                        edr_type: Optional[str] = 'peak_trough_mean') -> pd.DataFrame:
        """Extract ECG-derived respiration."""

        utils.check_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result
            rpeaks = ecg_processor.rpeaks
            sampling_rate = ecg_processor.sampling_rate

        if edr_type not in _edr_methods:
            raise ValueError(
                "`edr_type` must be one of {}, not {}".format(list(_edr_methods.keys()), edr_type))
        edr_func = _edr_methods[edr_type]

        peaks = np.squeeze(rpeaks['R_Peak_Idx'].values)

        troughs = signal.find_extrema_in_radius(ecg_signal['ECG_Clean'], peaks, radius=(int(0.1 * sampling_rate), 0))
        outlier_mask = rpeaks['R_Peak_Outlier'] == 1

        edr_signal_raw = edr_func(ecg_signal['ECG_Clean'], peaks, troughs)
        edr_signal = _remove_outlier_and_interpolate(edr_signal_raw, outlier_mask, peaks, len(ecg_signal))
        edr_signal = nk.signal_filter(edr_signal, sampling_rate=sampling_rate, lowcut=0.1, highcut=0.5, order=10)

        return pd.DataFrame(edr_signal, index=ecg_signal.index, columns=["ECG_Resp"])

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
    def rsp_rsa_process(cls, rpeaks: pd.DataFrame, ecg_signal: pd.DataFrame,
                        index: Optional[str] = None, index_name: Optional[str] = None,
                        sampling_rate: Optional[int] = 256, return_mean: Optional[bool] = True):
        rsp_rate = dict.fromkeys(_edr_methods.keys())
        rsa = dict.fromkeys(_edr_methods.keys())
        for method in _edr_methods.keys():
            rsp_signal = cls.ecg_extract_edr(ecg_signal=ecg_signal, rpeaks=rpeaks, sampling_rate=sampling_rate,
                                             edr_type=method)
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


def _correct_outlier_correlation(ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate: int,
                                 bool_mask: np.array, corr_thres: float) -> np.array:
    # signal outlier
    # segment individual heart beats
    heartbeats = nk.ecg_segment(ecg_signal['ECG_Clean'], rpeaks['R_Peak_Idx'], sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')
    heartbeats = heartbeats.set_index('Index')
    heartbeats = heartbeats.loc[heartbeats.index.intersection(rpeaks['R_Peak_Idx'])].sort_values(by="Label")
    heartbeats = heartbeats[~heartbeats.index.duplicated()]
    heartbeats_pivoted.columns = heartbeats.index

    # compute the average over all heart beats and compute the correlation coefficient between all beats and
    # the average
    mean_beat = heartbeats_pivoted.mean(axis=1)
    heartbeats_pivoted['mean'] = mean_beat
    corr_coeff = heartbeats_pivoted.corr()['mean'].abs().sort_values(ascending=True)
    corr_coeff.drop('mean', inplace=True)
    # compute RR intervals (in seconds) from R Peak Locations
    rpeaks['RR_Interval'] = np.ediff1d(rpeaks['R_Peak_Idx'], to_end=0) / sampling_rate

    # signal outlier: drop all beats that are below a correlation coefficient threshold
    bool_mask = np.logical_or(bool_mask, rpeaks['R_Peak_Idx'].isin(corr_coeff[corr_coeff < corr_thres].index))
    return bool_mask


def _correct_outlier_quality(ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate: int,
                             bool_mask: np.array, quality_thres: float) -> np.array:
    # signal outlier: drop all beats that are below a signal quality threshold
    bool_mask = np.logical_or(bool_mask, rpeaks['R_Peak_Quality'] < quality_thres)
    return bool_mask


def _correct_outlier_statistical(ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate: int,
                                 bool_mask: np.array, thres: float) -> np.array:
    # statistical outlier: remove the x% highest and lowest successive differences
    # (1.96 std = 5% outlier, 2.576 std = 1% outlier)
    diff_rri = np.ediff1d(rpeaks['RR_Interval'], to_end=0)
    z_score = (diff_rri - np.nanmean(diff_rri)) / np.nanstd(diff_rri, ddof=1)

    bool_mask = np.logical_or(bool_mask, np.abs(z_score) > thres)
    return bool_mask


def _correct_outlier_artifact(ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate: int,
                              bool_mask: np.array, art_thres: float) -> np.array:
    from scipy.stats import iqr
    # note: art_thres only needed to have a uniform signature

    # compute artifact-detection criterion based on Berntson et al. (1990), Psychophysiology
    # QD = Quartile Deviation = IQR / 2
    qd = iqr(rpeaks['RR_Interval'], nan_policy='omit') / 2.0
    # MAD = Minimal Artifact Difference
    mad = (rpeaks['RR_Interval'].median() - 2.9 * qd) / 3.0
    # MED = Maximum Expected Difference
    med = 3.32 * qd
    criterion = np.mean([mad, med])
    bool_mask = np.logical_or(bool_mask, np.abs(rpeaks['RR_Interval'] - rpeaks['RR_Interval'].median()) > criterion)
    return bool_mask


def _correct_outlier_physiological(ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate: int,
                                   bool_mask: np.array, hr_thres: Tuple[float, float]) -> np.array:
    # physiological outlier: minimum/maximum heart rate threshold
    bool_mask = np.logical_or(bool_mask, (rpeaks['RR_Interval'] > (60 / hr_thres[0])) | (
            rpeaks['RR_Interval'] < (60 / hr_thres[1])))
    return bool_mask


_edr_methods = {
    'peak_trough_mean': _edr_peak_trough_mean,
    'peak_trough_diff': _edr_peak_trough_diff,
    'peak_peak_interval': _edr_peak_peak_interval
}

_outlier_correction_methods: Dict[str, Callable] = {
    'correlation': _correct_outlier_correlation,
    'quality': _correct_outlier_quality,
    'artifact': _correct_outlier_artifact,
    'physiological': _correct_outlier_physiological,
    'statistical': _correct_outlier_statistical
}

_outlier_correction_params_default: Dict[str, Union[float, Sequence[float]]] = {
    'correlation': 0.3,
    'quality': 0.4,
    'artifact': 0.0,
    'physiological': (45, 200),
    'statistical': 1.96
}
