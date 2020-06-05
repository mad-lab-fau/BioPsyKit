from typing import Optional, Dict, Tuple

import EcgProcessingLib.utils as utils
import neurokit2 as nk
import numpy as np
import pandas as pd
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
        self.r_peak_loc: Dict[str, pd.DataFrame] = {}

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        return {k: pd.DataFrame(v['ECG_Clean']) for k, v in self.ecg_result.items()}

    def ecg_process(self, quality_thres: Optional[float] = 0.4, title: Optional[str] = None,
                    method: Optional[str] = "neurokit") -> None:
        for name, df in tqdm(self.data_dict.items(), desc=title):
            ecg_result, rpeak_idx = self._ecg_process(df, method=method)
            df_rpeak: pd.DataFrame = ecg_result.loc[
                ecg_result['ECG_R_Peaks'] == 1.0, ['ECG_Quality']]
            df_rpeak['R_Peak_Idx'] = rpeak_idx
            # ecg_result.drop('ECG_Rate', axis=1, inplace=True)
            df_rpeak = self.remove_outlier(df_rpeak, quality_thres, sampling_rate=self.sampling_rate)
            heart_rate = nk.ecg_rate(df_rpeak['R_Peak_Idx'], sampling_rate=self.sampling_rate,
                                     desired_length=len(ecg_result['ECG_Clean']))
            ecg_result['ECG_Rate'] = heart_rate
            self.ecg_result[name] = ecg_result
            self.heart_rate[name] = pd.DataFrame(nk.ecg_rate(df_rpeak['R_Peak_Idx'], sampling_rate=self.sampling_rate),
                                                 index=df_rpeak.index, columns=['ECG_Rate'])
            self.r_peak_loc[name] = df_rpeak[['R_Peak_Idx']]

    def _ecg_process(self, df: pd.DataFrame, method: Optional[str] = "neurokit") -> Tuple[pd.DataFrame, np.array]:
        ecg_signal = df['ecg'].values
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate, method=method)
        instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate, method=method)
        # heart_rate = nk.ecg_rate(rpeaks, sampling_rate=self.sampling_rate, desired_length=len(ecg_cleaned))
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks['ECG_R_Peaks'], sampling_rate=self.sampling_rate)

        signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Quality": quality},
                               index=df.index)
        instant_peaks.index = df.index
        return pd.concat([signals, instant_peaks], axis=1), rpeaks['ECG_R_Peaks']

    @classmethod
    def remove_outlier(cls, df_rr: pd.DataFrame, quality_thres: Optional[float] = 0.4,
                       sampling_rate: Optional[float] = 256.0,
                       hr_thres: Optional[Tuple[int, int]] = (45, 200)) -> pd.DataFrame:
        # compute RR intervals (in seconds) from R Peak Locations
        df_rr['RR_Interval'] = np.ediff1d(df_rr['R_Peak_Idx'], to_begin=0) / sampling_rate
        df_rr.loc[df_rr['ECG_Quality'] < quality_thres] = None
        z_score = (df_rr['RR_Interval'] - df_rr['RR_Interval'].mean()) / df_rr['RR_Interval'].std()
        # 1.96 std = 5% outlier
        df_rr.loc[np.abs(z_score) > 1.96] = None
        df_rr.loc[(df_rr['RR_Interval'] > (60 / hr_thres[0])) | (df_rr['RR_Interval'] < (60 / hr_thres[1]))] = None
        df_rr.drop('ECG_Quality', axis=1, inplace=True)
        df_rr.interpolate(method='linear', limit_direction='both', inplace=True)

        # convert RR intervals back to R Peak Locations
        df_rr['R_Peak_Idx'] = (df_rr['RR_Interval'].cumsum() * sampling_rate).astype(int)
        return df_rr

    def hrv_process(self, df: pd.DataFrame, index: Optional[str] = None,
                    index_name: Optional[str] = None) -> pd.DataFrame:
        return self._hrv_process(df, self.sampling_rate, index, index_name)

    @classmethod
    def _hrv_process(cls, df: pd.DataFrame, sampling_rate: Optional[int] = 256,
                     index: Optional[str] = None, index_name: Optional[str] = None) -> pd.DataFrame:
        hrv_time = nk.hrv_time(df['R_Peak_Idx'], sampling_rate=sampling_rate)
        hrv_nonlinear = nk.hrv_nonlinear(df['R_Peak_Idx'], sampling_rate=sampling_rate)
        df = pd.concat([hrv_time, hrv_nonlinear], axis=1)
        if index:
            df.index = [index]
            df.index.name = index_name
        return df
