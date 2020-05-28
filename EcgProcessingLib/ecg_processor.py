from typing import Optional, Dict

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

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        return {k: pd.DataFrame(v['ECG_Clean']) for k, v in self.ecg_result.items()}

    def ecg_process(self, quality_thres: Optional[float] = 0.2, title: Optional[str] = None) -> None:
        for name, df in tqdm(self.data_dict.items(), desc=title):
            ecg_result = self._ecg_process(df)

            heart_rate: pd.DataFrame = ecg_result.loc[ecg_result['ECG_R_Peaks'] == 1.0, ['ECG_Rate', 'ECG_Quality']]
            heart_rate.loc[heart_rate['ECG_Quality'] < quality_thres] = None
            z_score = (heart_rate['ECG_Rate'] - heart_rate['ECG_Rate'].mean()) / heart_rate['ECG_Rate'].std()
            # 2.576 std = 1% outlier
            heart_rate.loc[np.abs(z_score) > 2.576] = None
            heart_rate.drop('ECG_Quality', axis=1, inplace=True)
            heart_rate.interpolate(method='linear', inplace=True)

            self.ecg_result[name] = ecg_result
            self.heart_rate[name] = heart_rate

    def _ecg_process(self, df: pd.DataFrame) -> pd.DataFrame:
        ecg_signal = df['ecg'].values
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate, method='hamilton')
        instant_peaks, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate, method='hamilton')
        heart_rate = nk.ecg_rate(rpeaks, sampling_rate=self.sampling_rate, desired_length=len(ecg_cleaned))
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeaks['ECG_R_Peaks'], sampling_rate=self.sampling_rate)

        signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Quality": quality,
                                "ECG_Rate": heart_rate}, index=df.index)
        instant_peaks.index = df.index

        return pd.concat([signals, instant_peaks], axis=1)
