import neurokit2 as nk
import pandas as pd
from tpcp import Algorithm, Parameter

__all__ = ["EcgPreprocessingNeurokit"]

from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.dtypes import EcgRawDataFrame, is_ecg_raw_dataframe


class EcgPreprocessingNeurokit(Algorithm):

    _action_methods = "clean"

    method: Parameter[str]

    ecg_clean_: pd.DataFrame

    def __init__(self, method: str = "biosppy"):
        """Initialize ECG preprocessing algorithm.

        Parameters
        ----------
        method : str, optional
            Cleaning method (default is "biosppy"), can be either "neurokit" or "biosppy"

        """
        self.method = method

    def clean(self, *, ecg: EcgRawDataFrame, sampling_rate_hz: int):
        """Clean ECG signal using :func:`~neurokit2.ecg_clean`.

        Parameters
        ----------
        ecg : :class:`~pandas.DataFrame`
            pandas DataFrame containing the raw ECG signal
        sampling_rate_hz : int
            Sampling rate of the ECG signal in Hz

        Returns
        -------
        self

        """
        is_ecg_raw_dataframe(ecg)
        ecg = sanitize_input_dataframe_1d(ecg, column="ecg")
        ecg = ecg.squeeze()
        if ecg.empty:
            self.ecg_clean_ = pd.DataFrame(index=ecg.index, columns=["ecg"])
            return self

        if self.method not in ["neurokit", "biosppy"]:
            raise ValueError("Not implemented yet!")
        clean_signal = nk.ecg_clean(ecg, sampling_rate=sampling_rate_hz, method=self.method)
        self.ecg_clean_ = pd.DataFrame(clean_signal, index=ecg.index, columns=["ecg"])

        return self
