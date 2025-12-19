from typing import ClassVar

import neurokit2 as nk
import pandas as pd
from tpcp import Parameter

__all__ = ["EcgPreprocessingNeurokit"]

from biopsykit.signals.ecg.preprocessing import BaseEcgPreprocessing
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.dtypes import EcgRawDataFrame, is_ecg_raw_dataframe


class EcgPreprocessingNeurokit(BaseEcgPreprocessing):
    """ECG preprocessing algorithm using NeuroKit2.

    This class provides an interface to the ECG preprocessing functions of NeuroKit2.

    For more information on the NeuroKit2 library, see [Mak21]_.

    Parameters
    ----------
    method : str, optional
        Cleaning method to use. Options are:
            * "biosppy" (the default): Use the preprocessing parameters provided by
            `biosppy <https://biosppy.readthedocs.io/en/stable/>`_ library for cleaning. It uses an FIR filter
            with cut-off frequencies of [0.67, 45] Hz and order = 1.5 * sampling_rate.
            * "neurokit": Use the `NeuroKit2 <https://neurokit2.readthedocs.io/en/latest/>`_ library for cleaning.

    Attributes
    ----------
    ecg_clean_ : :class:`~pandas.DataFrame`
        Cleaned ECG signal

    References
    ----------
    .. [Mak21] Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lesspinasse, F., Pham, H., Schölzel, C., & S.H. Chen
        (2021). NeuroKit2: A Python Toolbox for Neurophysiological Signal Processing. Behavior Research Methods.
        https://doi.org/10.3758/s13428-020-01516-y

    """

    _action_methods = "clean"

    PRPROCESSING_METHODS: ClassVar[list[str]] = ["biosppy", "neurokit"]

    method: Parameter[str]

    def __init__(self, *, method: str = "biosppy"):
        """Initialize ``EcgPreprocessingNeurokit`` instance.

        Parameters
        ----------
        method : str, optional
            Cleaning method to use. Options are:
                * "biosppy" (the default): Use the preprocessing parameters provided by
                `biosppy <https://biosppy.readthedocs.io/en/stable/>`_ library for cleaning. It uses an FIR filter
                with cut-off frequencies of [0.67, 45] Hz and order = 1.5 * sampling_rate.
                * "neurokit": Use the `NeuroKit2 <https://neurokit2.readthedocs.io/en/latest/>`_ library for cleaning.

        """
        self.method = method

    def clean(self, *, ecg: EcgRawDataFrame, sampling_rate_hz: int):
        """Clean ECG signal using :func:`~neurokit2.ecg_clean`.

        Parameters
        ----------
        ecg : :class:`~pandas.DataFrame`
            ECG signal
        sampling_rate_hz : int
            Sampling rate of the ECG signal in Hz

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the specified method is not implemented yet

        """
        is_ecg_raw_dataframe(ecg)
        ecg = sanitize_input_dataframe_1d(ecg, column="ecg")
        ecg = ecg.squeeze()
        if ecg.empty:
            self.ecg_clean_ = pd.DataFrame(index=ecg.index, columns=["ecg"])
            return self

        if self.method not in self.PRPROCESSING_METHODS:
            raise ValueError(
                f"Preprocessing method not implemented yet! "
                f"Got: {self.method}, must be one of {self.PRPROCESSING_METHODS}."
            )
        clean_signal = nk.ecg_clean(ecg, sampling_rate=sampling_rate_hz, method=self.method)
        self.ecg_clean_ = pd.DataFrame(clean_signal, index=ecg.index, columns=["ecg"])

        return self
