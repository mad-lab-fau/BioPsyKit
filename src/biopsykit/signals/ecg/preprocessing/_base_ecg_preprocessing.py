import pandas as pd
from tpcp import Algorithm

__all__ = ["BaseEcgPreprocessing"]


class BaseEcgPreprocessing(Algorithm):
    """Base class for ECG preprocessing algorithms."""

    _action_methods = "clean"

    ecg_clean_: pd.DataFrame

    def clean(self, *, ecg: pd.DataFrame, sampling_rate_hz: float):
        """Clean ECG signal.

        This is an abstract method that needs to be implemented in a subclass.

        Parameters
        ----------
        ecg : :class:`~pandas.DataFrame`
            ECG signal
        sampling_rate_hz : float
            Sampling rate of ECG signal in Hz

        Returns
        -------
        self

        Raises
        ------
        NotImplementedError
            If this method is called from the base class

        """
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
