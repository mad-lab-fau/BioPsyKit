import pandas as pd

__all__ = ["BaseEcgExtraction", "BaseEcgExtractionWithHeartbeats"]

from biopsykit.signals._base_extraction import BaseExtraction


class BaseEcgExtraction(BaseExtraction):
    """Base class for ECG event extraction algorithms."""

    def extract(self, *, ecg: pd.DataFrame, sampling_rate_hz: float):
        """Extract events from ECG signal.

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


class BaseEcgExtractionWithHeartbeats(BaseExtraction):
    """Base class for ECG event extraction algorithms that require segmented heartbeats."""

    def extract(self, *, ecg: pd.DataFrame, heartbeats: pd.DataFrame, sampling_rate_hz: float):
        """Extract events from ECG signal.

        This is an abstract method that needs to be implemented in a subclass.

        Parameters
        ----------
        ecg : :class:`~pandas.DataFrame`
            ECG signal
        heartbeats : :class:`~pandas.DataFrame`
            DataFrame containing segmented heartbeats. Each row contains start, end, and R-peak location (in samples
            from beginning of signal) of that heartbeat, index functions as id of heartbeat
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
