import pandas as pd
from tpcp import Algorithm

__all__ = ["BaseHeartbeatSegmentation"]

from biopsykit.utils.dtypes import EcgRawDataFrame


class BaseHeartbeatSegmentation(Algorithm):
    """Base class for all heartbeat segmentation algorithms."""

    _action_methods = "extract"

    # result
    heartbeat_list_: pd.DataFrame

    def extract(
        self,
        *,
        ecg: EcgRawDataFrame,
        sampling_rate_hz: float,
    ):
        """Segment ECG signal into heartbeats.

        Parameters
        ----------
        ecg : EcgRawDataFrame
            ECG data.
        sampling_rate_hz : float
            Sampling rate of the ECG data in Hz.

        """
        raise NotImplementedError("Method 'extract' must be implemented in subclass.")
