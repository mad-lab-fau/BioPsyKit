from typing import Optional, Union

import pandas as pd
from tpcp import Algorithm

__all__ = ["BaseHeartbeatSegmentation"]

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS


class BaseHeartbeatSegmentation(Algorithm):
    """Base class for all heartbeat segmentation algorithms."""

    _action_methods = "extract"

    # result
    heartbeat_list_: pd.DataFrame

    def extract(
        self,
        *,
        ecg: Union[pd.Series, pd.DataFrame],
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        """Segment ECG signal into heartbeats."""
        raise NotImplementedError("Method 'extract' must be implemented in subclass.")