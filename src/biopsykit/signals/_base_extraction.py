from typing import Literal, Optional

import pandas as pd
from tpcp import Algorithm

HANDLE_MISSING_EVENTS = Literal["raise", "warn", "ignore"]


class BaseExtraction(Algorithm):
    """Base class which defines the interface for all fiducial point extraction algorithms.

    Results:
        points_ : saves positions of extracted points in pd.DataFrame
    """

    _action_methods = "extract"

    # results
    points_: pd.DataFrame

    # interface method
    # @abstractmethod
    # @make_action_safe
    def extract(
        self,
        *,
        signal_clean: pd.Series,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        """Extract specific fiducial points from cleaned signal."""
        raise NotImplementedError("Method 'extract' must be implemented in subclass.")
