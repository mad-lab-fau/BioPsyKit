from typing import Optional

import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from tpcp import Algorithm

__all__ = ["BaseEcgExtraction"]


class BaseEcgExtraction(Algorithm):
    _action_methods = "extract"

    points_: pd.DataFrame

    def extract(
        self,
        *,
        ecg: pd.Series,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
