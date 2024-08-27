from typing import Optional

import pandas as pd
from tpcp import Algorithm

from biopsykit.signals._base_extraction import EXTRACTION_HANDLING_BEHAVIOR


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
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
