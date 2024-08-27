from typing import Optional

import pandas as pd
from tpcp import Algorithm

from biopsykit.signals._base_extraction import BaseExtraction, EXTRACTION_HANDLING_BEHAVIOR


__all__ = ["BaseBPointExtraction"]


class BaseBPointExtraction(Algorithm):

    _action_methods = "extract"

    points_: pd.DataFrame

    def extract(
        self,
        *,
        icg: pd.Series,
        heartbeats: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
