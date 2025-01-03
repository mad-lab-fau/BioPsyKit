from typing import Optional

import pandas as pd

from biopsykit.signals._base_extraction import BaseExtraction

__all__ = ["BaseBPointExtraction"]


class BaseBPointExtraction(BaseExtraction):
    def extract(
        self, *, icg: pd.Series, heartbeats: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: Optional[float]
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
