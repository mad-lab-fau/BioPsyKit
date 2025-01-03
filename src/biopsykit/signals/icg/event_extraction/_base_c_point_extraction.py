from typing import Optional

import pandas as pd

from biopsykit.signals._base_extraction import BaseExtraction

__all__ = ["BaseCPointExtraction"]


class BaseCPointExtraction(BaseExtraction):
    def extract(self, *, icg: pd.Series, heartbeats: pd.DataFrame, sampling_rate_hz: Optional[float]):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
