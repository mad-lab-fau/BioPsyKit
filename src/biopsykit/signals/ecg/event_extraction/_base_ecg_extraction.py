import pandas as pd

__all__ = ["BaseEcgExtraction"]

from biopsykit.signals._base_extraction import BaseExtraction


class BaseEcgExtraction(BaseExtraction):
    def extract(
        self,
        *,
        ecg: pd.Series,
        heartbeats: pd.DataFrame,
        sampling_rate_hz: float,
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
