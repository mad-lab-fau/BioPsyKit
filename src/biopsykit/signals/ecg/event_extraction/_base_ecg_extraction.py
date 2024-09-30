import pandas as pd
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
        sampling_rate_hz: float,
    ):
        raise NotImplementedError("This is an abstract method and needs to be implemented in a subclass.")
