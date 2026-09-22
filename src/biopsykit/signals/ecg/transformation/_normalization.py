import pandas as pd
from tpcp import Algorithm, Parameter

__all__ = ["HeartRateNormalization"]

from typing_extensions import Self

from biopsykit.utils._types_internal import str_t


class HeartRateNormalization(Algorithm):
    _action_methods = ("apply",)

    group_level: Parameter[str_t]
    norm_level: Parameter[str]

    output_: pd.DataFrame

    def __init__(
        self,
        *,
        group_level: str_t,
        norm_level: str,
    ):
        self.group_level = group_level
        self.norm_level = norm_level

    def apply(self, data: pd.DataFrame, *, key: str) -> Self:
        mean_hr = data["heart_rate_bpm"].xs(key, level=self.norm_level).groupby(self.group_level).mean()
        data_norm = (
            data["heart_rate_bpm"]
            .groupby(self.group_level, group_keys=False)
            .apply(lambda df: 100 - 100 * (df / mean_hr.loc[df.name]))
        )
        data_norm = data_norm.to_frame(name="heart_rate_normalized_percent")
        self.output_ = data_norm
        return self
