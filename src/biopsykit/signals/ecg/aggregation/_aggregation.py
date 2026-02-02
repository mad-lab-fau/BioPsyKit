import numpy as np
import pandas as pd
from tpcp import Algorithm

__all__ = ["HeartRateAggregation"]

from typing_extensions import Self

from biopsykit.utils._types_internal import str_t


class HeartRateAggregation(Algorithm):
    _action_methods = ("apply",)

    AGG_TYPES = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "se": lambda x: np.std(x) / np.sqrt(len(x)),
    }

    group_level: str_t
    agg_type: str_t

    output_: pd.DataFrame

    def __init__(self, group_level: str_t, agg_type: str_t = "mean"):
        self.group_level = group_level
        self.agg_type = agg_type

    def apply(self, data: pd.DataFrame) -> Self:
        if isinstance(self.agg_type, str):
            agg_type = [self.agg_type]
        else:
            agg_type = self.agg_type
        agg_dict = {at: self.AGG_TYPES[at] for at in agg_type}

        out = data.groupby(self.group_level, sort=False).agg(tuple(agg_dict.items()))
        for level in self.group_level:
            out = out.reindex(data.index.get_level_values(level).unique(), level=level)
        self.output_ = out
        return self
