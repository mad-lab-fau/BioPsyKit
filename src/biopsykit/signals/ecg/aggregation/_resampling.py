import numpy as np
import pandas as pd
from scipy import interpolate
from tpcp import Algorithm, Parameter

from biopsykit.utils._types_internal import str_t

__all__ = ["HeartRateResampling"]


class HeartRateResampling(Algorithm):
    _action_methods = ("apply",)

    group_level: Parameter[str_t]
    resample_rate_hz: Parameter[float]
    cut_to_shortest: Parameter[bool]

    output_: pd.DataFrame

    def __init__(self, group_level: str_t, resample_rate_hz: float = 1.0, cut_to_shortest: bool = True):
        self.group_level = group_level
        self.resample_rate_hz = resample_rate_hz
        self.cut_to_shortest = cut_to_shortest

    def apply(self, data: pd.DataFrame):
        if isinstance(self.group_level, str):
            group_levels = [self.group_level]
        else:
            group_levels = list(self.group_level)
        last_level = group_levels[-1]

        data_resample = data.groupby(group_levels, sort=False).apply(lambda df: self._resample_df(df))

        if self.cut_to_shortest:
            data_resample = data_resample.reset_index("time_sec")
            durations = (
                data_resample["time_sec"].groupby(group_levels, sort=False).apply(lambda df: df.iloc[-1] - df.iloc[0])
            )
            min_durations = durations.groupby(last_level, sort=False).min()

            # Cut data to shortest duration per last level group
            cut_data = data_resample.groupby(group_levels, group_keys=False, sort=False).apply(
                lambda df: df[df["time_sec"] <= df["time_sec"].iloc[0] + min_durations.loc[df.name[-1]]]
            )
            cut_data = cut_data.set_index("time_sec", append=True)
            self.output_ = cut_data
        else:
            self.output_ = data_resample
            return self

    @staticmethod
    def _resample_df(data: pd.DataFrame) -> pd.DataFrame:
        x_old = np.array((data["r_peak_time"] - data["r_peak_time"].iloc[0]).dt.total_seconds())
        x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
        interpol_f = interpolate.interp1d(x=x_old, y=data["heart_rate_bpm"], fill_value="extrapolate")
        return pd.DataFrame(interpol_f(x_new), index=pd.Index(x_new, name="time_sec"), columns=["heart_rate_bpm"])
