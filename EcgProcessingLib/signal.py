from typing import Union, Tuple

import pandas as pd
import numpy as np
from numba import njit


def interpolate_sec(df: pd.DataFrame) -> pd.DataFrame:
    from scipy import interpolate
    x_old = np.array((df.index - df.index[0]).total_seconds())
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    interpol_f = interpolate.interp1d(x=x_old, y=df['ECG_Rate'], fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=df.columns)


# @njit(parallel=True)
def find_extrema_in_radius(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                           indices: Union[pd.DataFrame, pd.Series, np.ndarray], radius: Union[int, Tuple[int]],
                           extrema_type="min"):
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}

    if extrema_type not in extrema_funcs:
        raise ValueError("`extrema_type` must be one of {}, not {}".format(list(extrema_funcs.keys()), extrema_type))
    extrema_func = extrema_funcs[extrema_type]

    # ensure numpy
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.squeeze(data.values)
    if isinstance(indices, (pd.Series, pd.DataFrame)):
        indices = np.squeeze(indices.values)
    # Search region is twice the radius centered around each index
    data = data.astype(float)
    indices = indices.astype(int)
    start_padding = 0

    # pad end/start of array if last_index+radius/first_index-radius is longer/shorter than array
    if isinstance(radius, tuple):
        upper_limit = radius[-1]
    else:
        upper_limit = radius
    if isinstance(radius, tuple):
        lower_limit = radius[0]
    else:
        lower_limit = radius

    if len(data) - np.max(indices) <= upper_limit:
        data = np.pad(data, (0, upper_limit), constant_values=np.nan)
    if np.min(indices) < lower_limit:
        start_padding = lower_limit
        data = np.pad(data, (lower_limit, 0), constant_values=np.nan)

    windows = np.zeros(shape=(len(indices), lower_limit + upper_limit + 1))
    for i, index in enumerate(indices):
        windows[i] = data[index - lower_limit + start_padding:index + upper_limit + start_padding + 1]

    return extrema_func(windows, axis=1) + indices - lower_limit
