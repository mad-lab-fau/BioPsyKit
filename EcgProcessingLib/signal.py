from typing import Union, Tuple, Optional

import pandas as pd
import numpy as np


def interpolate_sec(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    from scipy import interpolate
    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    elif isinstance(data, pd.Series):
        column_name = [data.name]
    else:
        raise ValueError("Only 'pd.DataFrame' or 'pd.Series' allowed as input!")
    x_old = np.array((data.index - data.index[0]).total_seconds())
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = sanitize_input(data)
    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=column_name)


def sanitize_input(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # only 1D pandas DataFrame allowed
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError("Only 1D DataFrames allowed!")
        data = np.squeeze(data.values)
    return data


# @njit(parallel=True)
def find_extrema_in_radius(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                           indices: Union[pd.DataFrame, pd.Series, np.ndarray], radius: Union[int, Tuple[int, int]],
                           extrema_type="min"):
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}

    if extrema_type not in extrema_funcs:
        raise ValueError("`extrema_type` must be one of {}, not {}".format(list(extrema_funcs.keys()), extrema_type))
    extrema_func = extrema_funcs[extrema_type]

    # ensure numpy
    data = sanitize_input(data)
    indices = sanitize_input(indices)
    indices = indices.astype(int)
    # possible start offset if beginning of array needs to be padded to ensure radius
    start_padding = 0

    if isinstance(radius, tuple):
        upper_limit = radius[-1]
    else:
        upper_limit = radius
    if isinstance(radius, tuple):
        lower_limit = radius[0]
    else:
        lower_limit = radius

    # round up and make sure it's an integer
    lower_limit = np.ceil(lower_limit).astype(int)
    upper_limit = np.ceil(upper_limit).astype(int)

    # pad end/start of array if last_index+radius/first_index-radius is longer/shorter than array
    if len(data) - np.max(indices) <= upper_limit:
        data = np.pad(data, (0, upper_limit), constant_values=np.nan)
    if np.min(indices) < lower_limit:
        start_padding = lower_limit
        data = np.pad(data, (lower_limit, 0), constant_values=np.nan)

    windows = np.zeros(shape=(len(indices), lower_limit + upper_limit + 1))
    for i, index in enumerate(indices):
        windows[i] = data[index - lower_limit + start_padding:index + upper_limit + start_padding + 1]

    return extrema_func(windows, axis=1) + indices - lower_limit
