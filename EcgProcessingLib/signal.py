from typing import Union, Tuple, Optional

import pandas as pd
import numpy as np
import neurokit2 as nk


def interpolate_sec(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Interpolates input data to a frequency of 1 Hz.

    *Note*: This function requires the index of the dataframe or series to be a datetime index.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        data to interpolate. Index of data needs to be 'pd.DateTimeIndex'

    Returns
    -------
    pd.DataFrame
        dataframe with data interpolated to seconds


    Raises
    ------
    ValueError
        if no dataframe or series is passed, of if the dataframe/series has no datetime index

    """

    from scipy import interpolate
    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    elif isinstance(data, pd.Series):
        column_name = [data.name]
    else:
        raise ValueError("Only 'pd.DataFrame' or 'pd.Series' allowed as input!")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index of data needs to be 'pd.DateTimeIndex'!")

    x_old = np.array((data.index - data.index[0]).total_seconds())
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = sanitize_input(data)
    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=column_name)


def sanitize_input(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """
    Converts 1D array-like data (numpy array, pandas dataframe/series) to a numpy array.

    Parameters
    ----------
    data : array_like
        input data. Needs to be 1D

    Returns
    -------
    array_like
        data as numpy array

    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # only 1D pandas DataFrame allowed
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError("Only 1D DataFrames allowed!")
        data = np.squeeze(data.values)

    return data


def find_extrema_in_radius(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                           indices: Union[pd.DataFrame, pd.Series, np.ndarray], radius: Union[int, Tuple[int, int]],
                           extrema_type: Optional[str] = "min") -> np.ndarray:
    """
    Finds extrema values (min or max) within a given radius.

    Parameters
    ----------
    data : array_like
        input data
    indices : array_like
        array with indices for which to search for extrema values around
    radius: int or tuple of int
        radius around `indices` to search for extrema. If `radius` is an ``int`` then search for extrema equally
        in both directions in the interval [index - radius, index + radius].
        If `radius` is a ``tuple`` then search for extrema in the interval [ index - radius[0], index + radius[1] ]
    extrema_type : {'min', 'max'}, optional
        extrema type to be searched for. Default: 'min'

    Returns
    -------
    array_like
        array containing the indices of the found extrema values in the given radius around `indices`.
        Has the same length as `indices`.
    """
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

    # determine upper and lower limit
    if isinstance(radius, tuple):
        lower_limit = radius[0]
    else:
        lower_limit = radius
    if isinstance(radius, tuple):
        upper_limit = radius[-1]
    else:
        upper_limit = radius

    # round up and make sure it's an integer
    lower_limit = np.ceil(lower_limit).astype(int)
    upper_limit = np.ceil(upper_limit).astype(int)

    # pad end/start of array if last_index+radius/first_index-radius is longer/shorter than array
    if len(data) - np.max(indices) <= upper_limit:
        data = np.pad(data, (0, upper_limit), constant_values=np.nan)
    if np.min(indices) < lower_limit:
        start_padding = lower_limit
        data = np.pad(data, (lower_limit, 0), constant_values=np.nan)

    # initialize window array
    windows = np.zeros(shape=(len(indices), lower_limit + upper_limit + 1))
    for i, index in enumerate(indices):
        # get windows around index
        windows[i] = data[index - lower_limit + start_padding:index + upper_limit + start_padding + 1]

    return extrema_func(windows, axis=1) + indices - lower_limit


def remove_outlier_and_interpolate(data: np.ndarray, outlier_mask: np.ndarray, x_old: Optional[np.ndarray] = None,
                                   desired_length: Optional[int] = None) -> np.array:
    """
    Sets all detected outlier to nan, imputes them by linearly interpolation their neighbors and interpolates
    the resulting values to a desired length.

    Parameters
    ----------
    data : array_like
        input data
    outlier_mask: array_like
        outlier mask. has to be the same length like `data`. ``True`` entries indicate outliers
    x_old : array_like, optional
        x values of the input data to interpolate or ``None`` if no interpolation should be performed. Default: ``None``
    desired_length : int, optional
        desired length of the output signal or ``None`` to keep input length. Default: ``None``

    Returns
    -------
    np.array
        Outlier-removed and interpolated data

    Raises
    ------
    ValueError
        if `data` and `outlier_mask` don't have the same length or if `x_old` is ``None`` when `desired_length`
        is passed as parameter

    """
    # ensure numpy
    data = np.array(data)
    outlier_mask = np.array(outlier_mask)

    if len(data) != len(outlier_mask):
        raise ValueError("`data` and `outlier_mask` need to have same length!")
    if x_old is None and desired_length:
        raise ValueError("`x_old` must also be passed when `desired_length` is passed!")
    x_old = np.array(x_old)

    # remove outlier
    data[outlier_mask] = np.nan
    # fill outlier by linear interpolation of neighbors
    data = pd.Series(data).interpolate(limit_direction='both').values
    # interpolate signal
    return nk.signal_interpolate(x_old, data, desired_length=desired_length, method='linear')
