from typing import Union, Optional, Tuple, Dict

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal, interpolate


def sanitize_input_1d(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
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


def sanitize_input_nd(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    ncols: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """
    Converts nD array-like data (numpy array, pandas dataframe/series) to a numpy array.

    Parameters
    ----------
    data : array_like
        input data
    ncols : int or tuple of ints
        number of columns (2nd dimension) the 'data' array should have

    Returns
    -------
    array_like
        data as numpy array
    """

    # ensure tuple
    if isinstance(ncols, int):
        ncols = (ncols,)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if data.ndim == 1:
        if 1 in ncols:
            return data
        else:
            raise ValueError(
                "Invalid number of columns! Expected one of {}, got 1.".format(ncols)
            )
    elif data.shape[1] not in ncols:
        raise ValueError(
            "Invalid number of columns! Expected one of {}, got {}.".format(
                ncols, data.shape[1]
            )
        )
    return data


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
        if no dataframe or series is passed, or if the dataframe/series has no datetime index

    """

    from scipy import interpolate

    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    elif isinstance(data, pd.Series):
        column_name = [data.name]
    else:
        raise ValueError("Only 'pd.DataFrame' or 'pd.Series' allowed as input!")

    if isinstance(data.index, pd.DatetimeIndex):
        x_old = np.array((data.index - data.index[0]).total_seconds())
    else:
        x_old = np.array(data.index - data.index[0])
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = sanitize_input_1d(data)
    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    x_new = pd.Index(x_new, name="Time")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=column_name)


def interpolate_dict_sec(
    data_dict: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Interpolates all data in the dictionary to 1Hz data (see `interpolate_sec` for further information).

    Parameters
    ----------
    data_dict : dict
        nested data dict with heart rate data

    Returns
    -------
    dict
        nested data dict with heart rate data interpolated to seconds
    """

    result_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            result_dict[key] = interpolate_sec(value)
        elif isinstance(value, dict):
            result_dict[key] = interpolate_dict_sec(value)
        else:
            raise ValueError("Invalid input format!")
    return result_dict


def interpolate_and_cut(
    data_dict: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_dict = interpolate_dict_sec(data_dict)

    durations = np.array(
        [[len(df) for phase, df in data.items()] for data in data_dict.values()]
    )
    value_types = np.array([isinstance(value, dict) for value in data_dict.values()])

    if value_types.all():
        # all values are dictionaries
        phase_names = np.array(
            [np.array(list(val.keys())) for key, val in data_dict.items()]
        )
        if (phase_names[0] == phase_names).all():
            phase_names = phase_names[0]
        else:
            raise ValueError("Phases are not the same for all subjects!")
        # minimal duration of each Phase
        min_dur = {
            phase: dur for phase, dur in zip(phase_names, np.min(durations, axis=0))
        }
        for key, value in data_dict.items():
            dict_cut = {}
            for phase in phase_names:
                dict_cut[phase] = value[phase][0 : min_dur[phase]]
            data_dict[key] = dict_cut
    else:
        min_dur = np.min(durations)
        for key, value in data_dict.items():
            data_dict[key] = value[0:min_dur]
    return data_dict


def find_extrema_in_radius(
    data: Union[pd.DataFrame, pd.Series, np.ndarray],
    indices: Union[pd.DataFrame, pd.Series, np.ndarray],
    radius: Union[int, Tuple[int, int]],
    extrema_type: Optional[str] = "min",
) -> np.ndarray:
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

    Examples
    --------
    >>> from biopsykit.utils.array_handling import find_extrema_in_radius
    >>> data = pd.read_csv("data.csv")
    >>> indices = np.array([16, 25, 40, 57, 86, 100])
    >>>
    >>> radius = 4
    >>> # search minima in 'data' in a 4 sample 'radius' around each entry of 'indices'
    >>> find_extrema_in_radius(data, indices, radius)
    >>>
    >>> radius = (5, 0)
    >>> # search maxima in 'data' in a 5 samples before each entry of 'indices'
    >>> find_extrema_in_radius(data, indices, radius, extrema_type='max')
    """
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}

    if extrema_type not in extrema_funcs:
        raise ValueError(
            "`extrema_type` must be one of {}, not {}".format(
                list(extrema_funcs.keys()), extrema_type
            )
        )
    extrema_func = extrema_funcs[extrema_type]

    # ensure numpy
    data = sanitize_input_1d(data)
    indices = sanitize_input_1d(indices)
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
        windows[i] = data[
            index
            - lower_limit
            + start_padding : index
            + upper_limit
            + start_padding
            + 1
        ]

    return extrema_func(windows, axis=1) + indices - lower_limit


def remove_outlier_and_interpolate(
    data: np.ndarray,
    outlier_mask: np.ndarray,
    x_old: Optional[np.ndarray] = None,
    desired_length: Optional[int] = None,
) -> np.ndarray:
    """
    Sets all detected outlier to nan, imputes them by linear interpolation their neighbors and interpolates
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
    data = pd.Series(data).interpolate(limit_direction="both").values
    # interpolate signal
    x_new = np.linspace(x_old[0], x_old[-1], desired_length)
    return nk.signal_interpolate(x_old, data, x_new, method="linear")


def sliding_window(
    data: Union[np.array, pd.Series, pd.DataFrame],
    window_samples: Optional[int] = None,
    window_sec: Optional[int] = None,
    sampling_rate: Optional[Union[int, float]] = 0,
    overlap_samples: Optional[int] = None,
    overlap_percent: Optional[float] = None,
):
    # check input
    data = sanitize_input_nd(data, ncols=(1, 3))

    window, overlap = sanitize_sliding_window_input(
        window_samples=window_samples,
        window_sec=window_sec,
        sampling_rate=sampling_rate,
        overlap_samples=overlap_samples,
        overlap_percent=overlap_percent,
    )

    return sliding_window_view(
        data, window_length=window, overlap=overlap, nan_padding=True
    )


def sanitize_sliding_window_input(
    window_samples: Optional[int] = None,
    window_sec: Optional[int] = None,
    sampling_rate: Optional[Union[int, float]] = 0,
    overlap_samples: Optional[int] = None,
    overlap_percent: Optional[float] = None,
) -> Tuple[int, int]:
    if all([x is None for x in (window_samples, window_sec)]):
        raise ValueError(
            "Either `window_samples` or `window_sec` in combination with "
            "`sampling_rate` must be supplied as parameter!"
        )

    if window_samples is None:
        if sampling_rate == 0:
            raise ValueError(
                "Sampling rate must be specified when `window_sec` is used!"
            )
        window = int(sampling_rate * window_sec)
    else:
        window = int(window_samples)

    if overlap_samples is not None:
        overlap = int(overlap_samples)
    elif overlap_percent is not None:
        overlap = int(overlap_percent * window)
    else:
        overlap = window - 1

    return window, overlap


def downsample(
    data: np.ndarray,
    sampling_rate: Union[int, float],
    final_sampling_rate: Union[int, float],
) -> np.ndarray:
    if (sampling_rate / final_sampling_rate) % 1 == 0:
        return signal.decimate(data, int(sampling_rate / final_sampling_rate), axis=0)
    else:
        # aliasing filter
        b, a = signal.cheby1(
            N=8, rp=0.05, Wn=0.8 / (sampling_rate / final_sampling_rate)
        )
        data_lp = signal.filtfilt(a=a, b=b, x=data)
        # interpolation
        x_old = np.linspace(0, len(data_lp), num=len(data_lp), endpoint=False)
        x_new = np.linspace(
            0,
            len(data_lp),
            num=int(len(data_lp) / (sampling_rate / final_sampling_rate)),
            endpoint=False,
        )
        interpol = interpolate.interp1d(x=x_old, y=data_lp)
        return interpol(x_new)


def sliding_window_view(
    arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False
) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
       This function will return by default a view onto your input array, modifying values in your result will directly
       affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
       fraction of input may not be returned! However, if `nan_padding` is enabled, this will always return a copy
       instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.

    window_length : int
        length of desired window (must be smaller than array length n)

    overlap : int
        length of desired overlap (must be smaller than window_length)

    nan_padding: bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    windowed view (or copy for nan_padding) of input array as specified, last window might be nan padded if necessary to
    match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 5, overlap = 3, nan_padding = True)
    >>> windowed_view
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.,  8.],
           [ 6.,  7.,  8.,  9., nan]])

    """
    if overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(arr) - window_length) / (window_length - overlap)).astype(
        int
    )
    pad_length = window_length + n_windows * (window_length - overlap) - len(arr)

    # had to handle 1D arrays separately
    if arr.ndim == 1:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(
                arr.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan
            )

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[
        0 :: (window_length - overlap)
    ]

    view = np.squeeze(
        view
    )  # get rid of single-dimensional entries from the shape of an array.

    return view


def _bool_fill(
    indices: np.ndarray, bool_values: np.ndarray, array: np.ndarray
) -> np.ndarray:
    """Fill a preallocated array with bool_values.

    This method iterates over the indices and adds the values to the array at the given indices using a logical or.
    """
    for i in range(len(indices)):  # noqa: consider-using-enumerate
        index = indices[i]
        val = bool_values[i]
        index = index[~np.isnan(index)]
        # perform logical or operation to combine all overlapping window results
        array[index] = np.logical_or(array[index], val)
    return array


def bool_array_to_start_end_array(bool_array: np.ndarray) -> np.ndarray:
    """Find regions in bool array and convert those to start-end indices.

    The end index is exclusive!

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0,0,1,1,0,0,1,1,1])
    >>> start_end_list = bool_array_to_start_end_array(example_array)
    >>> start_end_list
    array([[2, 4],
           [6, 9]])
    >>> example_array[start_end_list[0, 0]: start_end_list[0, 1]]
    array([1, 1])

    """
    # check if input is actually a boolean array
    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array!")

    slices = np.ma.flatnotmasked_contiguous(np.ma.masked_equal(bool_array, 0))
    return np.array([[s.start, s.stop] for s in slices])


def split_array_equally(data: pd.DataFrame, n_splits: int):
    idx_split = np.arange(0, n_splits + 1) * ((len(data) - 1) // n_splits)
    split_boundaries = list(zip(idx_split[:-1], idx_split[1:]))
    return split_boundaries
