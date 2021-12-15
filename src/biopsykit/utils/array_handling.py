"""Module providing various functions for low-level handling of array data."""
from typing import List, Optional, Tuple, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import interpolate, signal

from biopsykit.utils._types import arr_t


def sanitize_input_1d(data: arr_t) -> np.ndarray:
    """Convert 1-d array-like data (:class:`~numpy.ndarray`, :class:`~pandas.DataFrame`/:class:`~pandas.Series`) \
    to a numpy array.

    Parameters
    ----------
    data : array_like
        input data. Needs to be 1-d


    Returns
    -------
    :class:`~numpy.ndarray`
        data as 1-d :class:`~numpy.ndarray`

    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # only 1-d pandas DataFrame allowed
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError("Only 1-d dataframes allowed!")
        data = np.squeeze(data.values)

    return data


def sanitize_input_nd(
    data: arr_t,
    ncols: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """Convert n-d array-like data (:class:`~numpy.ndarray`, :class:`~pandas.DataFrame`/:class:`~pandas.Series`) \
    to a numpy array.

    Parameters
    ----------
    data : array_like
        input data
    ncols : int or tuple of ints
        number of columns (2nd dimension) the ``data`` is expected to have, a list of such if ``data``
        can have a set of possible column numbers or ``None`` to allow any number of columns. Default: ``None``


    Returns
    -------
    :class:`~numpy.ndarray`
        data as n-d numpy array

    """
    # ensure list
    if isinstance(ncols, int):
        ncols = [ncols]

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if ncols is not None:
        if data.ndim == 1:
            if 1 in ncols:
                return data
            raise ValueError("Invalid number of columns! Expected one of {}, got 1.".format(ncols))
        if data.shape[1] not in ncols:
            raise ValueError("Invalid number of columns! Expected one of {}, got {}.".format(ncols, data.shape[1]))
    return data


def find_extrema_in_radius(
    data: arr_t,
    indices: arr_t,
    radius: Union[int, Tuple[int, int]],
    extrema_type: Optional[str] = "min",
) -> np.ndarray:
    """Find extrema values (min or max) within a given radius around array indices.

    Parameters
    ----------
    data : array_like
        input data
    indices : array_like
        array with indices for which to search for extrema values around
    radius: int or tuple of int
        radius around ``indices`` to search for extrema:

        * if ``radius`` is an ``int`` then search for extrema equally in both directions in the interval
          [index - radius, index + radius].
        * if ``radius`` is a ``tuple`` then search for extrema in the interval
          [ index - radius[0], index + radius[1] ]

    extrema_type : {'min', 'max'}, optional
        extrema type to be searched for. Default: 'min'

    Returns
    -------
    :class:`~numpy.ndarray`
        numpy array containing the indices of the found extrema values in the given radius around ``indices``.
        Has the same length as ``indices``.

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
        raise ValueError("`extrema_type` must be one of {}, not {}".format(list(extrema_funcs.keys()), extrema_type))
    extrema_func = extrema_funcs[extrema_type]

    # ensure numpy
    data = sanitize_input_1d(data)
    indices = sanitize_input_1d(indices)
    indices = indices.astype(int)
    # possible start offset if beginning of array needs to be padded to ensure radius
    start_padding = 0

    lower_limit, upper_limit = _find_extrema_in_radius_get_limits(radius)

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
        windows[i] = data[index - lower_limit + start_padding : index + upper_limit + start_padding + 1]

    return extrema_func(windows, axis=1) + indices - lower_limit


def _find_extrema_in_radius_get_limits(radius: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
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
    return lower_limit, upper_limit


def remove_outlier_and_interpolate(
    data: arr_t,
    outlier_mask: np.ndarray,
    x_old: Optional[np.ndarray] = None,
    desired_length: Optional[int] = None,
) -> np.ndarray:
    """Remove outliers, impute missing values and optionally interpolate data to desired length.

    Detected outliers are removed from array and imputed by linear interpolation.
    Optionally, the output array can be linearly interpolated to a new length.


    Parameters
    ----------
    data : array_like
        input data
    outlier_mask : :class:`~numpy.ndarray`
        boolean outlier mask. Has to be the same length as ``data``. ``True`` entries indicate outliers.
        If ``outlier_mask`` is not a bool array values will be casted to bool
    x_old : array_like, optional
        x values of the input data to be interpolated or ``None`` if output data should not be interpolated
        to new length. Default: ``None``
    desired_length : int, optional
        desired length of the output signal or ``None`` to keep length of input signal. Default: ``None``


    Returns
    -------
    :class:`~numpy.ndarray`
        data with removed and imputed outliers, optionally interpolated to desired length


    Raises
    ------
    ValueError
        if ``data`` and ``outlier_mask`` don't have the same length or if ``x_old`` is ``None`` when ``desired_length``
        is passed as parameter

    """
    # ensure numpy and ensure that outlier_mask is a boolean array
    data = sanitize_input_nd(data)
    outlier_mask = np.array(outlier_mask).astype(bool)

    if len(data) != len(outlier_mask):
        raise ValueError("'data' and 'outlier_mask' need to have same length!")
    if x_old is None and desired_length:
        raise ValueError("'x_old' must also be passed when 'desired_length' is passed!")
    x_old = np.array(x_old)

    # remove outlier
    data[outlier_mask] = np.nan
    # fill outlier by linear interpolation of neighbors
    data = pd.Series(data).interpolate(limit_direction="both").values
    # interpolate signal
    x_new = np.linspace(x_old[0], x_old[-1], desired_length)
    return nk.signal_interpolate(x_old, data, x_new, method="linear")


def sliding_window(
    data: arr_t,
    window_samples: Optional[int] = None,
    window_sec: Optional[int] = None,
    sampling_rate: Optional[float] = None,
    overlap_samples: Optional[int] = None,
    overlap_percent: Optional[float] = None,
) -> np.ndarray:
    """Create sliding windows from an input array.

    The window size of sliding windows can either be specified in *samples* (``window_samples``)
    or in *seconds* (``window_sec``, together with ``sampling_rate``).

    The overlap of windows can either be specified in *samples* (``overlap_samples``)
    or in *percent* (``overlap_percent``).

    .. note::
        If ``data`` has more than one dimension the sliding window view is applied to the **first** dimension.
        In the 2-d case this would correspond to applying windows along the **rows**.


    Parameters
    ----------
    data : array_like
        input data
    window_samples : int, optional
        window size in samples or ``None`` if window size is specified in seconds + sampling rate. Default: ``None``
    window_sec : int, optional
        window size in seconds or ``None`` if window size is specified in samples. Default: ``None``
    sampling_rate : float, optional
        sampling rate of data in Hz. Only needed if window size is specified in seconds (``window_sec`` parameter).
        Default: ``None``
    overlap_samples : int, optional
        overlap of windows in samples or ``None`` if window overlap is specified in percent. Default: ``None``
    overlap_percent : float, optional
        overlap of windows in percent or ``None`` if window overlap is specified in samples. Default: ``None``


    Returns
    -------
    :class:`~numpy.ndarray`
        sliding windows from input array.


    See Also
    --------
    :func:`~biopsykit.utils.array_handling.sliding_window_view`
        create sliding window of input array. low-level function with less input parameter configuration possibilities

    """
    # check input
    data = sanitize_input_nd(data)

    window, overlap = sanitize_sliding_window_input(
        window_samples=window_samples,
        window_sec=window_sec,
        sampling_rate=sampling_rate,
        overlap_samples=overlap_samples,
        overlap_percent=overlap_percent,
    )

    return sliding_window_view(data, window_length=window, overlap=overlap, nan_padding=True)


def sanitize_sliding_window_input(
    window_samples: Optional[int] = None,
    window_sec: Optional[int] = None,
    sampling_rate: Optional[float] = None,
    overlap_samples: Optional[int] = None,
    overlap_percent: Optional[float] = None,
) -> Tuple[int, int]:
    """Sanitize input parameters for creating sliding windows from array data.

    The window size of sliding windows can either be specified in *samples* (``window_samples``)
    or in *seconds* (``window_sec``, together with ``sampling_rate``).

    The overlap of windows can either be specified in *samples* (``overlap_samples``)
    or in *percent* (``overlap_percent``).


    Parameters
    ----------
    window_samples : int, optional
        window size in samples or ``None`` if window size is specified in seconds + sampling rate. Default: ``None``
    window_sec : int, optional
        window size in seconds or ``None`` if window size is specified in samples. Default: ``None``
    sampling_rate : float, optional
        sampling rate of data in Hz. Only needed if window size is specified in seconds (``window_sec`` parameter).
        Default: ``None``
    overlap_samples : int, optional
        overlap of windows in samples or ``None`` if window overlap is specified in percent. Default: ``None``
    overlap_percent : float, optional
        overlap of windows in percent or ``None`` if window overlap is specified in samples. Default: ``None``


    Returns
    -------
    window : int
        window size in samples
    overlap : int
        window overlap in samples

    """
    if all(x is None for x in (window_samples, window_sec, sampling_rate)):
        raise ValueError(
            "Either 'window_samples', or 'window_sec' in combination with "
            "'sampling_rate' must be supplied as parameter!"
        )

    if window_samples is None:
        if sampling_rate == 0:
            raise ValueError("Sampling rate must be specified when 'window_sec' is used!")
        window = int(sampling_rate * window_sec)
    else:
        window = int(window_samples)

    overlap = _compute_overlap_samples(window, overlap_samples, overlap_percent)

    return window, overlap


def _compute_overlap_samples(
    window: int, overlap_samples: Optional[int] = None, overlap_percent: Optional[float] = None
):
    if overlap_samples is not None:
        overlap = int(overlap_samples)
    elif overlap_percent is not None:
        if overlap_percent > 1:
            overlap_percent /= 100
        overlap = int(overlap_percent * window)
    else:
        overlap = window - 1
    return overlap


def sliding_window_view(array: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
       This function will return by default a view onto your input array, modifying values in your result will directly
       affect your input data which might lead to unexpected behaviour! If padding is disabled (default), last window
       fraction of input may not be returned! However, if `nan_padding` is enabled, this will always return a copy
       instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    array : :class:`~numpy.ndarray` with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.
    window_length : int
        length of desired window (must be smaller than array length n)
    overlap : int
        length of desired overlap (must be smaller than window_length)
    nan_padding : bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    :class:`~numpy.ndarray`
        windowed view (or copy if ``nan_padding`` is ``True``) of input array as specified,
        last window might be nan-padded if necessary to match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(array = data, window_length = 5, overlap = 3, nan_padding = True)
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
    n_windows = np.ceil((len(array) - window_length) / (window_length - overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - overlap) - len(array)

    # had to handle 1D arrays separately
    if array.ndim == 1:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            array = np.pad(array.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (array.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            array = np.pad(array.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan)

        shape = (window_length, array.shape[-1])
        n = np.array(array.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((array.strides, array.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(array, new_shape, new_strides)[0 :: (window_length - overlap)]

    view = np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.

    return view


def downsample(
    data: arr_t,
    fs_in: float,
    fs_out: float,
) -> np.ndarray:
    """Downsample input signal to a new sampling rate.

    If the output sampling rate is a divisor of the input sampling rate, the signal is downsampled using
    :func:`~scipy.signal.decimate`. Otherwise, data is first filtered using an aliasing filter before it is
    downsampled using linear interpolation.


    Parameters
    ----------
    data : :class:`~numpy.ndarray`
        input data
    fs_in : float
        sampling rate of input data in Hz.
    fs_out : float
        sampling rate of output data in Hz


    Returns
    -------
    :class:`~numpy.ndarray`
        output data with new sampling rate

    """
    data = sanitize_input_nd(data)

    if (fs_in / fs_out) % 1 == 0:
        return signal.decimate(data, int(fs_in / fs_out), axis=0)
    # aliasing filter
    b, a = signal.cheby1(N=8, rp=0.05, Wn=0.8 / (fs_in / fs_out))
    data_lp = signal.filtfilt(a=a, b=b, x=data)
    # interpolation
    x_old = np.linspace(0, len(data_lp), num=len(data_lp), endpoint=False)
    x_new = np.linspace(0, len(data_lp), num=int(len(data_lp) / (fs_in / fs_out)), endpoint=False)
    interpol = interpolate.interp1d(x=x_old, y=data_lp)
    return interpol(x_new)


def _bool_fill(indices: np.ndarray, bool_values: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Fill a pre-allocated array with bool values.

    This function iterates over the indices and adds the bool_values to the array at the given
    indices using a logical "or".

    Parameters
    ----------
    indices : :class:`~numpy.ndarray`
        list with indices to add to the array
    bool_values : :class:`~numpy.ndarray`
        bool values to add to the array
    array : :class:`~numpy.ndarray`
        pre-allocated array to fill with bool values

    Returns
    -------
    :class:`~numpy.ndarray`
        array filled with bool values

    """
    for i in range(len(indices)):  # pylint:disable=consider-using-enumerate
        index = indices[i]
        val = bool_values[i]
        index = index[~np.isnan(index)]
        # perform logical or operation to combine all overlapping window results
        array[index] = np.logical_or(array[index], val)
    return array


def bool_array_to_start_end_array(bool_array: np.ndarray) -> np.ndarray:
    """Find regions in bool array and convert those to start-end indices.

    .. note::
        The end index is inclusive!

    Parameters
    ----------
    bool_array : :class:`~numpy.ndarray` with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    :class:`~numpy.ndarray`
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


def split_array_equally(data: arr_t, n_splits: int) -> List[Tuple[int, int]]:
    """Generate indices to split array into parts with equal lengths.

    Parameters
    ----------
    data : array_like
        data to split
    n_splits : int
        number of splits

    Returns
    -------
    list of tuples
        list with start and end indices which will lead to splitting array into parts with equal lengths

    """
    idx_split = np.arange(0, n_splits + 1) * ((len(data) - 1) // n_splits)
    split_boundaries = list(zip(idx_split[:-1], idx_split[1:]))
    return split_boundaries
