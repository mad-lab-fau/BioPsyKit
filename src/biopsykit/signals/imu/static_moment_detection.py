"""A set of util functions to detect static regions in a IMU signal given certain constrains."""
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm
from typing_extensions import Literal

from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import (
    _bool_fill,
    bool_array_to_start_end_array,
    sanitize_input_nd,
    sanitize_sliding_window_input,
    sliding_window_view,
)

# supported metric functions
_METRIC_FUNCTIONS = {
    "maximum": np.nanmax,
    "variance": np.nanvar,
    "mean": np.nanmean,
    "median": np.nanmedian,
}
METRIC_FUNCTION_NAMES = Literal["maximum", "variance", "mean", "median"]


def _find_static_samples(
    data: np.ndarray,
    window_length: int,
    inactive_signal_th: float,
    metric: METRIC_FUNCTION_NAMES = "mean",
    overlap: int = None,
) -> np.ndarray:
    """Search for static samples within given input signal, based on windowed L2-norm thresholding.

    .. warning::
        Due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!


    Parameters
    ----------
    data : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)
    window_length : int
        Length of desired window in units of samples
    inactive_signal_th : float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold
    metric : str, optional
        Metric which will be calculated per window, one of the following strings:

        * 'mean' (default)
          Calculates mean value per window
        * 'maximum'
          Calculates maximum value per window
        * 'median'
          Calculates median value per window
        * 'variance'
          Calculates variance value per window

    overlap : int, optional
        Length of desired overlap in units of samples. If ``None`` (default) overlap will be window_length - 1


    Returns
    -------
    Boolean array with length n to indicate static (=``True``) or non-static (=``False``) for each sample


    Examples
    --------
    >>> _find_static_samples(data, window_length=128, overlap=64, inactive_signal_th = 5, metric = 'mean')

    See Also
    --------
    :func:`~biopsykit.utils.array_handling.sliding_window_view`
        Details on the used windowing function for this method.

    """
    # test for correct input data shape
    if np.shape(data)[-1] != 3:
        raise ValueError("Invalid signal dimensions, signal must be of shape (n,3).")

    if metric not in _METRIC_FUNCTIONS:
        raise ValueError("Invalid metric passed! {} as metric is not supported.".format(metric))

    # add default overlap value
    if overlap is None:
        overlap = window_length - 1

    # allocate output array
    inactive_signal_bool_array = np.zeros(len(data))

    # calculate norm of input signal (do this outside of loop to boost performance at cost of memory!)
    signal_norm = norm(data, axis=1)

    mfunc = _METRIC_FUNCTIONS[metric]

    # Create windowed view of norm
    windowed_norm = sliding_window_view(signal_norm, window_length, overlap, nan_padding=False)
    is_static = np.broadcast_to(mfunc(windowed_norm, axis=1) <= inactive_signal_th, windowed_norm.shape[::-1]).T

    # create the list of indices for sliding windows with overlap
    windowed_indices = sliding_window_view(np.arange(0, len(data)), window_length, overlap, nan_padding=False)

    # iterate over sliding windows
    inactive_signal_bool_array = _bool_fill(windowed_indices, is_static, inactive_signal_bool_array)

    return inactive_signal_bool_array.astype(bool)


def _find_static_sequences(
    data: np.ndarray,
    window_length: int,
    inactive_signal_th: float,
    metric: METRIC_FUNCTION_NAMES = "variance",
    overlap: int = None,
) -> np.ndarray:
    """Search for static sequences within given input signal, based on windowed L2-norm thresholding.

    .. warning::
        Due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    data : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)
    window_length : int
        Length of desired window in units of samples
    inactive_signal_th : float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold
    metric : str, optional
        Metric which will be calculated per window, one of the following strings:
            * 'variance' (default): Calculates variance value per window
            * 'mean': Calculates mean value per window
            * 'maximum': Calculates maximum value per window
            * 'median': Calculates median value per window

    overlap : int, optional
        Length of desired overlap in units of samples. If None (default) overlap will be window_length - 1

    Returns
    -------
    Array of [start, end] labels indication static regions within the input signal

    Examples
    --------
    >>> _find_static_sequences(data, window_length=128, overlap=64, inactive_signal_th = 5, metric = 'mean')

    See Also
    --------
    :func:`~biopsykit.signals.utils.array_handling.sliding_window`
        Details on the used windowing function for this method.

    """
    static_moment_bool_array = _find_static_samples(
        data=data, window_length=window_length, inactive_signal_th=inactive_signal_th, metric=metric, overlap=overlap
    )
    return bool_array_to_start_end_array(static_moment_bool_array)


def find_static_moments(
    data: arr_t,
    threshold: float,
    window_samples: Optional[int] = None,
    window_sec: Optional[int] = None,
    sampling_rate: Optional[Union[int, float]] = 0,
    overlap_samples: Optional[int] = None,
    overlap_percent: Optional[float] = None,
    metric: METRIC_FUNCTION_NAMES = "variance",
) -> pd.DataFrame:
    """Search for static moments within given input signal, based on windowed L2-norm thresholding.

    The window size of sliding windows can either be specified in *samples* (``window_samples``)
    or in *seconds* (``window_sec``, together with ``sampling_rate``).

    The overlap of windows can either be specified in *samples* (``overlap_samples``)
    or in *percent* (``overlap_percent``).


    Parameters
    ----------
    data : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)
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
    threshold : float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold
    metric : str, optional
        Metric which will be calculated per window, one of the following strings:

        * 'variance' (default): Calculates variance value per window
        * 'mean': Calculates mean value per window
        * 'maximum': Calculates maximum value per window
        * 'median': Calculates median value per window


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with ["start", "end"] columns indicating beginning and end of static regions within the input signal


    Examples
    --------
    >>> _find_static_sequences(data, window_length=128, overlap=64, inactive_signal_th = 5, metric = 'mean')


    See Also
    --------
    :func:`~biopsykit.utils.array_handling.sliding_window`
        Details on the used windowing function for this method.

    """
    # compute the data_norm of the variance in the windows
    window, overlap = sanitize_sliding_window_input(
        window_samples=window_samples,
        window_sec=window_sec,
        sampling_rate=sampling_rate,
        overlap_samples=overlap_samples,
        overlap_percent=overlap_percent,
    )
    if data.empty:
        start_end = np.zeros(shape=(0, 2))
    else:
        data = sanitize_input_nd(data)
        start_end = _find_static_sequences(
            data, window_length=window, overlap=overlap, inactive_signal_th=threshold, metric=metric
        )
        if len(start_end) == 0:
            return pd.DataFrame(columns=["start", "end"])
        # end indices are *inclusive*!
        start_end[:, 1] -= 1
    return pd.DataFrame(start_end, columns=["start", "end"])


def find_first_static_window_multi_sensor(
    signals: Sequence[np.ndarray],
    window_length: int,
    inactive_signal_th: float,
    metric: METRIC_FUNCTION_NAMES,
) -> Tuple[int, int]:
    """Find the first time window in the signal where all provided sensors are static.

    Parameters
    ----------
    signals : Sequence of n arrays with shape (k, m) or a 3D-array with shape (k, n, m)
        The signals of n senors with m axis and k samples.
    window_length
        Length of the required static signal in samples
    inactive_signal_th
        The threshold for static windows.
        If metric(norm(window, axis=-1)) <= `inactive_signal_th` for all sensors, it is considered static.
    metric
        The metric that should be calculated on the vectornorm over all axis for each sensor in each window

    Returns
    -------
    (start, end)
        Start and end index of the first static window.

    Examples
    --------
    >>> sensor_1_gyro = ...
    >>> sensor_2_gyro = ...
    >>> find_first_static_window_multi_sensor([sensor_1_gyro, sensor_2_gyro], window_length=128, inactive_signal_th=5)

    """
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError("`metric` must be one of {}".format(list(_METRIC_FUNCTIONS.keys())))

    if not isinstance(signals, np.ndarray):
        # all signals should have the same shape
        if not all(signals[0].shape == signal.shape for signal in signals):
            raise ValueError("All provided signals need to have the same shape.")
        if signals[0].ndim != 2:
            raise ValueError(
                "The array of each sensor must be 2D, where the first dimension is the time and the second dimension "
                "the sensor axis."
            )
        signals = np.hstack(signals)
    else:
        if signals.ndim != 3:
            raise ValueError(
                "If a array is used as input, it must be 3D, where the first dimension is the time, "
                "the second indicates the sensor and the third the axis of the sensor."
            )

    n_signals = signals.shape[1]

    windows = sliding_window_view(
        signals.reshape((signals.shape[0], -1)),
        window_length=window_length,
        overlap=window_length - 1,
        nan_padding=False,
    )
    reshaped_windows = windows.reshape((*windows.shape[:-1], n_signals, -1))
    window_norm = norm(reshaped_windows, axis=-1)

    method = _METRIC_FUNCTIONS[metric]
    # This is pretty wasteful as we calculate the the function on all windows, even though we are only interested in
    # the first, where our threshold is valid.
    window_over_thres = method(window_norm, axis=1).max(axis=-1) <= inactive_signal_th

    valid_windows = np.nonzero(window_over_thres)[0]
    if len(valid_windows) == 0:
        raise ValueError("No static window was found")

    return valid_windows[0], valid_windows[0] + window_length
