"""Module with functions to process IMU data."""
from typing import Union, Optional

import pandas as pd
import numpy as np
from biopsykit.utils._types import arr_t
from biopsykit.utils.datatype_helper import AccDataFrame, GyrDataFrame, ImuDataFrame

from biopsykit.utils.time import utc
from biopsykit.utils.array_handling import sliding_window


def convert_acc_data_to_g(
    data: Union[AccDataFrame, ImuDataFrame], inplace: Optional[bool] = False
) -> Optional[Union[AccDataFrame, ImuDataFrame]]:
    """Convert acceleration data from :math:`m/s^2` to g.

    Parameters
    ----------
    data : :class:`~biopsykit.utils.datatype_helper.AccDataFrame` or \
            :class:`~biopsykit.utils.datatype_helper.ImuDataFrame`
        dataframe containing acceleration data.
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~biopsykit.utils.datatype_helper.AccDataFrame` or :class:`~biopsykit.utils.datatype_helper.ImuDataFrame`
        acceleration data converted to g

    """
    if not inplace:
        data = data.copy()
    acc_cols = data.filter(like="acc").columns
    data.loc[:, acc_cols] = data.loc[:, acc_cols] / 9.81

    if inplace:
        return None
    return data


def sliding_windows_imu(
    data: arr_t,
    window_samples: Optional[int] = None,
    window_sec: Optional[int] = None,
    sampling_rate: Optional[float] = 0,
    overlap_samples: Optional[int] = None,
    overlap_percent: Optional[float] = None,
) -> pd.DataFrame:
    """Create sliding windows from IMU data.

    The window size of sliding windows can either be specified in *samples* (``window_samples``)
    or in *seconds* (``window_sec``, together with ``sampling_rate``).

    The overlap of windows can either be specified in *samples* (``overlap_samples``)
    or in *percent* (``overlap_percent``).

    If the input data has the shape (m, n) the output data will have the following shape:
    (m * window_size, num_windows), where x depends on the number of data sources (acc, gyr, or both) and the
    axes (x, y, z). The number of windows depends on the input length of the array (n), the window size, and the
    window overlap.

    The output data will be a :class:`~pandas.DataFrame` where the windowed values of each axes are concatenated
    to one row. This means that the output data has a multi-level column index with two levels if the input data only
    has one column level (``axis``):  ``axis`` ("x", "y", "z") as the first level and ``samples`` (array index within
    the window) as second level. If the input data has two column levels (``channel`` and ``axis``) the output
    dataframe will have three levels: ``channel``, ``axis``, and ``samples``.


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
    :class:`~pandas.DataFrame`
        sliding windows from input data


    See Also
    --------
    :func:`~biopsykit.utils.array_handling.sliding_window_view`
        create sliding window of input array. low-level function with less input parameter configuration possibilities

    """
    index = None
    index_resample = None
    if isinstance(data, (pd.DataFrame, pd.Series)):
        index = data.index

    data_window = sliding_window(
        data,
        window_samples=window_samples,
        window_sec=window_sec,
        sampling_rate=sampling_rate,
        overlap_samples=overlap_samples,
        overlap_percent=overlap_percent,
    )
    if index is not None:
        index_resample = sliding_window(
            index.values,
            window_samples=window_samples,
            window_sec=window_sec,
            sampling_rate=sampling_rate,
            overlap_samples=overlap_samples,
            overlap_percent=overlap_percent,
        )[:, 0]
        if isinstance(index, pd.DatetimeIndex):
            index_resample = pd.DatetimeIndex(index_resample)
            index_resample = index_resample.tz_localize(utc).tz_convert(index.tzinfo)
        else:
            index_resample = pd.Index(index_resample)
        index_resample = index_resample.set_names(index.names)

    data_window = np.transpose(data_window)
    data_window = {
        axis: pd.DataFrame(np.transpose(data), index=index_resample) for axis, data in zip(data.columns, data_window)
    }
    data_window = pd.concat(data_window, axis=1)
    if data_window.columns.nlevels == 2:
        data_window.columns.names = ["axis", "samples"]
    elif data_window.columns.nlevels == 3:
        data_window.columns.names = ["channel", "axis", "samples"]
    return data_window


def var_norm_windows(data: Union[AccDataFrame, GyrDataFrame]) -> pd.DataFrame:
    r"""Compute the norm of the variance of each axis for a windowed signal.

    This function computes the norm of variance according to:

    .. math::
        var_{norm} = \sqrt{var_x^2 + var_y^2 + var_z^2}

    where :math:`var_i` is the variance of axis :math:`i` in the window

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data, split into windows using :func:`~biopsykit.signals.imu.sliding_windows_imu`

    Returns
    -------
    :class:`~pandas.DataFrame`
        norm of variance for windowed signal

    """
    var = data.groupby(level="axis", axis=1).apply(lambda x: np.var(x, axis=1))
    norm = pd.DataFrame(np.linalg.norm(var, axis=1), index=var.index, columns=["var_norm"])
    return norm
