"""Module containing helper functions to handle time data."""

import datetime
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytz

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils._types import path_t

tz = pytz.timezone("Europe/Berlin")
utc = pytz.timezone("UTC")

__all__ = [
    "check_tz_aware",
    "extract_time_from_filename",
    "get_time_from_date",
    "time_to_timedelta",
    "timedelta_to_time",
]


def check_tz_aware(data: pd.DataFrame) -> bool:
    """Check whether dataframe index is timezone-aware.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with index to check


    Returns
    -------
    bool
        ``True`` if index of dataframe is a :class:`~pandas.DatetimeIndex` and index is timezone-aware,
        ``False`` otherwise

    """
    _assert_is_dtype(data, pd.DataFrame)
    return isinstance(data.index, pd.DatetimeIndex) and (data.index.tzinfo is not None)


def extract_time_from_filename(
    file_path: path_t, filename_pattern: str, date_pattern: Optional[str] = None
) -> datetime.datetime:
    """Extract time information from filename.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        filename
    filename_pattern : str
        regex string indicating how to extract time information from filename
    date_pattern : str, optional
        date format pattern or ``None`` to use default date format pattern ("%Y%m%d_%H_%M_%S").
        Default: ``None``

    Returns
    -------
    :class:`datetime.datetime`
        extracted date information

    """
    # ensure pathlib
    file_path = Path(file_path)
    if date_pattern is None:
        date_pattern = "%Y%m%d_%H_%M_%S"
    start_time = re.findall(filename_pattern, file_path.name)
    if len(start_time) == 0:
        raise ValueError(f"No valid string matching the pattern '{filename_pattern}' found in filename!")
    start_time = start_time[0]
    start_time = datetime.datetime.strptime(start_time, date_pattern)
    return pd.to_datetime(start_time)


def get_time_from_date(
    data: pd.Series,
    is_utc: Optional[bool] = False,
    tz_convert: Optional[bool] = False,
    timezone: Optional[Union[str, datetime.tzinfo]] = None,
) -> pd.Series:
    """Extract time information from series with date information.

    Some functions expect only time information (hour, minute, second, ...)
    without date information (year, month, day). This function can be used to extract only the relevant time
    information from the complete datetime data.


    Parameters
    ----------
    data : :class:`~pandas.Series`
        series with date information
    is_utc : bool, optional
        ``True`` if datetime is in UTC, ``False`` otherwise. Default: ``False``
    tz_convert : bool, optional
        ``True`` to convert datetime into correct timezone before extracting time information or
        ``False`` to localize datetime. Default: ``False``
    timezone : str or :class:`datetime.tzinfo`
        timezone the datetime objects are in or should be converted to.
        Default: ``None``, which defaults to time zone "Europe/Berlin"


    Returns
    -------
    :class:`~pandas.Series`
        pandas series with time information extracted from datetime

    """
    _assert_is_dtype(data, pd.Series)
    if timezone is None:
        timezone = tz
    data = pd.to_datetime(data, utc=is_utc)
    if tz_convert or pd.DatetimeIndex(data).tzinfo is not None:
        data = data.dt.tz_convert(timezone)
    else:
        data = data.dt.tz_localize(timezone)

    data = data - data.dt.normalize()
    return data


def time_to_timedelta(data: pd.Series) -> pd.Series:
    """Convert time information in a series into ``datetime.timedelta`` data.

    Parameters
    ----------
    data : :class:`~pandas.Series`
        series with time information


    Returns
    -------
    :class:`~pandas.Series`
        series with data converted into :class:`datetime.timedelta`

    """
    _assert_is_dtype(data, pd.Series)
    if np.issubdtype(data.dtype, np.timedelta64):
        # data is already a timedelta
        return data
    return pd.to_timedelta(data.astype(str))


def timedelta_to_time(data: pd.Series) -> pd.Series:
    """Convert ``datetime.timedelta`` data in a series ``datetime.time`` data.

    Parameters
    ----------
    data : :class:`~pandas.Series`
        series with data as :class:`datetime.timedelta`


    Returns
    -------
    :class:`~pandas.Series`
        series with data converted into :class:`datetime.time`

    """
    data_cpy = data.copy()
    # ensure pd.Timedelta
    data = data + pd.Timedelta("0h")
    # convert to datetime
    data = datetime.datetime.min + data.dt.to_pytimedelta()
    # convert to time
    data = [d.time() if d is not pd.NaT else None for d in data]
    data = pd.Series(np.array(data), index=data_cpy.index, name=data_cpy.name)
    return data
