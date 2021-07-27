"""Module containing helper functions to handle time data."""

from typing import Optional, Union

import datetime
import pytz

import pandas as pd
import numpy as np
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype

tz = pytz.timezone("Europe/Berlin")
utc = pytz.timezone("UTC")


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


def time_to_datetime(data: pd.Series) -> pd.Series:
    """Convert time information in a series into ``datetime.datetime`` data.

    Parameters
    ----------
    data : :class:`~pandas.Series`
        series with time information


    Returns
    -------
    :class:`~pandas.Series`
        series with data converted into :class:`datetime.datetime`

    """
    col_data = pd.to_datetime(data.astype(str))
    return col_data - col_data.dt.normalize()


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
