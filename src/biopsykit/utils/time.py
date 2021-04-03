from typing import Optional, Union

import pytz

import pandas as pd
import numpy as np

tz = pytz.timezone("Europe/Berlin")
utc = pytz.timezone("UTC")


def check_tz_aware(data: pd.DataFrame) -> bool:
    return isinstance(data.index, pd.DatetimeIndex) and (data.index.tzinfo is not None)


def get_time_from_date(
    data: pd.Series,
    is_utc: Optional[bool] = False,
    tz_convert: Optional[bool] = False,
    timezone: Optional[Union[str]] = tz,
) -> pd.Series:
    if tz_convert:
        data = pd.to_datetime(data, utc=is_utc).dt.tz_convert(timezone)
    else:
        data = pd.to_datetime(data, utc=is_utc).dt.tz_localize(timezone)
    data = data - data.dt.normalize()

    return data


def time_to_datetime(data: pd.Series) -> pd.Series:
    col_data = pd.to_datetime(data.astype(str))
    return col_data - col_data.dt.normalize()


def timedelta_to_time(data: pd.Series) -> pd.Series:
    import datetime

    data_cpy = data.copy()
    # ensure pd.Timedelta
    data = data + pd.Timedelta("0h")
    # convert to datetime
    data = datetime.datetime.min + data.dt.to_pytimedelta()
    # convert to time
    data = [d.time() if d is not pd.NaT else None for d in data]
    data = pd.Series(np.array(data), index=data_cpy.index, name=data_cpy.name)
    return data
