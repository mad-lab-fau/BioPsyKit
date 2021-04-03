import json
from datetime import datetime
from typing import Union, Optional, Sequence, Dict

import pandas as pd
import numpy as np

from biopsykit.utils.time import tz


def get_filtered_logs(log_data: "LogData") -> pd.DataFrame:
    return get_logs_for_action(
        log_data,
        log_action=log_data.selected_action,
        selected_day=log_data.selected_day,
    )


def get_logs_for_date(
    log_data: Union["LogData", pd.DataFrame], date: Union[str, datetime.date]
) -> pd.DataFrame:
    if isinstance(log_data, pd.DataFrame):
        df = log_data
    else:
        df = log_data.df

    date = pd.Timestamp(date).tz_localize(tz)

    if date is pd.NaT:
        return df

    return df.loc[df.index.normalize() == date]


def split_nights(
    log_data: Union["LogData", pd.DataFrame], diff_hours: Optional[int] = 12
) -> Sequence[pd.DataFrame]:
    if isinstance(log_data, pd.DataFrame):
        df = log_data
    else:
        df = log_data.df

    idx_split = np.where(
        np.diff(df.index, prepend=df.index[0]) > pd.Timedelta(diff_hours, "hours")
    )[0]
    list_nights = np.split(df, idx_split)
    return list_nights


def get_logs_for_action(
    log_data: Union["LogData", pd.DataFrame],
    log_action: str,
    selected_day: Optional[datetime] = None,
    rows: Optional[Union[str, int, Sequence[int]]] = None,
) -> Union[pd.DataFrame, pd.Series]:
    from biopsykit.carwatch_logs.log_data import LogData

    if isinstance(log_data, pd.DataFrame):
        df = log_data
    else:
        df = log_data.df

    if selected_day is not None:
        df = get_logs_for_date(df, date=selected_day)

    if log_action is None:
        return df
    elif log_action not in LogData.log_actions:
        return pd.DataFrame()

    if rows:
        actions = df[df["action"] == log_action].iloc[rows, :]
    else:
        actions = df[df["action"] == log_action]
    return actions


def get_extras_for_log(
    log_data: Union["LogData", pd.DataFrame], log_action: str
) -> Dict[str, str]:
    row = get_logs_for_action(log_data, log_action, rows=0)
    if row.empty:
        # warnings.warn("Log file has no action {}!".format(log_action))
        return {}

    return json.loads(row["extras"].iloc[0])
