import datetime
from pathlib import Path
from typing import Optional, Union, Sequence, Dict, List

import pandas as pd
import pytz

from biopsykit.utils import path_t, tz


def load_time_log(file_path: path_t, index_cols: Optional[Union[str, Sequence[str]]] = None,
                  phase_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Loads time log file.

    Parameters
    ----------
    file_path : str or path
        path to time log file, either Excel or csv file
    index_cols : str list, optional
        column name (or list of column names) that should be used for dataframe index or ``None`` for no index.
        Default: ``None``
    phase_cols : list, optional
        list of column names that contain time log information or ``None`` to use all columns. Default: ``None``

    Returns
    -------
    pd.DataFrame
        pandas dataframe with time log information

    """
    # ensure pathlib
    file_path = Path(file_path)
    if file_path.suffix in ['.xls', '.xlsx']:
        df_time_log = pd.read_excel(file_path)
    elif file_path.suffix in ['.csv']:
        df_time_log = pd.read_csv(file_path)
    else:
        raise ValueError("Unrecognized file format {}!".format(file_path.suffix))

    if isinstance(index_cols, str):
        index_cols = [index_cols]

    if index_cols:
        df_time_log.set_index(index_cols, inplace=True)
    if phase_cols:
        df_time_log = df_time_log.loc[:, phase_cols]
    return df_time_log


def convert_time_log_datetime(time_log: pd.DataFrame, dataset: Optional['Dataset'] = None,
                              date: Optional[Union[str, 'datetime']] = None,
                              timezone: Optional[str] = "Europe/Berlin") -> pd.DataFrame:
    if dataset is None and date is None:
        raise ValueError("Either `dataset` or `date` must be supplied as argument!")

    if dataset is not None:
        date = dataset.info.utc_datetime_start.date()
    if isinstance(date, str):
        # ensure datetime
        date = datetime.datetime(date)
    time_log = time_log.applymap(lambda x: pytz.timezone(timezone).localize(datetime.datetime.combine(date, x)))
    return time_log
