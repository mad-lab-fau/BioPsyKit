from ast import literal_eval
from pathlib import Path
from typing import Optional, Union

import pytz
import pandas as pd
import numpy as np

from biopsykit.utils import path_t, tz

RAW_DATA_SOURCES = {
    'hr': 'heart_rate',
    'respiratory-rate': 'respiration_rate',
    'sleep-state': 'sleep_state',
    'snoring': 'snoring'
}
""" suffix in filename : name of biosignal (and name of dataframe column) """


def load_withings_sleep_analyzer_raw_folder(folder_path: path_t,
                                            timezone: Optional[Union[pytz.timezone, str]] = tz) -> pd.DataFrame:
    import re
    # ensure pathlib
    folder_path = Path(folder_path)
    raw_files = list(sorted(folder_path.glob("raw_sleep-monitor_*.csv")))
    data_sources = [re.findall("raw_sleep-monitor_(\S*).csv", s.name)[0] for s in raw_files]

    list_data = [load_withings_sleep_analyzer_raw_file(file_path, RAW_DATA_SOURCES[data_source], timezone)
                 for file_path, data_source in zip(raw_files, data_sources)]
    return pd.concat(list_data, axis=1)


def load_withings_sleep_analyzer_raw_file(file_path: path_t, data_source: str,
                                          timezone: Optional[Union[pytz.timezone, str]] = tz) -> pd.DataFrame:
    if data_source not in RAW_DATA_SOURCES.values():
        raise ValueError(
            "Unsupported data source {}! Must be one of {}.".format(data_source, list(RAW_DATA_SOURCES.values())))

    data = pd.read_csv(file_path)
    # convert string timestamps to datetime
    data['start'] = pd.to_datetime(data['start'])
    # convert strings of arrays to arrays
    data['duration'] = data['duration'].apply(literal_eval)
    data['value'] = data['value'].apply(literal_eval)
    # set index and sort
    data = data.set_index('start').sort_index()
    # rename index
    data.index.name = 'time'
    # explode data and apply timestamp explosion to groups
    data_explode = data.apply(pd.Series.explode)
    data_explode = data_explode.groupby('time', group_keys=False).apply(_explode_timestamp)
    # convert it into the right time zone
    data_explode = data_explode.tz_localize('UTC').tz_convert(timezone)
    # rename the value column
    data_explode.columns = [data_source]
    return data_explode


def _explode_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # sum up the time durations and subtract the first value from it (so that we start from 0)
    # dur_sum then looks like this: [0, 60, 120, 180, ...]
    dur_sum = df['duration'].cumsum() - df['duration'].iloc[0]
    # Add these time durations to the index timestamps.
    # For that, we need to convert the datetime objects from the pandas DatetimeIndex into a float
    # and add the time onto it (we first need to multiply it with 10^9 because the time in the index
    # is stored in nanoseconds)
    index_sum = df.index.values.astype(float) + 1e9 * dur_sum
    # convert the float values back into a DatetimeIndex
    df['time'] = pd.to_datetime(index_sum)
    # set this as index
    df.set_index('time', inplace=True)
    # we don't need the duration column anymore so we can drop it
    df.drop(columns='duration', inplace=True)
    return df
