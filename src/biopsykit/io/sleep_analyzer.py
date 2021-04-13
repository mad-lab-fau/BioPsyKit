"""Module containing different I/O functions to load data recorded by Withings Sleep Analyzer."""
from ast import literal_eval
from pathlib import Path
from typing import Optional, Union, Sequence

import re

import pandas as pd
import pytz
from biopsykit.sleep.utils import split_nights
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t
from biopsykit.utils.time import tz


__all__ = [
    "WITHINGS_RAW_DATA_SOURCES",
    "load_withings_sleep_analyzer_raw_file",
    "load_withings_sleep_analyzer_raw_folder",
    "load_withings_sleep_analyzer_summary",
]

WITHINGS_RAW_DATA_SOURCES = {
    "hr": "heart_rate",
    "respiratory-rate": "respiration_rate",
    "sleep-state": "sleep_state",
    "snoring": "snoring",
}
""" Mapping of data source names to names of the biosignal (and the exported dataframe column)"""


def load_withings_sleep_analyzer_raw_folder(
    folder_path: path_t,
    timezone: Optional[Union[pytz.timezone, str]] = tz,
    split_into_nights: Optional[bool] = True,
) -> Union[pd.DataFrame, Sequence[pd.DataFrame]]:
    """Load folder with raw data from a Withings Sleep Analyzer recording session and convert into time-series data.

    The function will return a list of dataframes (one dataframe per night, if ``split_into_nights`` is ``True``)
    with continuous time-series data (sampling distance: 1min) of all data sources
    (heart rate, respiratory rate, sleep state, snoring) combined. The dataframe columns will be:
        * ``heart_rate``: heart rate in beats-per-minute (bpm)
        * ``respiration_rate``: respiration rate in breaths-per-minute (bpm)
        * ``sleep_state``: current sleep state: 0 = awake, 1 = light sleep, 2 = deep sleep, 3 = rem sleep
        * ``snoring``: flag whether snoring was detected: 0 = no snoring, 100 = snoring

    .. warn::
    If data is not split into single nights (``split_into_nights`` is ``False``)
    data in the dataframe will **not** be resampled.

    Parameters
    ----------
    folder_path: :any:`pathlib.Path` or str
        path to folder with Sleep Analyzer raw data
    timezone : str or pytz.timezone, optional
        timezone of the acquired data, either as string of as pytz object.
        Default: 'Europe/Berlin'
    split_into_nights : bool, optional
        whether to split the dataframe into the different recording nights (and return a list of dataframes) or not.
        Default: ``True``

    See Also
    --------
    load_withings_sleep_analyzer_raw_file
        load a single Sleep Analyzer file with only one data source

    Returns
    -------
    :class:`~pandas.DataFrame` or list of such
        dataframe with Sleep Analyzer date

    """
    # ensure pathlib
    folder_path = Path(folder_path)
    raw_files = list(sorted(folder_path.glob("raw_sleep-monitor_*.csv")))
    data_sources = [re.findall(r"raw_sleep-monitor_(\S*).csv", s.name)[0] for s in raw_files]

    list_data = [
        load_withings_sleep_analyzer_raw_file(file_path, WITHINGS_RAW_DATA_SOURCES[data_source], timezone)
        for file_path, data_source in zip(raw_files, data_sources)
        if data_source in WITHINGS_RAW_DATA_SOURCES
    ]
    data = pd.concat(list_data, axis=1)
    if split_into_nights:
        data = split_nights(data)
        data = [d.resample("1min").interpolate() for d in data]
    return data


def load_withings_sleep_analyzer_raw_file(
    file_path: path_t,
    data_source: str,
    timezone: Optional[Union[pytz.timezone, str]] = tz,
) -> pd.DataFrame:
    """Load single Withings Sleep Analyzer raw data file and convert into time-series data.

    Parameters
    ----------
    file_path : :any:`pathlib.Path` or str
        path to file
    data_source : str
        data source of file specified by ``file_path``
    timezone : str or pytz.timezone, optional
        timezone of the acquired data, either as string of as pytz object.
        Default: 'Europe/Berlin'

    Returns
    -------
    :class:`pandas.DataFrame`
        dataframe with Sleep Analyzer raw data

    Raises
    ------
    ValueError
        if unsupported data source was passed

    """
    if data_source not in WITHINGS_RAW_DATA_SOURCES.values():
        raise ValueError(
            "Unsupported data source {}! Must be one of {}.".format(
                data_source, list(WITHINGS_RAW_DATA_SOURCES.values())
            )
        )

    data = pd.read_csv(file_path)
    # convert string timestamps to datetime
    data["start"] = pd.to_datetime(data["start"])
    # convert strings of arrays to arrays
    data["duration"] = data["duration"].apply(literal_eval)
    data["value"] = data["value"].apply(literal_eval)
    # set index and sort
    data = data.set_index("start").sort_index()
    # rename index
    data.index.name = "time"
    # explode data and apply timestamp explosion to groups
    data_explode = data.apply(pd.Series.explode)
    data_explode = data_explode.groupby("time", group_keys=False).apply(_explode_timestamp)
    # convert it into the right time zone
    data_explode = data_explode.tz_localize("UTC").tz_convert(timezone)
    # rename the value column
    data_explode.columns = [data_source]
    # convert dtypes from object into numerical values
    data_explode = data_explode.astype(int)
    # sort index and drop duplicate index values
    data_explode = data_explode.sort_index()
    data_explode = data_explode[~data_explode.index.duplicated()]
    return data_explode


def load_withings_sleep_analyzer_summary(file_path: path_t) -> pd.DataFrame:
    """Load Sleep Analyzer summary file.

    This function additionally computes several other sleep endpoints from the Sleep Analyzer summary data to be
    comparable with the output with the format of other sleep analysis algorithms.
    All time information are reported in minutes.
    The resulting dataframe has the following columns:
    * ``total_duration``: Total recording time
    * ``total_time_light_sleep``: Total time of light sleep
    * ``total_time_deep_sleep``: Total time of deep sleep
    * ``total_time_rem_sleep``: Total time of REM sleep
    * ``total_time_awake``: Total time of being awake
    * ``num_wake_bouts``: Total number of wake bouts
    * ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep
    * ``getup_onset_latency``: Get Up Latency, i.e., time in bed after awakening until getting up
    * ``sleep_onset``: Sleep Onset, i.e., time of falling asleep, in absolute time
    * ``wake_onset``: Wake Onset, i.e., time of awakening, in absolute time
    * ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep
    * ``total_sleep_duration``: Total sleep duration, i.e., time between Sleep Onset and Wake Onsetvg
    * ``count_snoring_episodes``: Total number of snoring episodes
    * ``total_time_snoring``: Total time of snoring
    * ``heart_rate_avg``: Average heart rate during recording in bpm
    * ``heart_rate_min``: Minimum heart rate during recording in bpm
    * ``heart_rate_max``: Maximum heart rate during recording in bpm

    Parameters
    ----------
    file_path : :any:`pathlib.Path` or str
        path to file

    Returns
    -------
    :class:`pandas.DataFrame`
        dataframe with Sleep Analyzer summary data

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, ".csv")

    data = pd.read_csv(file_path)

    for col in ["von", "bis"]:
        # convert into date time
        data[col] = pd.to_datetime(data[col])

    # total duration in seconds
    data["total_duration"] = [int(td.total_seconds()) for td in data["bis"] - data["von"]]
    data["time"] = data["von"]

    data.rename(
        {
            "leicht (s)": "total_time_light_sleep",
            "tief (s)": "total_time_deep_sleep",
            "rem (s)": "total_time_rem_sleep",
            "wach (s)": "total_time_awake",
            "Aufwachen": "num_wake_bouts",
            "Duration to sleep (s)": "sleep_onset_latency",
            "Duration to wake up (s)": "getup_latency",
            "Snoring episodes": "count_snoring_episodes",
            "Snoring (s)": "total_time_snoring",
            "Average heart rate": "heart_rate_avg",
            "Heart rate (min)": "heart_rate_min",
            "Heart rate (max)": "heart_rate_max",
        },
        axis="columns",
        inplace=True,
    )

    data["sleep_onset"] = data["time"] + pd.to_timedelta(data["sleep_onset_latency"], unit="seconds")
    # Wake after Sleep Onset (WASO): total time awake after sleep onset
    data["wake_after_sleep_onset"] = data["total_time_awake"] - data["sleep_onset_latency"] - data["getup_latency"]
    data["wake_onset"] = data["bis"] - pd.to_timedelta(data["getup_latency"], unit="seconds")
    # compute total sleep duration
    # = total duration - (time to fall asleep + time to get up (= time spent in bed after waking up))
    data["total_sleep_duration"] = data["total_duration"] - data["sleep_onset_latency"] - data["getup_latency"]

    transform_cols = [
        "total_time_light_sleep",
        "total_time_deep_sleep",
        "total_time_rem_sleep",
        "total_time_awake",
        "sleep_onset_latency",
        "getup_latency",
        "total_time_snoring",
        "wake_after_sleep_onset",
        "total_sleep_duration",
    ]
    data[transform_cols] = data[transform_cols].transform(lambda column: (column / 60).astype(int))

    data.drop(columns=["von", "bis"], inplace=True)
    data.set_index("time", inplace=True)

    # reindex column order
    new_cols = list(data.columns)
    sowo = ["sleep_onset", "wake_onset"]
    for d in sowo:
        new_cols.remove(d)
    data = data[sowo + new_cols]
    return data


def _explode_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # sum up the time durations and subtract the first value from it (so that we start from 0)
    # dur_sum then looks like this: [0, 60, 120, 180, ...]
    dur_sum = df["duration"].cumsum() - df["duration"].iloc[0]
    # Add these time durations to the index timestamps.
    # For that, we need to convert the datetime objects from the pandas DatetimeIndex into a float
    # and add the time onto it (we first need to multiply it with 10^9 because the time in the index
    # is stored in nanoseconds)
    index_sum = df.index.values.astype(float) + 1e9 * dur_sum
    # convert the float values back into a DatetimeIndex
    df["time"] = pd.to_datetime(index_sum)
    # set this as index
    df.set_index("time", inplace=True)
    # we don't need the duration column anymore so we can drop it
    df.drop(columns="duration", inplace=True)
    return df
