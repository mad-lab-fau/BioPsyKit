"""Module containing different I/O functions to load data recorded by Withings Sleep Analyzer."""
from ast import literal_eval
import datetime
from pathlib import Path
from typing import Optional, Union, Sequence

import re

import pandas as pd

from biopsykit.sleep.utils import split_nights
from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_has_columns, _assert_is_dir
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import is_sleep_endpoint_dataframe, SleepEndpointDataFrame
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
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
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

    The files are all expected to have the following name pattern: ``raw-sleep-monitor_<datasource>.csv``.

    .. warning::
        If data is not split into single nights (``split_into_nights`` is ``False``),
        data in the dataframe will **not** be resampled.

    Parameters
    ----------
    folder_path: :class:`~pathlib.Path` or str
        path to folder with Sleep Analyzer raw data
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of the acquired data, either as string of as tzinfo object.
        Default: 'Europe/Berlin'
    split_into_nights : bool, optional
        whether to split the dataframe into the different recording nights (and return a list of dataframes) or not.
        Default: ``True``

    Returns
    -------
    :class:`~pandas.DataFrame` or list of such
        dataframe (or list of dataframes, if ``split_into_nights`` is ``True``) with Sleep Analyzer data

    Raises
    ------
    ValueError
        if ``folder_path`` is not a directory
        if no Sleep Analyzer Raw files are in directory specified by ``folder_path``


    See Also
    --------
    load_withings_sleep_analyzer_raw_file
        load a single Sleep Analyzer file with only one data source

    """
    # ensure pathlib
    folder_path = Path(folder_path)

    _assert_is_dir(folder_path)

    raw_files = list(sorted(folder_path.glob("raw_sleep-monitor_*.csv")))
    if len(raw_files) == 0:
        raise ValueError("No sleep analyzer raw files found in {}!".format(folder_path))
    data_sources = [re.findall(r"raw_sleep-monitor_(\S*).csv", s.name)[0] for s in raw_files]

    list_data = [
        load_withings_sleep_analyzer_raw_file(
            file_path,
            data_source=WITHINGS_RAW_DATA_SOURCES[data_source],
            timezone=timezone,
            split_into_nights=split_into_nights,
        )
        for file_path, data_source in zip(raw_files, data_sources)
        if data_source in WITHINGS_RAW_DATA_SOURCES
    ]
    if split_into_nights:
        # "transpose" nested list.
        # before: outer lists = data source, inner lists = nights.
        # after: outer lists = nights, inner lists = data source
        list_data = list(map(list, zip(*list_data)))
        data = [pd.concat(list_nights, axis=1) for list_nights in list_data]
    else:
        data = pd.concat(list_data, axis=1)
    return data


def load_withings_sleep_analyzer_raw_file(
    file_path: path_t,
    data_source: str,
    timezone: Optional[Union[datetime.tzinfo, str]] = None,
    split_into_nights: Optional[bool] = True,
) -> Union[pd.DataFrame, Sequence[pd.DataFrame]]:
    """Load single Withings Sleep Analyzer raw data file and convert into time-series data.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    data_source : str
        data source of file specified by ``file_path``
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of recorded data, either as string or as tzinfo object.
        Default: 'Europe/Berlin'
    split_into_nights : bool, optional
        whether to split the dataframe into the different recording nights (and return a list of dataframes) or not.
        Default: ``True``

    Returns
    -------
    :class:`~pandas.DataFrame` or list of such
        dataframe (or list of dataframes, if ``split_into_nights`` is ``True``) with Sleep Analyzer data

    Raises
    ------
    ValueError
        if unsupported data source was passed
    `~biopsykit.utils.exceptions.FileExtensionError`
        if ``file_path`` is not a csv file
    `~biopsykit.utils.exceptions.ValidationError`
        if file does not have the required columns ``start``, ``duration``, ``value``

    """
    if data_source not in WITHINGS_RAW_DATA_SOURCES.values():
        raise ValueError(
            "Unsupported data source {}! Must be one of {}.".format(
                data_source, list(WITHINGS_RAW_DATA_SOURCES.values())
            )
        )

    file_path = Path(file_path)
    _assert_file_extension(file_path, ".csv")

    data = pd.read_csv(file_path)

    _assert_has_columns(data, [["start", "duration", "value"]])

    if timezone is None:
        timezone = tz

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

    if split_into_nights:
        data_explode = split_nights(data_explode)
        data_explode = [d.resample("1min").interpolate() for d in data_explode]
    return data_explode


def load_withings_sleep_analyzer_summary(file_path: path_t, timezone: Optional[str] = None) -> SleepEndpointDataFrame:
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
        * ``total_sleep_duration``: Total sleep duration, i.e., time between Sleep Onset and Wake Onset
        * ``number_wake_bouts``: Total number of wake bouts
        * ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep
        * ``getup_onset_latency``: Get Up Latency, i.e., time in bed after awakening until getting up
        * ``sleep_onset``: Sleep Onset, i.e., time of falling asleep, in absolute time
        * ``wake_onset``: Wake Onset, i.e., time of awakening, in absolute time
        * ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep
        * ``count_snoring_episodes``: Total number of snoring episodes
        * ``total_time_snoring``: Total time of snoring
        * ``heart_rate_avg``: Average heart rate during recording in bpm
        * ``heart_rate_min``: Minimum heart rate during recording in bpm
        * ``heart_rate_max``: Maximum heart rate during recording in bpm


    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of recorded data, either as string or as tzinfo object.
        Default: 'Europe/Berlin'


    Returns
    -------
    :obj:`~biopsykit.datatype_helper.SleepEndpointDataFrame`
        dataframe with Sleep Analyzer summary data, i.e., sleep endpoints

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, ".csv")

    data = pd.read_csv(file_path)

    _assert_has_columns(data, [["von", "bis"]])

    if timezone is None:
        timezone = tz

    for col in ["von", "bis"]:
        # convert into date time
        data[col] = pd.to_datetime(data[col]).dt.tz_convert(timezone)

    # total duration in seconds
    data["total_duration"] = [int(td.total_seconds()) for td in data["bis"] - data["von"]]
    data["date"] = data["von"]
    data["date"] = data["date"].apply(
        lambda date: ((date - pd.Timedelta("1d")) if date.hour < 12 else date).normalize()
    )

    data.rename(
        {
            "von": "recording_start",
            "bis": "recording_end",
            "leicht (s)": "total_time_light_sleep",
            "tief (s)": "total_time_deep_sleep",
            "rem (s)": "total_time_rem_sleep",
            "wach (s)": "total_time_awake",
            "Aufwachen": "number_wake_bouts",
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

    data["sleep_onset"] = data["recording_start"] + pd.to_timedelta(data["sleep_onset_latency"], unit="seconds")
    # Wake after Sleep Onset (WASO): total time awake after sleep onset
    data["wake_after_sleep_onset"] = data["total_time_awake"] - data["sleep_onset_latency"] - data["getup_latency"]
    data["wake_onset"] = data["recording_end"] - pd.to_timedelta(data["getup_latency"], unit="seconds")
    # compute total sleep duration
    # = total duration - (time to fall asleep + time to get up (= time spent in bed after waking up))
    data["total_sleep_duration"] = data["total_duration"] - data["sleep_onset_latency"] - data["getup_latency"]

    # compute net sleep duration (time spent actually sleeping) = total sleep duration - wake after sleep onset
    data["net_sleep_duration"] = data["total_sleep_duration"] - data["wake_after_sleep_onset"]

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
        "net_sleep_duration",
    ]
    data[transform_cols] = data[transform_cols].transform(lambda column: (column / 60).astype(int))

    data = data.set_index("date")

    # reindex column order
    new_cols = list(data.columns)
    sowo = ["sleep_onset", "wake_onset"]
    for d in sowo:
        new_cols.remove(d)
    data = data[sowo + new_cols]

    # assert output is in the correct format
    is_sleep_endpoint_dataframe(data)

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
