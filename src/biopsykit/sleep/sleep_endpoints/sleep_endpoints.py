"""Functions for computing sleep endpoints, i.e., parameters that characterize a recording during a sleep study."""
from numbers import Number
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from biopsykit.utils.datatype_helper import SleepEndpointDataFrame, SleepEndpointDict, _SleepEndpointDataFrame


def compute_sleep_endpoints(
    sleep_wake: pd.DataFrame, bed_interval: Sequence[Union[str, int, np.datetime64]]
) -> SleepEndpointDict:
    """Compute a set of sleep endpoints based on sleep/wake information and time spent in bed.

    This functions computes the following sleep endpoints:

    * ``date``: date of recording if input data is time-aware, ``0`` otherwise. **NOTE**: If the participant went
      to bed between 12 am and 6 am (i.e, the beginning of ``bed_interval`` between 12 am and 6 am) ``date`` will
      be set to the day before (because this night is assumed to "belong" to the day before).
    * ``sleep_onset``: Sleep Onset, i.e., time of falling asleep, in absolute time
    * ``wake_onset``: Wake Onset, i.e., time of awakening, in absolute time
    * ``total_sleep_duration``: Total duration spent sleeping, i.e., the duration between the beginning of the first
      sleep interval and the end of the last sleep interval, in minutes
    * ``net_sleep_duration``: Net duration spent sleeping, in minutes
    * ``bed_interval_start``: Bed Interval Start, i.e., time when participant went to bed, in absolute time
    * ``bed_interval_end``: Bed Interval End, i.e., time when participant left bed, in absolute time
    * ``sleep_efficiency``: Sleep Efficiency, defined as the ratio between net sleep duration and total sleep duration
      in percent
    * ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep
      (difference between *Sleep Onset* and *Bed Interval Start*), in minutes
    * ``getup_latency``: Get Up Latency, i.e., time in bed after awakening until getting up (difference between
      *Bed Interval End* and *Wake Onset*), in minutes
    * ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep
      (after *Sleep Onset* and before *Wake Onset*), in minutes
    * ``sleep_bouts``: List with start and end times of sleep bouts
    * ``wake_bouts``: List with start and end times of wake bouts
    * ``number_wake_bouts``: Total number of wake bouts


    Parameters
    ----------
    sleep_wake : :class:`~pandas.DataFrame`
        dataframe with sleep/wake scoring of night. 0 is expected to indicate *sleep*, 1 to indicate *wake*
    bed_interval : array_like
        beginning and end of bed interval, i.e., the time spent in bed


    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDict`
        dictionary with computed sleep endpoints

    """
    # slice sleep data = data between sleep onset and wake onset during bed interval
    sleep_wake = sleep_wake.loc[bed_interval[0] : bed_interval[1]]
    # total sleep duration = length of sleep data
    total_sleep_duration = len(sleep_wake)

    # net sleep duration in minutes = length of 'sleep' predictions (value 1) in sleep data
    net_sleep_time = sleep_wake[sleep_wake["sleep_wake"].eq(1)]
    net_sleep_duration = len(net_sleep_time)
    if net_sleep_time.empty:
        return {}
    df_sw_sleep = sleep_wake[net_sleep_time.index[0] : net_sleep_time.index[-1]].copy()

    # get percent of total time asleep
    sleep_efficiency = 100.0 * (len(net_sleep_time) / len(sleep_wake))
    # wake after sleep onset = duration of wake during first and last 'sleep' sample
    wake_after_sleep_onset = len(df_sw_sleep) - int(df_sw_sleep.sum()[0])

    df_sw_sleep["block"] = df_sw_sleep["sleep_wake"].diff().ne(0).cumsum()
    df_sw_sleep.reset_index(inplace=True)
    df_sw_sleep.rename(columns={"index": "time"}, inplace=True)
    bouts = df_sw_sleep.groupby(by="block")
    df_start_stop = bouts.first()
    df_start_stop.rename(columns={"time": "start"}, inplace=True)
    df_start_stop["end"] = bouts.last()["time"]

    # add 1 min to end for continuous time coverage
    if df_start_stop["end"].dtype == np.int64:
        df_start_stop["end"] = df_start_stop["end"] + 1
    else:
        df_start_stop["end"] = df_start_stop["end"] + pd.Timedelta("1m")

    sleep_bouts = df_start_stop[df_start_stop["sleep_wake"].eq(1)].drop(columns=["sleep_wake"]).reset_index(drop=True)
    wake_bouts = df_start_stop[df_start_stop["sleep_wake"].ne(1)].drop(columns=["sleep_wake"]).reset_index(drop=True)
    num_wake_bouts = len(wake_bouts)
    sleep_onset = net_sleep_time.index[0]
    wake_onset = net_sleep_time.index[-1]

    # start and end of bed interval
    bed_start = bed_interval[0]
    bed_end = bed_interval[1]

    # sleep onset latency = duration between bed interval start and sleep onset
    sleep_onset_latency = len(sleep_wake[sleep_wake.index[0] : sleep_onset])
    # getup latency = duration between wake onset (last 'sleep' sample) and bed interval end
    getup_latency = len(sleep_wake[wake_onset : sleep_wake.index[-1]])

    if isinstance(bed_start, Number):
        date = 0
    else:
        bed_start = str(bed_start)
        bed_end = str(bed_end)
        sleep_onset = str(sleep_onset)
        wake_onset = str(wake_onset)
        date = pd.to_datetime(bed_start)
        if date.hour < 6:
            date = date - pd.Timedelta("1d")
            date = str(date.normalize())

    dict_result = {
        "date": date,
        "sleep_onset": sleep_onset,
        "wake_onset": wake_onset,
        "total_sleep_duration": total_sleep_duration,
        "net_sleep_duration": net_sleep_duration,
        "bed_interval_start": bed_start,
        "bed_interval_end": bed_end,
        "sleep_efficiency": sleep_efficiency,
        "sleep_onset_latency": sleep_onset_latency,
        "getup_latency": getup_latency,
        "wake_after_sleep_onset": wake_after_sleep_onset,
        "sleep_bouts": sleep_bouts,
        "wake_bouts": wake_bouts,
        "number_wake_bouts": num_wake_bouts,
    }
    return dict_result


def endpoints_as_df(sleep_endpoints: SleepEndpointDict) -> Optional[SleepEndpointDataFrame]:
    """Convert ``SleepEndpointDict`` into ``SleepEndpointDataFrame``.

    Parameters
    ----------
    sleep_endpoints : :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDict`
        dictionary with computed Sleep Endpoints

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDataFrame`
        dataframe with computed Sleep Endpoints or ``None`` if ``sleep_endpoints`` is ``None``

    """
    if sleep_endpoints is None:
        return None

    sleep_endpoints = sleep_endpoints.copy()
    sleep_bouts = sleep_endpoints.pop("sleep_bouts", None).values.tolist()
    wake_bouts = sleep_endpoints.pop("wake_bouts", None).values.tolist()

    sleep_bouts = [tuple(v) for v in sleep_bouts]
    wake_bouts = [tuple(v) for v in wake_bouts]

    index = pd.to_datetime(pd.Index([sleep_endpoints["date"]], name="date"))
    sleep_endpoints.pop("date")

    df = pd.DataFrame(sleep_endpoints, index=index)
    df.fillna(value=np.nan, inplace=True)
    df["sleep_bouts"] = None
    df["wake_bouts"] = None
    df.at[df.index[0], "sleep_bouts"] = sleep_bouts
    df.at[df.index[0], "wake_bouts"] = wake_bouts
    return _SleepEndpointDataFrame(df)
