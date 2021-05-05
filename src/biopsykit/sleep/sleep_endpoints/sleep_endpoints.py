from typing import Dict, Optional, Tuple, Union, Sequence

import pandas as pd
import numpy as np


def compute_sleep_endpoints(
    sleep_wake: pd.DataFrame, bed_interval: Sequence[Union[str, int, np.datetime64], ...]
) -> Dict:
    from numbers import Number

    # cut sleep data = data between sleep onset and wake onset during major rest period
    sleep_wake = sleep_wake.loc[bed_interval[0] : bed_interval[1]]
    # total sleep duration = length of sleep data
    total_sleep_duration = len(sleep_wake)

    # net sleep duration in minutes = length of 'sleep' predictions (value 0) in sleep data
    net_sleep_time = sleep_wake[sleep_wake["sleep_wake"].eq(0)]
    net_sleep_duration = len(net_sleep_time)
    if net_sleep_time.empty:
        return {}
    df_sw_sleep = sleep_wake[net_sleep_time.index[0] : net_sleep_time.index[-1]].copy()

    # get percent of total time asleep
    sleep_efficiency = 100.0 * (len(net_sleep_time) / len(sleep_wake))
    # wake after sleep onset = duration of wake during first and last 'sleep' sample
    wake_after_sleep_onset = int(df_sw_sleep.sum()[0])

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

    sleep_bouts = df_start_stop[df_start_stop["sleep_wake"].eq(0)].drop(columns=["sleep_wake"]).reset_index(drop=True)
    wake_bouts = df_start_stop[df_start_stop["sleep_wake"].ne(0)].drop(columns=["sleep_wake"]).reset_index(drop=True)
    num_wake_bouts = len(wake_bouts)
    sleep_onset = net_sleep_time.index[0]
    wake_onset = net_sleep_time.index[-1]

    # start and end of major rest period
    mrp_start = bed_interval[0]
    mrp_end = bed_interval[1]

    # sleep onset latency = duration between major rest period start and sleep onset
    sleep_onset_latency = len(sleep_wake[sleep_wake.index[0] : sleep_onset])
    # getup latency = duration between wake onset (last 'sleep' sample) and major rest period end
    getup_latency = len(sleep_wake[wake_onset : sleep_wake.index[-1]])

    if isinstance(mrp_start, Number):
        date = 0
    else:
        mrp_start = str(mrp_start)
        mrp_end = str(mrp_end)
        sleep_onset = str(sleep_onset)
        wake_onset = str(wake_onset)
        date = pd.to_datetime(mrp_start)
        if date.hour < 12:
            date = date - pd.Timedelta("1d")
            date = str(date.normalize())

    dict_result = {
        "date": date,
        "sleep_onset": sleep_onset,
        "wake_onset": wake_onset,
        "total_sleep_duration": total_sleep_duration,
        "net_sleep_duration": net_sleep_duration,
        "major_rest_period_start": mrp_start,
        "major_rest_period_end": mrp_end,
        "sleep_efficiency": sleep_efficiency,
        "sleep_onset_latency": sleep_onset_latency,
        "getup_latency": getup_latency,
        "wake_after_sleep_onset": wake_after_sleep_onset,
        "sleep_bouts": sleep_bouts,
        "wake_bouts": wake_bouts,
        "number_wake_bouts": num_wake_bouts,
    }
    return dict_result


def endpoints_as_df(sleep_endpoints: Dict, subject_id: str) -> Optional[pd.DataFrame]:
    if sleep_endpoints is None:
        return None

    sleep_endpoints = sleep_endpoints.copy()
    sleep_bouts = sleep_endpoints.pop("sleep_bouts", None).values.tolist()
    wake_bouts = sleep_endpoints.pop("wake_bouts", None).values.tolist()

    sleep_bouts = [tuple(v) for v in sleep_bouts]
    wake_bouts = [tuple(v) for v in wake_bouts]

    df = pd.DataFrame(sleep_endpoints, index=[subject_id])
    df.index.names = ["subject_id"]
    df.set_index("date", append=True, inplace=True)
    df.fillna(value=np.nan, inplace=True)
    df["sleep_bouts"] = None
    df["wake_bouts"] = None
    df.at[df.index[0], "sleep_bouts"] = sleep_bouts
    df.at[df.index[0], "wake_bouts"] = wake_bouts
    return df
