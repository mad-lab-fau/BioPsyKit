import warnings
from typing import Dict, Union, Optional, Tuple

import pandas as pd
import numpy as np

from biopsykit.sleep.imu.sleep_wake import SleepWake
from biopsykit.sleep.imu.mrp import MajorRestPeriod
from biopsykit.sleep.imu.wear_detection import WearDetection
from biopsykit.signals.imu.activity_counts import ActivityCounts


def predict_pipeline(data: Union[pd.DataFrame, np.array], sampling_rate: int,
                     sleep_wake_scale_factor: Optional[float] = None) -> Dict:
    ac = ActivityCounts(sampling_rate)
    if sleep_wake_scale_factor is None:
        sw = SleepWake('cole_kripke')
    else:
        sw = SleepWake('cole_kripke', scale_factor=sleep_wake_scale_factor)
    wd = WearDetection(sampling_rate=sampling_rate)
    mrp = MajorRestPeriod(sampling_rate=sampling_rate)

    df_wear = wd.predict(data)
    major_wear_block = wd.get_major_wear_block(df_wear)

    # cut data to major wear block
    data = cut_to_wear_block(data, major_wear_block)

    if len(data) == 0:
        return {}

    df_ac = ac.calculate(data)
    df_sw = sw.predict(df_ac)
    df_mrp = mrp.predict(data)
    sleep_endpoints = calculate_endpoints(df_sw, df_mrp)
    if not sleep_endpoints:
        return {}

    major_wear_block = [str(d) for d in major_wear_block]

    dict_result = {
        'wear_detection': df_wear,
        'activity_counts': df_ac,
        'sleep_wake_prediction': df_sw,
        'major_wear_block': major_wear_block,
        'major_rest_period': df_mrp,
        'sleep_endpoints': sleep_endpoints
    }
    return dict_result


def cut_to_wear_block(data: pd.DataFrame, wear_block: Tuple) -> pd.DataFrame:
    if isinstance(data.index, pd.DatetimeIndex):
        return data.loc[wear_block[0]:wear_block[-1]]
    else:
        return data.iloc[wear_block[0]:wear_block[-1]]


def calculate_endpoints(sleep_wake: pd.DataFrame, major_rest_periods: pd.DataFrame) -> Dict:
    from numbers import Number
    # sleep/wake data during major rest period
    sleep_wake = sleep_wake.loc[major_rest_periods['start'][0]:major_rest_periods['end'][0]]

    # total sleep time in minutes (= length of 'sleep' predictions (value 0) in dataframe)
    sleep_time = sleep_wake[sleep_wake['sleep_wake'].eq(0)]
    tst = len(sleep_time)
    if sleep_time.empty:
        return {}
    df_sw_sleep = sleep_wake[sleep_time.index[0]:sleep_time.index[-1]].copy()

    # get percent of total time asleep
    se = 100.0 * (len(sleep_time) / len(sleep_wake))
    # wake after sleep onset = duration of wake during first and last 'sleep' sample
    waso = int(df_sw_sleep.sum()[0])
    # sleep onset latency = duration between beginning of recording and first 'sleep' sample
    sol = len(sleep_wake[sleep_wake.index[0]:sleep_time.index[0]])

    df_sw_sleep["block"] = df_sw_sleep['sleep_wake'].diff().ne(0).cumsum()
    df_sw_sleep.reset_index(inplace=True)
    df_sw_sleep.rename(columns={'index': 'time'}, inplace=True)
    bouts = df_sw_sleep.groupby(by="block")
    df_start_stop = bouts.first()
    df_start_stop.rename(columns={'time': 'start'}, inplace=True)
    df_start_stop['end'] = bouts.last()['time']

    # add 1 min to end for continuous time coverage
    if df_start_stop['end'].dtype == np.int64:
        df_start_stop['end'] = df_start_stop['end'] + 1
    else:
        df_start_stop['end'] = df_start_stop['end'] + pd.Timedelta("1m")

    sleep_bouts = df_start_stop[df_start_stop['sleep_wake'].eq(0)].drop(columns=["sleep_wake"]).reset_index(
        drop=True)
    wake_bouts = df_start_stop[df_start_stop['sleep_wake'].ne(0)].drop(columns=["sleep_wake"]).reset_index(
        drop=True)
    num_wake_bouts = len(wake_bouts)
    sleep_onset = sleep_time.index[0]
    wake_onset = sleep_time.index[-1]
    mrp_start = major_rest_periods['start'][0]
    mrp_end = major_rest_periods['end'][0]

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
        'date': date,
        'sleep_onset': sleep_onset,
        'wake_onset': wake_onset,
        'total_sleep_time': tst,
        'major_rest_period_start': mrp_start,
        'major_rest_period_end': mrp_end,
        'sleep_efficiency': se,
        'sleep_onset_latency': sol,
        'wake_after_sleep_onset': waso,
        'sleep_bouts': sleep_bouts,
        'wake_bouts': wake_bouts,
        'number_wake_bouts': num_wake_bouts
    }
    return dict_result


def endpoints_as_df(sleep_endpoints: Dict, subject_id: str) -> pd.DataFrame:
    sleep_endpoints = sleep_endpoints.copy()
    sleep_bouts = sleep_endpoints.pop('sleep_bouts', None).values.tolist()
    wake_bouts = sleep_endpoints.pop('wake_bouts', None).values.tolist()

    sleep_bouts = [tuple(v) for v in sleep_bouts]
    wake_bouts = [tuple(v) for v in wake_bouts]

    df = pd.DataFrame(sleep_endpoints, index=[subject_id])
    df.index.names = ['subject_id']
    df.set_index('date', append=True, inplace=True)
    df.fillna(value=np.nan, inplace=True)
    df['sleep_bouts'] = None
    df['wake_bouts'] = None
    df.at[df.index[0], 'sleep_bouts'] = sleep_bouts
    df.at[df.index[0], 'wake_bouts'] = wake_bouts
    return df
