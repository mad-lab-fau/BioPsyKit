from typing import Dict, Union, Optional

import pandas as pd
import numpy as np

import biopsykit.signals as signals
import biopsykit.sleep as sleep


def predict_pipeline(data: Union[pd.DataFrame, np.array], sampling_rate: int,
                     sleep_wake_scale_factor: Optional[float] = None) -> Dict:
    ac = signals.imu.activity_counts.ActivityCounts(sampling_rate)
    if sleep_wake_scale_factor is None:
        sw = sleep.imu.sleep_wake.SleepWake('cole_kripke')
    else:
        sw = sleep.imu.sleep_wake.SleepWake('cole_kripke', scale_factor=sleep_wake_scale_factor)
    wd = sleep.imu.wear_detection.WearDetection(sampling_rate=sampling_rate)
    mrp = sleep.imu.mrp.MajorRestPeriod(sampling_rate=sampling_rate)

    df_wear = wd.predict(data)
    df_ac = ac.calculate(data)
    df_sw = sw.predict(df_ac)
    df_mrp = mrp.predict(data)
    sleep_endpoints = calculate_endpoints(df_sw, df_mrp)

    dict_result = {
        'wear_detection': df_wear,
        'activity_counts': df_ac,
        'sleep_wake_prediction': df_sw,
        'major_rest_period': df_mrp,
        'sleep_endpoints': sleep_endpoints
    }
    return dict_result


def calculate_endpoints(sleep_wake: pd.DataFrame, major_rest_periods: pd.DataFrame) -> Dict:
    # sleep/wake data during major rest period
    sleep_wake = sleep_wake[major_rest_periods['start'][0]:major_rest_periods['end'][0]]

    # total sleep time in minutes (= length of 'sleep' predictions (value 0) in dataframe)
    sleep_time = sleep_wake[sleep_wake['sleep_wake'].eq(0)]
    tst = len(sleep_time)
    df_sw_sleep = sleep_wake[sleep_time.index[0]:sleep_time.index[-1]].copy()

    # get percent of total time asleep
    se = 100.0 * (len(sleep_time) / len(sleep_wake))
    # wake after sleep onset = duration of wake during first and last 'sleep' sample
    waso = int(df_sw_sleep.sum()[0])
    # sleep onset latency = duration between beginning of recording and first 'sleep' sample
    sol = len(sleep_wake[sleep_wake.index[0]:sleep_time.index[0]])

    df_sw_sleep["block"] = df_sw_sleep['sleep_wake'].diff().ne(0).cumsum()
    df_sw_sleep.reset_index(inplace=True)
    bouts = df_sw_sleep.groupby(by="block")
    df_start_stop = bouts.first()
    df_start_stop.rename(columns={'time': 'start'}, inplace=True)
    df_start_stop['end'] = bouts.last()['time']
    # add 1 min to end for continuous time coverage
    df_start_stop['end'] = df_start_stop['end'] + pd.Timedelta("1m")

    sleep_bouts = df_start_stop[df_start_stop['sleep_wake'].eq(0)].drop(columns=["sleep_wake"]).reset_index(
        drop=True)
    wake_bouts = df_start_stop[df_start_stop['sleep_wake'].ne(0)].drop(columns=["sleep_wake"]).reset_index(
        drop=True)
    num_wake_bouts = len(wake_bouts)
    sleep_onset = sleep_time.index[0]
    wake_onset = sleep_time.index[-1]

    dict_result = {
        'sleep_onset': sleep_onset,
        'wake_onset': wake_onset,
        'total_sleep_time': tst,
        'major_rest_period_start': major_rest_periods['start'][0],
        'major_rest_period_end': major_rest_periods['end'][0],
        'sleep_efficiency': se,
        'sleep_onset_latency': sol,
        'wake_after_sleep_onset': waso,
        'sleep_bouts': sleep_bouts,
        'wake_bouts': wake_bouts,
        'number_wake_bouts': num_wake_bouts
    }
    return dict_result
