from typing import Dict, Union, Optional

import numpy as np
import pandas as pd

from biopsykit.signals.imu import convert_acc_data_to_g

from biopsykit.signals.imu.major_rest_periods import MajorRestPeriods
from biopsykit.signals.imu.wear_detection import WearDetection
from biopsykit.signals.imu.activity_counts import ActivityCounts
from biopsykit.sleep.sleep_wake_detection.sleep_wake_detection import SleepWakeDetection

from biopsykit.sleep.sleep_endpoints import compute_sleep_endpoints


def predict_pipeline_accel(
    data: Union[pd.DataFrame, np.array], sampling_rate: int, convert_to_g: Optional[bool] = True, **kwargs
) -> Dict:
    ac = ActivityCounts(sampling_rate)
    wd = WearDetection(sampling_rate=sampling_rate)
    mrp = MajorRestPeriods(sampling_rate=sampling_rate)
    sw = SleepWakeDetection("cole_kripke", **kwargs)

    if convert_to_g:
        data = convert_acc_data_to_g(data, inplace=False)

    df_wear = wd.predict(data)
    major_wear_block = wd.get_major_wear_block(df_wear)

    # cut data to major wear block
    data = wd.cut_to_wear_block(data, major_wear_block)

    if len(data) == 0:
        return {}

    df_ac = ac.calculate(data)
    df_sw = sw.predict(df_ac)
    df_mrp = mrp.predict(data)
    bed_interval = [df_mrp["start"][0], df_mrp["end"][0]]
    sleep_endpoints = compute_sleep_endpoints(df_sw, bed_interval)
    if not sleep_endpoints:
        return {}

    major_wear_block = [str(d) for d in major_wear_block]

    dict_result = {
        "wear_detection": df_wear,
        "activity_counts": df_ac,
        "sleep_wake_prediction": df_sw,
        "major_wear_block": major_wear_block,
        "major_rest_period": df_mrp,
        "sleep_endpoints": sleep_endpoints,
    }
    return dict_result


def predict_pipeline_actigraph(data: Union[pd.DataFrame, np.array], algorithm: str, bed_interval_index, **kwargs) -> Dict:



    sw = SleepWakeDetection(algorithm_type=algorithm, **kwargs)
    df_sw = sw.predict(data['activity'])

    df_sw = pd.DataFrame([df_sw]).transpose().rename(columns={0: "sleep_wake"})

    sleep_endpoints = compute_sleep_endpoints(df_sw, bed_interval_index)

    dict_result = {
        "sleep_wake_prediction": df_sw,
        "sleep_endpoints": sleep_endpoints
    }


    return dict_result

