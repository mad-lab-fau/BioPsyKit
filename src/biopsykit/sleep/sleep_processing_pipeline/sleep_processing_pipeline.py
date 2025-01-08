"""Functions to process sleep data from raw IMU data or Actigraph data."""
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np

from biopsykit.signals.imu import convert_acc_data_to_g
from biopsykit.signals.imu.activity_counts import ActivityCounts
from biopsykit.signals.imu.rest_periods import RestPeriods
from biopsykit.signals.imu.wear_detection import WearDetection
from biopsykit.sleep.sleep_endpoints import compute_sleep_endpoints
from biopsykit.sleep.sleep_wake_detection.sleep_wake_detection import SleepWakeDetection
from biopsykit.utils._types_internal import arr_t
from biopsykit.utils.array_handling import accumulate_array


def predict_pipeline_acceleration(
    data: arr_t, sampling_rate: float, convert_to_g: Optional[bool] = True, **kwargs
) -> dict[str, Any]:
    """Apply sleep processing pipeline on raw acceleration data.

    This function processes raw acceleration data collected during sleep. The pipeline consists of the following steps:

    * *Activity Count Conversion*: Convert (3-axis) raw acceleration data into activity counts. Most sleep/wake
      detection algorithms use activity counts (as typically provided by Actigraph devices) as input data.
    * *Wear Detection*: Detect wear and non-wear periods. Cut data to the longest continuous wear block.
    * *Rest Periods*: Detect rest periods, i.e., periods with large physical inactivity. The longest continuous
      rest period (*Major Rest Period*) is used to determine the *Bed Interval*, i.e., the period spent in bed.
    * *Sleep/Wake Detection*: Apply sleep/wake detection algorithm to classify phases of sleep and wake.
    * *Sleep Endpoint Computation*: Compute Sleep Endpoints from sleep/wake detection results and bed interval.

    Parameters
    ----------
    data : array_like with shape (n,3)
        input data. Must be a 3-d acceleration signal
    sampling_rate : float
        sampling rate of recorded data in Hz
    convert_to_g : bool, optional
        ``True`` if input data is provided in :math:`m/s^2` and should be converted in :math:`g`, ``False`` if input
        data is already in :math:`g` and does not need to be converted.
        Default: ``True``
    **kwargs :
        additional parameters to configure sleep/wake detection. The possible parameters depend on the selected
        sleep/wake detection algorithm and are passed to
        :class:`~biopsykit.sleep.sleep_wake_detection.SleepWakeDetection`. Examples are:

        * *algorithm_type*: name of sleep/wake detection algorithm to internally use for sleep/wake detection.
          Default: "Cole/Kripke"
        * *epoch_length*: epoch length in seconds. Default: 60


    Returns
    -------
    dict
        dictionary with Sleep Processing Pipeline results.

    """
    # TODO: add entries of result dictionary to docstring and add possibility to specify sleep/wake prediction algorithm
    kwargs.setdefault("algorithm_type", "cole_kripke")
    kwargs.setdefault("epoch_length", 60)
    ac = ActivityCounts(sampling_rate)
    wd = WearDetection(sampling_rate=sampling_rate)
    rp = RestPeriods(sampling_rate=sampling_rate)
    sw = SleepWakeDetection(**kwargs)

    if convert_to_g:
        data = convert_acc_data_to_g(data, inplace=False)

    df_wear = wd.predict(data)
    major_wear_block = wd.get_major_wear_block(df_wear)

    # cut data to major wear block
    data = wd.cut_to_wear_block(data, major_wear_block)

    if len(data) == 0:
        return {}

    df_ac = ac.calculate(data)
    df_ac = accumulate_array(df_ac, 1, 1 / kwargs.get("epoch_length"))

    df_sw = sw.predict(df_ac)
    df_rp = rp.predict(data)
    bed_interval = [df_rp["start"][0], df_rp["end"][0]]
    print(df_sw.value_counts())
    sleep_endpoints = compute_sleep_endpoints(df_sw, bed_interval)
    if not sleep_endpoints:
        return {}

    major_wear_block = [str(d) for d in major_wear_block]

    dict_result = {
        "wear_detection": df_wear,
        "activity_counts": df_ac,
        "sleep_wake_prediction": df_sw,
        "major_wear_block": major_wear_block,
        "rest_periods": df_rp,
        "bed_interval": bed_interval,
        "sleep_endpoints": sleep_endpoints,
    }
    return dict_result


def predict_pipeline_actigraph(
    data: arr_t, algorithm_type: str, bed_interval: Sequence[Union[str, int, np.datetime64]], **kwargs
) -> dict[str, Any]:
    """Apply sleep processing pipeline on actigraph data.

    This function processes actigraph data collected during sleep and performs sleep/wake detection.

    Parameters
    ----------
    data : array_like with shape (n,3)
        input data. Must be a 3-d acceleration signal
    algorithm_type : str
        name of sleep/wake detection algorithm to internally use for sleep/wake detection
    bed_interval : array_like
        beginning and end of bed interval, i.e., the time spent in bed

    **kwargs :
        additional parameters to configure sleep/wake detection. The possible parameters depend on the selected
        sleep/wake detection algorithm and are passed to
        :class:`~biopsykit.sleep.sleep_wake_detection.SleepWakeDetection`.


    Returns
    -------
    dict
        dictionary with Sleep Processing Pipeline results.

    """
    # TODO: add entries of result dictionary to docstring and add possibility to specify sleep/wake prediction algorithm
    sw = SleepWakeDetection(algorithm_type=algorithm_type, **kwargs)
    df_sw = sw.predict(data[["activity"]])

    sleep_endpoints = compute_sleep_endpoints(df_sw, bed_interval)

    dict_result = {"sleep_wake_prediction": df_sw, "sleep_endpoints": sleep_endpoints}
    return dict_result
