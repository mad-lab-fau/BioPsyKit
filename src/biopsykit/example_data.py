"""Module providing functions to load different example data."""
from pathlib import Path
from typing import Sequence, Dict, Optional, Tuple, Union

import pandas as pd

from biopsykit.utils.datatype_helper import (
    is_saliva_mean_se_dataframe,
    SubjectConditionDataFrame,
    SalivaRawDataFrame,
    SalivaMeanSeDataFrame,
    HeartRatePhaseDict,
    SleepEndpointDataFrame,
    _SalivaMeanSeDataFrame,
)
from biopsykit.io import load_subject_condition_list, load_time_log, load_questionnaire_data
from biopsykit.io.carwatch_logs import load_log_one_subject
from biopsykit.io.eeg import load_eeg_raw_muse
from biopsykit.io.ecg import load_hr_phase_dict
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.io.saliva import load_saliva_wide_format
from biopsykit.io.sleep_analyzer import load_withings_sleep_analyzer_raw_folder
from biopsykit.io.sleep_analyzer import load_withings_sleep_analyzer_summary
from biopsykit.utils._types import path_t

_EXAMPLE_DATA_PATH = Path(__file__).parent.parent.parent.joinpath("example_data")

__all__ = [
    "get_file_path",
    "get_condition_list_example",
    "get_saliva_example",
    "get_saliva_mean_se_example",
    "get_mist_hr_example",
    "get_ecg_example",
    "get_ecg_example_02",
    "get_sleep_analyzer_raw_example",
    "get_sleep_analyzer_summary_example",
    "get_sleep_imu_example",
    "get_car_watch_log_data_example",
    "get_time_log_example",
    "get_questionnaire_example",
]


def get_file_path(file_name: path_t) -> Path:
    """Return path to example data file.

    Parameters
    ----------
    file_name : str or :class:`~pathlib.Path`
        file name

    Returns
    -------
    :class:`~pathlib.Path`
        absolute path to file

    """
    file_path = _EXAMPLE_DATA_PATH.joinpath(file_name)
    if file_path.is_file():
        # file exists
        return file_path
    raise ValueError("File {} does not exist!".format(file_name))


def get_condition_list_example() -> SubjectConditionDataFrame:
    """Return example data for subject condition assignment.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame`
        dataframe with example subject condition assignment

    """
    return load_subject_condition_list(
        _EXAMPLE_DATA_PATH.joinpath("condition_list.csv"),
        subject_col="subject",
        condition_col="condition",
    )


def get_saliva_example(sample_times: Optional[Sequence[int]] = None) -> SalivaRawDataFrame:
    """Return saliva example data.

    Parameters
    ----------
    sample_times : list of int, optional
        sample times of saliva samples in minutes

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        dataframe with example raw saliva data

    """
    return load_saliva_wide_format(
        _EXAMPLE_DATA_PATH.joinpath("cortisol_sample.csv"),
        saliva_type="cortisol",
        condition_col="condition",
        sample_times=sample_times,
    )


# def get_saliva_example_stroop(
#     sample_times: Optional[Sequence[int]] = None,
# ) -> pd.DataFrame:
#     return load_saliva_wide_format(
#         _EXAMPLE_DATA_PATH.joinpath("cortisol_sample_stroop.csv"),
#         saliva_type="cortisol",
#         sample_times=sample_times,
#     )


def get_saliva_mean_se_example() -> Dict[str, SalivaMeanSeDataFrame]:
    """Return dictionary with mean and standard error from example data for different saliva types.

    Returns
    -------
    dict
        dictionary with :obj:`~biopsykit.utils.datatype_helper.SalivaMeanSeDataFrame` from different saliva types

    """
    data_dict = pd.read_excel(_EXAMPLE_DATA_PATH.joinpath("saliva_sample_mean_se.xlsx"), sheet_name=None)
    for key in data_dict:
        data_dict[key] = _SalivaMeanSeDataFrame(data_dict[key].set_index(["sample", "time"]))
        is_saliva_mean_se_dataframe(data_dict[key])
    return data_dict


def get_mist_hr_example() -> HeartRatePhaseDict:
    """Return heart rate time-series example data collected during MIST from one subject.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        dictionary with heart rate time-series data from one subject during multiple phases

    """
    return load_hr_phase_dict(_EXAMPLE_DATA_PATH.joinpath("hr_sample_mist.xlsx"))


def get_ecg_example() -> Tuple[pd.DataFrame, float]:
    """Return raw ECG example data from one subject.

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with raw ECG data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_dataset_nilspod(
        file_path=_EXAMPLE_DATA_PATH.joinpath("ecg").joinpath("ecg_sample_Vp01.bin"), datastreams=["ecg"]
    )


def get_ecg_example_02() -> Tuple[pd.DataFrame, float]:
    """Return second raw ECG example data from another subject.

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with raw ECG data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_dataset_nilspod(
        file_path=_EXAMPLE_DATA_PATH.joinpath("ecg").joinpath("ecg_sample_Vp02.bin"), datastreams=["ecg"]
    )


def get_sleep_analyzer_raw_example(
    split_into_nights: Optional[bool] = True,
) -> Union[pd.DataFrame, Sequence[pd.DataFrame]]:
    """Return Withings Sleep Analyzer example raw data.

    Parameters
    ----------
    split_into_nights : bool, optional
        ``True`` to split data into single dataframes, one dataframe per night,
        ``False`` to keep all data in one dataframe. Default: ``True``

    Returns
    -------
    :class:`~pandas.DataFrame` or list
        dataframe with raw sleep analyzer data or a list of such if ``split_into_nights`` is ``True``

    """
    return load_withings_sleep_analyzer_raw_folder(
        _EXAMPLE_DATA_PATH.joinpath("sleep"), split_into_nights=split_into_nights
    )


def get_sleep_analyzer_summary_example() -> SleepEndpointDataFrame:
    """Return Withings Sleep Analyzer example summary data.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDataFrame`
        dataframe with example sleep endpoints computed from Withings Sleep Analyzer Summary data

    """
    return load_withings_sleep_analyzer_summary(_EXAMPLE_DATA_PATH.joinpath("sleep").joinpath("sleep.csv"))


def get_sleep_imu_example() -> Tuple[pd.DataFrame, float]:
    """Return raw IMU example data collected from a wrist-worn IMU sensor during night.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with raw IMU data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_dataset_nilspod(file_path=_EXAMPLE_DATA_PATH.joinpath("sleep_imu").joinpath("sleep_imu_sample_01.bin"))


def get_eeg_example() -> Tuple[pd.DataFrame, float]:
    """Return raw EEG example data collected from a Muse EEG headband.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with raw EEG data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_eeg_raw_muse(_EXAMPLE_DATA_PATH.joinpath("eeg_muse_example.csv"))


def get_car_watch_log_data_example() -> pd.DataFrame:
    """Return *CARWatch App* example log data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with example log data from the *CARWatch* app

    """
    return load_log_one_subject(_EXAMPLE_DATA_PATH.joinpath("log_data").joinpath("AB12C"))


def get_time_log_example() -> pd.DataFrame:
    """Return time log example data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with example time log information. The time log match the data from the two ECG data example
        functions :func:`~biopsykit.example_data.get_ecg_example` and :func:`~biopsykit.example_data.get_ecg_example_02`

    """
    return load_time_log(_EXAMPLE_DATA_PATH.joinpath("ecg_time_log.xlsx"))


def get_questionnaire_example() -> pd.DataFrame:
    """Return questionnaire example data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with questionnaire data

    """
    return load_questionnaire_data(_EXAMPLE_DATA_PATH.joinpath("questionnaire_sample.csv"), index_col=["subject"])
