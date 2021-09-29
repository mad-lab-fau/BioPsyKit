"""Module providing functions to load different example data.

The data is either taken from the local file system in case biopsykit was installed manually or the example data is
downloaded into the local user folder.
"""
from typing import Sequence, Dict, Optional, Tuple, Union
from urllib.request import urlretrieve
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd

from biopsykit.utils.datatype_helper import (
    is_saliva_mean_se_dataframe,
    SubjectConditionDataFrame,
    SalivaRawDataFrame,
    SalivaMeanSeDataFrame,
    HeartRatePhaseDict,
    HeartRateSubjectDataDict,
    SleepEndpointDataFrame,
    _SalivaMeanSeDataFrame,
)
from biopsykit.utils.file_handling import mkdirs
from biopsykit.io import load_subject_condition_list, load_time_log, load_questionnaire_data
from biopsykit.io.carwatch_logs import load_log_one_subject
from biopsykit.io.eeg import load_eeg_raw_muse
from biopsykit.io.ecg import load_hr_phase_dict
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.io.saliva import load_saliva_wide_format
from biopsykit.io.sleep_analyzer import load_withings_sleep_analyzer_raw_folder
from biopsykit.io.sleep_analyzer import load_withings_sleep_analyzer_summary
from biopsykit.utils._types import path_t

_EXAMPLE_DATA_PATH_LOCAL = Path(__file__).parent.parent.parent.joinpath("example_data")
_EXAMPLE_DATA_PATH_HOME = Path.home().joinpath(".biopsykit_data")
_REMOTE_DATA_PATH = "https://raw.githubusercontent.com/mad-lab-fau/BioPsyKit/main/example_data/"

__all__ = [
    "get_file_path",
    "get_condition_list_example",
    "get_saliva_example",
    "get_saliva_mean_se_example",
    "get_mist_hr_example",
    "get_hr_subject_data_dict_example",
    "get_ecg_example",
    "get_ecg_example_02",
    "get_sleep_analyzer_raw_example",
    "get_sleep_analyzer_summary_example",
    "get_sleep_imu_example",
    "get_car_watch_log_data_example",
    "get_time_log_example",
    "get_questionnaire_example",
]


def _is_installed_manually() -> bool:
    """Check whether biopsykit was installed manually and example data exists in the local path.

    Returns
    -------
    bool
        ``True`` if biopsykit was installed manually, ``False`` otherwise

    """
    return (_EXAMPLE_DATA_PATH_LOCAL / "__init__.py").is_file()


def _get_data(file_name: str) -> path_t:
    if _is_installed_manually():
        return _EXAMPLE_DATA_PATH_LOCAL.joinpath(file_name)
    path = _EXAMPLE_DATA_PATH_HOME.joinpath(file_name)
    if path.exists():
        return path
    mkdirs(path.parent)
    return _fetch_from_remote(file_name, path)


def _fetch_from_remote(file_name: str, file_path: path_t) -> path_t:
    """Download remote dataset (helper function).

    Parameters
    ----------
    file_name : str
        file name
    file_path : str
        path to file

    Returns
    -------
    :class:`~pathlib.Path`
        path to downloaded file

    """
    url = _REMOTE_DATA_PATH + file_name
    print(f"Downloading file {file_name} from remote URL: {url}.")
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=file_name) as t:
        urlretrieve(url, filename=file_path, reporthook=_tqdm_hook(t))
    return file_path


def _tqdm_hook(t):
    """Wrap tqdm instance."""
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def get_file_path(file_name: path_t) -> Optional[Path]:
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
    file_path = _EXAMPLE_DATA_PATH_LOCAL.joinpath(file_name)
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
        _EXAMPLE_DATA_PATH_LOCAL.joinpath("condition_list.csv"),
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
        _get_data("cortisol_sample.csv"),
        saliva_type="cortisol",
        condition_col="condition",
        sample_times=sample_times,
    )


# def get_saliva_example_stroop(
#     sample_times: Optional[Sequence[int]] = None,
# ) -> pd.DataFrame:
#     return load_saliva_wide_format(
#         _EXAMPLE_DATA_PATH_LOCAL.joinpath("cortisol_sample_stroop.csv"),
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
    data_dict = pd.read_excel(_get_data("saliva_sample_mean_se.xlsx"), sheet_name=None)
    for key in data_dict:
        data_dict[key] = _SalivaMeanSeDataFrame(data_dict[key].set_index(["condition", "sample", "time"]))
        is_saliva_mean_se_dataframe(data_dict[key])
    return data_dict


def get_hr_subject_data_dict_example() -> HeartRateSubjectDataDict:
    """Return heart rate example data in the form of a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`
        dictionary with heart rate time-series data from multiple subjects, each containing data from different phases.

    """
    study_data_dict_hr = {}
    subject_ids = ["Vp01", "Vp02"]
    for subject_id in subject_ids:
        file_path = _get_data(f"ecg_results/hr_result_{subject_id}.xlsx")
        study_data_dict_hr[subject_id] = pd.read_excel(file_path, sheet_name=None, index_col="time")
    return study_data_dict_hr


def get_mist_hr_example() -> HeartRatePhaseDict:
    """Return heart rate time-series example data collected during MIST from one subject.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        dictionary with heart rate time-series data from one subject during multiple phases

    """
    return load_hr_phase_dict(_get_data("hr_sample_mist.xlsx"))


def get_ecg_example() -> Tuple[pd.DataFrame, float]:
    """Return raw ECG example data from one subject.

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with raw ECG data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_dataset_nilspod(file_path=_get_data("ecg/ecg_sample_Vp01.bin"), datastreams=["ecg"])


def get_ecg_example_02() -> Tuple[pd.DataFrame, float]:
    """Return second raw ECG example data from another subject.

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with raw ECG data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_dataset_nilspod(file_path=_get_data("ecg/ecg_sample_Vp02.bin"), datastreams=["ecg"])


def get_sleep_analyzer_raw_example(
    split_into_nights: Optional[bool] = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return Withings Sleep Analyzer example raw data.

    Parameters
    ----------
    split_into_nights : bool, optional
        ``True`` to split data into single dataframes per recording night, and return a dict of dataframes,
        ``False`` to keep all data in one dataframe.
        Default: ``True``

    Returns
    -------
    :class:`~pandas.DataFrame` or dict
        dataframe with raw sleep analyzer data or a dict of such if ``split_into_nights`` is ``True``

    """
    # ensure that all files are available
    file_list = [
        "raw_sleep-monitor_hr.csv",
        "raw_sleep-monitor_respiratory-rate.csv",
        "raw_sleep-monitor_sleep-state.csv",
        "raw_sleep-monitor_snoring.csv",
    ]
    file_path = None
    for file in file_list:
        file_path = _get_data(f"sleep/{file}")
    # get parent directory
    file_path = file_path.parent
    return load_withings_sleep_analyzer_raw_folder(file_path, split_into_nights=split_into_nights)


def get_sleep_analyzer_summary_example() -> SleepEndpointDataFrame:
    """Return Withings Sleep Analyzer example summary data.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDataFrame`
        dataframe with example sleep endpoints computed from Withings Sleep Analyzer Summary data

    """
    return load_withings_sleep_analyzer_summary(_get_data("sleep/sleep.csv"))


def get_sleep_imu_example() -> Tuple[pd.DataFrame, float]:
    """Return raw IMU example data collected from a wrist-worn IMU sensor during night.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with raw IMU data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_dataset_nilspod(file_path=_get_data("sleep_imu/sleep_imu_sample_01.bin"))


def get_eeg_example() -> Tuple[pd.DataFrame, float]:
    """Return raw EEG example data collected from a Muse EEG headband.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with raw EEG data
    sampling_rate : float
        sampling rate of recorded data

    """
    return load_eeg_raw_muse(_get_data("eeg_muse_example.csv"))


def get_car_watch_log_data_example() -> pd.DataFrame:
    """Return *CARWatch App* example log data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with example log data from the *CARWatch* app

    """
    # ensure that all files are available
    file_list = [
        "carwatch_de34f_20191205.csv",
        "carwatch_de34f_20191206.csv",
        "carwatch_de34f_20191207.csv",
        "carwatch_de34f_20191208.csv",
    ]
    file_path = None
    for file in file_list:
        file_path = _get_data(f"log_data/DE34F/{file}")
    # get parent directory
    file_path = file_path.parent
    return load_log_one_subject(file_path)


def get_time_log_example() -> pd.DataFrame:
    """Return time log example data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with example time log information. The time log match the data from the two ECG data example
        functions :func:`~biopsykit.example_data.get_ecg_example` and :func:`~biopsykit.example_data.get_ecg_example_02`

    """
    return load_time_log(_get_data("ecg_time_log.xlsx"))


def get_questionnaire_example_wrong_range() -> pd.DataFrame:
    """Return questionnaire example data with score in the wrong range.

    In this example the items of the "PSS" questionnaire are coded in the wrong range ([1, 5] instead of [0, 4])
    originally defined in the paper.

    This example data is used to demonstrate BioPsyKit's feature of asserting that questionnaire score items are
    provided in the correct score range according to the original definition of the questionnaire.


    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with questionnaire data where the items of the PSS questionnaire are coded in the wrong range

    """
    return load_questionnaire_data(_get_data("questionnaire_sample_wrong_range.csv"))


def get_questionnaire_example() -> pd.DataFrame:
    """Return questionnaire example data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with questionnaire data

    """
    return load_questionnaire_data(_get_data("questionnaire_sample.csv"))
