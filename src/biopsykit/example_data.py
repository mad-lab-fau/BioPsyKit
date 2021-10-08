"""Module providing functions to load different example data.

The data is either taken from the local file system in case biopsykit was installed manually or the example data is
downloaded into the local user folder.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
from urllib.request import urlretrieve

import pandas as pd
from tqdm.auto import tqdm

from biopsykit.io import (
    load_long_format_csv,
    load_pandas_dict_excel,
    load_questionnaire_data,
    load_subject_condition_list,
    load_time_log,
)
from biopsykit.io.carwatch_logs import load_log_one_subject
from biopsykit.io.ecg import load_hr_phase_dict
from biopsykit.io.eeg import load_eeg_raw_muse
from biopsykit.io.nilspod import load_dataset_nilspod
from biopsykit.io.saliva import load_saliva_plate, load_saliva_wide_format
from biopsykit.io.sleep_analyzer import (
    WITHINGS_RAW_DATA_SOURCES,
    load_withings_sleep_analyzer_raw_file,
    load_withings_sleep_analyzer_raw_folder,
    load_withings_sleep_analyzer_summary,
)
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import (
    HeartRatePhaseDict,
    HeartRateSubjectDataDict,
    SalivaMeanSeDataFrame,
    SalivaRawDataFrame,
    SleepEndpointDataFrame,
    SubjectConditionDataFrame,
    _SalivaMeanSeDataFrame,
    is_saliva_mean_se_dataframe,
)
from biopsykit.utils.file_handling import mkdirs

_EXAMPLE_DATA_PATH_LOCAL = Path(__file__).parent.parent.parent.joinpath("example_data")
_EXAMPLE_DATA_PATH_HOME = Path.home().joinpath(".biopsykit_data")
_REMOTE_DATA_PATH = "https://raw.githubusercontent.com/mad-lab-fau/BioPsyKit/main/example_data/"

__all__ = [
    "get_condition_list_example",
    "get_saliva_example_plate_format",
    "get_saliva_example",
    "get_saliva_mean_se_example",
    "get_mist_hr_example",
    "get_hr_result_sample",
    "get_hr_ensemble_sample",
    "get_hr_subject_data_dict_example",
    "get_ecg_processing_results_path_example",
    "get_ecg_path_example",
    "get_ecg_example",
    "get_ecg_example_02",
    "get_eeg_example",
    "get_sleep_analyzer_raw_file_unformatted",
    "get_sleep_analyzer_raw_file",
    "get_sleep_analyzer_raw_example",
    "get_sleep_analyzer_summary_example",
    "get_sleep_imu_example",
    "get_car_watch_log_path_example",
    "get_car_watch_log_data_zip_path_example",
    "get_car_watch_log_path_all_subjects_example",
    "get_car_watch_log_data_example",
    "get_time_log_example",
    "get_questionnaire_example",
    "get_questionnaire_example_wrong_range",
    "get_stats_example",
]

# TODO add SHA256 check to assert whether remote example data was changed and should be re-downloaded.


def _is_installed_manually() -> bool:
    """Check whether biopsykit was installed manually and example data exists in the local path.

    Returns
    -------
    bool
        ``True`` if biopsykit was installed manually, ``False`` otherwise

    """
    return (_EXAMPLE_DATA_PATH_LOCAL / "__init__.py").is_file()


def _get_data(file_name: str) -> path_t:  # pragma: no cover
    if _is_installed_manually():
        return _EXAMPLE_DATA_PATH_LOCAL.joinpath(file_name)
    path = _EXAMPLE_DATA_PATH_HOME.joinpath(file_name)
    if path.exists():
        return path
    mkdirs(path.parent)
    return _fetch_from_remote(file_name, path)


def _fetch_from_remote(file_name: str, file_path: path_t) -> path_t:  # pragma: no cover
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


def _tqdm_hook(t):  # pragma: no cover
    """Wrap tqdm instance."""
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def get_condition_list_example() -> SubjectConditionDataFrame:
    """Return example data for subject condition assignment.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame`
        dataframe with example subject condition assignment

    """
    return load_subject_condition_list(
        _get_data("condition_list.csv"), subject_col="subject", condition_col="condition"
    )


def get_saliva_example_plate_format(
    sample_id_col: Optional[str] = None,
    data_col: Optional[str] = None,
    id_col_names: Optional[Sequence[str]] = None,
    regex_str: Optional[str] = None,
    sample_times: Optional[Sequence[int]] = None,
    condition_list: Optional[Union[Sequence, Dict[str, Sequence], pd.Index]] = None,
) -> pd.DataFrame:
    r"""Return example saliva data from "plate" format.

    Parameters
    ----------
    sample_id_col: str, optional
        column name of the Excel sheet containing the sample ID. Default: "sample ID"
    data_col: str, optional
        column name of the Excel sheet containing saliva data to be analyzed.
        Default: Select default column name based on ``biomarker_type``, e.g. ``cortisol`` => ``cortisol (nmol/l)``
    id_col_names: list of str, optional
        names of the extracted ID column names. ``None`` to use the default column names (['subject', 'day', 'sample'])
    regex_str: str, optional
        regular expression to extract subject ID, day ID and sample ID from the sample identifier.
        ``None`` to use default regex string (``r"(Vp\d+) (S\d)"``)
    sample_times: list of int, optional
        times at which saliva samples were collected
    condition_list: 1d-array, optional
        list of conditions which subjects were assigned to

    Returns
    -------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format

    See Also
    --------
    :func:`~biopsykit.io.saliva.load_saliva_plate`
        loader function for saliva data in plate format

    """
    return load_saliva_plate(
        _get_data("cortisol_sample_plate.xlsx"),
        "cortisol",
        sample_id_col=sample_id_col,
        data_col=data_col,
        id_col_names=id_col_names,
        regex_str=regex_str,
        sample_times=sample_times,
        condition_list=condition_list,
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


def get_hr_result_sample() -> pd.DataFrame:
    """Return heart rate results example data.

    The heart rate results example data consists of the mean normalized heart rate for different subjects,
    different study phases, and study subphases.

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with heart rate results example data

    """
    return load_long_format_csv(_get_data("hr_result_sample.csv"))


def get_hr_ensemble_sample() -> Dict[str, pd.DataFrame]:
    """Return heart rate ensemble example data.

    The example data consists of time-series heart rate of multiple subjects for different study phases,
    each synchronized and resampled to 1 Hz, and normalized to baseline heart rate.

    Returns
    -------
    dict
        dictionary with pandas dataframes containing heart rate ensemble data

    """
    return load_pandas_dict_excel(_get_data("hr_ensemble_sample_normalized.xlsx"))


def get_mist_hr_example() -> HeartRatePhaseDict:
    """Return heart rate time-series example data collected during MIST from one subject.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        dictionary with heart rate time-series data from one subject during multiple phases

    """
    return load_hr_phase_dict(_get_data("hr_sample_mist.xlsx"))


def get_ecg_path_example() -> path_t:
    """Return folder path to ECG example data.

    Returns
    -------
    :class:`~pathlib.Path` or str
        path to folder with ECG raw files

    """
    subject_ids = ["Vp01", "Vp02"]
    file_path = None
    # ensure that folder exists and data in folder is available
    for subject_id in subject_ids:
        file_path = _get_data(f"ecg/ecg_sample_{subject_id}.bin")
    return file_path.parent


def get_ecg_processing_results_path_example() -> path_t:
    """Return folder path to ECG processing results.

    Returns
    -------
    :class:`~pathlib.Path` or str
        path to folder with ECG processing results

    """
    subject_ids = ["Vp01", "Vp02"]
    file_path = None
    # ensure that folder exists and data in folder is available
    for subject_id in subject_ids:
        for file_type in ["hr_result", "rpeaks_result"]:
            file_path = _get_data(f"ecg_processing_results/{file_type}_{subject_id}.xlsx")
    return file_path.parent


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


def get_sleep_analyzer_raw_file_unformatted(data_source: str) -> pd.DataFrame:
    """Return *unformatted* Withings Sleep Analyzer raw data example file.

    Parameters
    ----------
    data_source : str
        Withings Sleep Analyzer data source name.
        Must be one of ['heart_rate', 'respiration_rate', 'sleep_state', 'snoring'].

    Returns
    -------
    :class:`~pandas.DataFrame`
        Dataframe with unformatted example raw data

    """
    if data_source not in WITHINGS_RAW_DATA_SOURCES.values():
        raise ValueError(
            "Unsupported data source {}! Must be one of {}.".format(
                data_source, list(WITHINGS_RAW_DATA_SOURCES.values())
            )
        )
    ds_name = list(WITHINGS_RAW_DATA_SOURCES.keys())[list(WITHINGS_RAW_DATA_SOURCES.values()).index(data_source)]
    return pd.read_csv(_get_data(f"sleep/raw_sleep-monitor_{ds_name}.csv"))


def get_sleep_analyzer_raw_file(
    data_source: str,
    split_into_nights: Optional[bool] = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return Withings Sleep Analyzer raw data example file.

    Parameters
    ----------
    data_source : str
        Withings Sleep Analyzer data source name.
        Must be one of ['heart_rate', 'respiration_rate', 'sleep_state', 'snoring'].
    split_into_nights : bool, optional
        whether to split the dataframe into the different recording nights (and return a dictionary of dataframes)
        or not.
        Default: ``True``

    Returns
    -------
    :class:`~pandas.DataFrame` or dict of such
        dataframe (or dict of dataframes, if ``split_into_nights`` is ``True``) with Sleep Analyzer data

    """
    if data_source not in WITHINGS_RAW_DATA_SOURCES.values():
        raise ValueError(
            "Unsupported data source {}! Must be one of {}.".format(
                data_source, list(WITHINGS_RAW_DATA_SOURCES.values())
            )
        )

    ds_name = list(WITHINGS_RAW_DATA_SOURCES.keys())[list(WITHINGS_RAW_DATA_SOURCES.values()).index(data_source)]
    return load_withings_sleep_analyzer_raw_file(
        _get_data(f"sleep/raw_sleep-monitor_{ds_name}.csv"),
        data_source=data_source,
        split_into_nights=split_into_nights,
    )


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


def get_car_watch_log_path_example() -> path_t:
    """Return folder path to *CARWatch App* log files from *one* subject.

    Returns
    -------
    :class:`~pathlib.Path` or str
        path to folder with *CARWatch App* log files.

    """
    # ensure that folder exists and data in folder is available
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
    return file_path.parent


def get_car_watch_log_data_zip_path_example() -> path_t:
    """Return path to *CARWatch App* example log data as zip file from *one* subject.

    Returns
    -------
    :class:`~pathlib.Path` or str
        path to *CARWatch App* example log data as zip file

    """
    return _get_data("log_data/logs_AB12C.zip")


def get_car_watch_log_data_example() -> pd.DataFrame:
    """Return *CARWatch App* example log data from folder from *one* subject.

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


def get_car_watch_log_path_all_subjects_example() -> path_t:
    """Return folder path to *CARWatch App* log files for *multiple* subjects.

    Returns
    -------
    :class:`~pathlib.Path` or str
        path to folder with *CARWatch App* log files.

    """
    # ensure that folder exists and data in folder is available
    file_list = [
        "DE34F/carwatch_de34f_20191205.csv",
        "DE34F/carwatch_de34f_20191206.csv",
        "DE34F/carwatch_de34f_20191207.csv",
        "DE34F/carwatch_de34f_20191208.csv",
        "GH56I/carwatch_gh56i_20191205.csv",
        "GH56I/carwatch_gh56i_20191206.csv",
        "GH56I/carwatch_gh56i_20191207.csv",
        "GH56I/carwatch_gh56i_20191208.csv",
    ]
    file_path = None
    for file in file_list:
        file_path = _get_data(f"log_data/{file}")
    # get parent directory
    return file_path.parent.parent


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
        dataframe with questionnaire example data where the items of the PSS questionnaire are coded in the wrong range

    """
    return load_questionnaire_data(_get_data("questionnaire_sample_wrong_range.csv"))


def get_questionnaire_example() -> pd.DataFrame:
    """Return questionnaire example data.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with questionnaire example data

    """
    return load_questionnaire_data(_get_data("questionnaire_sample.csv"))


def get_stats_example() -> pd.DataFrame:
    """Return example data for statistical analysis.

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with example data that can be used for statistical analysis

    """
    return load_long_format_csv(_get_data("stats_sample.csv"))
