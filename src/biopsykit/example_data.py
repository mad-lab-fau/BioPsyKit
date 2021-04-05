from pathlib import Path
from typing import Sequence, Dict, Optional, Tuple, Union

import pandas as pd

_EXAMPLE_DATA_PATH = Path(__file__).parent.parent.parent.joinpath("example_data")

__all__ = [
    "list_example_data",
    "get_file_path",
    "get_saliva_example",
    "get_saliva_mean_se_example",
    "get_mist_hr_example",
    "get_condition_list_example",
    "get_sleep_analyzer_raw_example",
    "get_sleep_analyzer_summary_example",
]


def list_example_data() -> Sequence[str]:
    file_list = [filename.name for filename in _EXAMPLE_DATA_PATH.glob("*") if filename.suffix in [".csv", ".xlsx"]]
    return file_list


def get_file_path(file_name: str) -> Path:
    file_path = _EXAMPLE_DATA_PATH.joinpath(file_name)
    if file_path.is_file():
        # file exists
        return file_path
    raise ValueError("File {} does not exist!".format(file_name))


def get_condition_list_example() -> pd.DataFrame:
    from biopsykit.io import load_subject_condition_list

    return load_subject_condition_list(
        _EXAMPLE_DATA_PATH.joinpath("condition_list.csv"),
        subject_col="subject",
        condition_col="condition",
    )


def get_saliva_example(saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    from biopsykit.io.saliva import load_saliva_wide_format

    return load_saliva_wide_format(
        _EXAMPLE_DATA_PATH.joinpath("cortisol_sample.csv"),
        biomarker_name="cortisol",
        condition_col="condition",
        saliva_times=saliva_times,
    )


def get_saliva_example_stroop(
    saliva_times: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    from biopsykit.io.saliva import load_saliva_wide_format

    return load_saliva_wide_format(
        _EXAMPLE_DATA_PATH.joinpath("cortisol_sample_stroop.csv"),
        biomarker_name="cortisol",
        saliva_times=saliva_times,
    )


def get_saliva_mean_se_example() -> Dict[str, pd.DataFrame]:
    return pd.read_excel(
        _EXAMPLE_DATA_PATH.joinpath("saliva_sample_mean_se.xlsx"),
        sheet_name=None,
        index_col="time",
    )


def get_mist_hr_example() -> Dict[str, pd.DataFrame]:
    from biopsykit.io.ecg import load_hr_subject_dict

    return load_hr_subject_dict(_EXAMPLE_DATA_PATH.joinpath("hr_sample_mist.xlsx"))


def get_ecg_example() -> Tuple[pd.DataFrame, int]:
    from biopsykit.io.nilspod import load_dataset_nilspod

    return load_dataset_nilspod(
        file_path=_EXAMPLE_DATA_PATH.joinpath("ecg").joinpath("ecg_sample_Vp01.bin"),
        datastreams=["ecg"],
    )


def get_ecg_example_02() -> Tuple[pd.DataFrame, int]:
    from biopsykit.io.nilspod import load_dataset_nilspod

    return load_dataset_nilspod(
        file_path=_EXAMPLE_DATA_PATH.joinpath("ecg").joinpath("ecg_sample_Vp02.bin"),
        datastreams=["ecg"],
    )


def get_sleep_analyzer_raw_example(
    split_nights: Optional[bool] = True,
) -> Union[pd.DataFrame, Sequence[pd.DataFrame]]:
    from biopsykit.io.sleep import load_withings_sleep_analyzer_raw_folder

    return load_withings_sleep_analyzer_raw_folder(_EXAMPLE_DATA_PATH.joinpath("sleep"), split_nights=split_nights)


def get_sleep_analyzer_summary_example() -> pd.DataFrame:
    from biopsykit.io.sleep import load_withings_sleep_analyzer_summary

    return load_withings_sleep_analyzer_summary(_EXAMPLE_DATA_PATH.joinpath("sleep").joinpath("sleep.csv"))


def get_sleep_imu_example() -> Tuple[pd.DataFrame, int]:
    from biopsykit.io.nilspod import load_dataset_nilspod

    try:
        return load_dataset_nilspod(
            file_path=_EXAMPLE_DATA_PATH.joinpath("sleep_imu").joinpath("sleep_imu_sample_01.bin")
        )
    except:
        raise ValueError(
            "Dataset can not be loaded. Checking out this large binary file requires Git "
            "Large File Storage ('git-lfs'), probably don't have it installed. "
            "Please install 'git-lfs' (https://git-lfs.github.com/) and check out the repository again!"
        )


def get_eeg_example() -> Tuple[pd.DataFrame, int]:
    from biopsykit.io.eeg import load_eeg_raw_muse

    return load_eeg_raw_muse(_EXAMPLE_DATA_PATH.joinpath("eeg_muse_example.csv"))


def get_log_data_example() -> pd.DataFrame:
    from biopsykit.io.carwatch_logs import load_log_one_subject

    return load_log_one_subject(_EXAMPLE_DATA_PATH.joinpath("log_data").joinpath("AB12C"))


def get_time_log_example() -> pd.DataFrame:
    from biopsykit.io import load_time_log

    return load_time_log(_EXAMPLE_DATA_PATH.joinpath("ecg_time_log.xlsx"))


def get_questionnaire_example() -> pd.DataFrame:
    return pd.read_csv(_EXAMPLE_DATA_PATH.joinpath("questionnaire_sample.csv"), index_col=["subject"])
