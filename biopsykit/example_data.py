from pathlib import Path
from typing import Sequence, Dict, Optional, Tuple

import pandas as pd

_EXAMPLE_DATA_PATH = Path(__file__).parent.parent.joinpath("example_data")

__all__ = [
    'list_example_data',
    'get_file_path',
    'get_saliva_example',
    'get_saliva_mean_se_example',
    'get_mist_hr_example',
    'get_condition_list_example',
    'get_sleep_analyzer_raw_example',
    'get_sleep_analyzer_summary_example',
]


def list_example_data() -> Sequence[str]:
    file_list = [filename.name for filename in _EXAMPLE_DATA_PATH.glob("*") if filename.suffix in ['.csv', '.xlsx']]
    return file_list


def get_file_path(file_name: str) -> Path:
    file_path = _EXAMPLE_DATA_PATH.joinpath(file_name)
    if file_path.is_file():
        # file exists
        return file_path
    raise ValueError("File {} does not exist!".format(file_name))


def get_condition_list_example() -> pd.DataFrame:
    from biopsykit.io import load_subject_condition_list
    return load_subject_condition_list(_EXAMPLE_DATA_PATH.joinpath("condition_list.csv"), subject_col='subject',
                                       condition_col='condition')


def get_saliva_example(saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    from biopsykit.saliva.io import load_saliva
    return load_saliva(_EXAMPLE_DATA_PATH.joinpath("cortisol_sample.csv"), biomarker_type='cortisol',
                       saliva_times=saliva_times)


def get_saliva_mean_se_example() -> Dict[str, pd.DataFrame]:
    return pd.read_excel(_EXAMPLE_DATA_PATH.joinpath("saliva_sample_mean_se.xlsx"), sheet_name=None, index_col='time')


def get_mist_hr_example() -> Dict[str, pd.DataFrame]:
    from biopsykit.signals.ecg.io import load_hr_subject_dict
    return load_hr_subject_dict(_EXAMPLE_DATA_PATH.joinpath("hr_sample_mist.xlsx"))


def get_ecg_example() -> Tuple[pd.DataFrame, int]:
    from biopsykit.io import load_dataset_nilspod
    return load_dataset_nilspod(file_path=_EXAMPLE_DATA_PATH.joinpath("ecg_sample.bin"), datastreams=['ecg'])



def get_sleep_analyzer_raw_example() -> pd.DataFrame:
    from biopsykit.sleep.io import load_withings_sleep_analyzer_raw_folder

    return load_withings_sleep_analyzer_raw_folder(_EXAMPLE_DATA_PATH.joinpath("sleep"))


def get_sleep_analyzer_summary_example() -> pd.DataFrame:
    from biopsykit.sleep.io import load_withings_sleep_analyzer_summary

    return load_withings_sleep_analyzer_summary(_EXAMPLE_DATA_PATH.joinpath("sleep").joinpath("sleep.csv"))
