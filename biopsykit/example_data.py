from pathlib import Path
from typing import Sequence, Dict

import pandas as pd

_EXAMPLE_DATA_PATH = Path(__file__).parent.parent.joinpath("example_data")

__all__ = [
    'list_example_data',
    'get_file_path',
    'get_saliva_example',
    'get_mist_hr_example'
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


def get_saliva_example():
    from biopsykit.saliva.io import load_saliva
    return load_saliva(_EXAMPLE_DATA_PATH.joinpath("cortisol_sample.csv"), biomarker_type='cortisol')


def get_mist_hr_example() -> Dict[str, pd.DataFrame]:
    from biopsykit.signals.ecg.io import load_hr_subject_dict
    return load_hr_subject_dict(_EXAMPLE_DATA_PATH.joinpath("hr_sample_mist.xlsx"))

