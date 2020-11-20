from pathlib import Path
from typing import Sequence

EXAMPLE_DATA_PATH = Path(__file__).parent.parent.joinpath("example_data")


def list_example_data() -> Sequence[str]:
    file_list = [filename.name for filename in EXAMPLE_DATA_PATH.glob("*") if filename.suffix in ['.csv', '.xlsx']]
    return file_list


def get_file_path(file_name: str) -> Path:
    file_path = EXAMPLE_DATA_PATH.joinpath(file_name)
    if file_path.is_file():
        # file exists
        return file_path
    raise ValueError("File {} does not exist!".format(file_name))


def get_saliva_example():
    from biopsykit.saliva.io import load_saliva
    return load_saliva(EXAMPLE_DATA_PATH.joinpath("cortisol_sample.csv"))
    pass


def get_mist_hr_example():
    pass
