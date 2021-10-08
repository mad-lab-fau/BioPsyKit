from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from biopsykit.io.ecg import load_hr_phase_dict, load_hr_phase_dict_folder, write_hr_phase_dict
from biopsykit.utils.datatype_helper import is_hr_phase_dict, is_hr_subject_data_dict
from biopsykit.utils.exceptions import FileExtensionError, ValidationError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data")


@contextmanager
def does_not_raise():
    yield


def hr_phase_dict_correct():
    phases = ["phase1", "phase2", "phase3"]
    df = pd.DataFrame(columns=["Heart_Rate"], index=pd.Index(range(0, 5), name="time"))
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


def hr_phase_dict_datetimeindex_correct():
    phases = ["phase1", "phase2", "phase3"]
    df = pd.DataFrame(
        columns=["Heart_Rate"],
        index=pd.DatetimeIndex(
            pd.to_datetime(["00:01 12.02.2021", "00:02 12.02.2021", "00:03 12.02.2021", "00:04 12.02.2021"]),
            name="time",
        ),
    )
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


def hr_phase_dict_none():
    return None


def hr_phase_dict_wrong_col_name():
    phases = ["phase1", "phase2", "phase3"]
    df = pd.DataFrame(columns=["hr"], index=range(0, 5))
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


def hr_phase_dict_too_many_columns():
    phases = ["phase1", "phase2", "phase3"]
    df = pd.DataFrame(columns=["Heart_Rate", "test"], index=range(0, 5))
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


def hr_phase_dict_wrong_col_index():
    phases = ["phase1", "phase2", "phase3"]
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([["hr"], [1, 2, 3]], names=["Heart_Rate", "number"]), index=range(0, 5)
    )
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


def hr_phase_dict_wrong_index_name():
    phases = ["phase1", "phase2", "phase3"]
    index = pd.Index(range(0, 5), name="Time")
    df = pd.DataFrame(columns=["Heart_Rate"], index=index)
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


def hr_phase_dict_wrong_index_levels():
    phases = ["phase1", "phase2", "phase3"]
    index = pd.MultiIndex.from_product([["phase1"], range(0, 5)], names=["Phase", "Time"])
    df = pd.DataFrame(columns=["Heart_Rate"], index=index)
    hr_phase_dict = {key: df for key in phases}
    return hr_phase_dict


class TestIoEcg:
    @pytest.mark.parametrize(
        "input_data, expected",
        [
            (hr_phase_dict_none(), pytest.raises(ValidationError)),
            (hr_phase_dict_wrong_col_name(), pytest.raises(ValidationError)),
            (hr_phase_dict_too_many_columns(), pytest.raises(ValidationError)),
            (hr_phase_dict_wrong_col_index(), pytest.raises(ValidationError)),
            (hr_phase_dict_wrong_index_name(), pytest.raises(ValidationError)),
            (hr_phase_dict_wrong_index_levels(), pytest.raises(ValidationError)),
            (hr_phase_dict_correct(), does_not_raise()),
        ],
        ids=[
            "none",
            "wrong_col_names",
            "too_many_cols",
            "wrong_col_idx",
            "wrong_idx_name",
            "wrong_idx_levels",
            "correct",
        ],
    )
    def test_check_data_format_invalid_exception(self, input_data, expected):
        with expected:
            is_hr_phase_dict(data=input_data, raise_exception=True)

    @pytest.mark.parametrize(
        "input_data, expected",
        [
            (hr_phase_dict_none(), False),
            (hr_phase_dict_wrong_col_name(), False),
            (hr_phase_dict_too_many_columns(), False),
            (hr_phase_dict_wrong_col_index(), False),
            (hr_phase_dict_wrong_index_name(), False),
            (hr_phase_dict_wrong_index_levels(), False),
            (hr_phase_dict_correct(), True),
        ],
        ids=[
            "none",
            "wrong_col_names",
            "too_many_cols",
            "wrong_col_idx",
            "wrong_idx_name",
            "wrong_idx_levels",
            "correct",
        ],
    )
    def test_check_data_format_invalid_bool(self, input_data, expected):
        assert is_hr_phase_dict(data=input_data, raise_exception=False) == expected

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("hr_result.xlsx", does_not_raise()),
            ("hr_result_wrong_format.xlsx", pytest.raises(ValidationError)),
            ("abc.xlsx", pytest.raises(FileNotFoundError)),
        ],
    )
    def test_load_hr_subject_dict(self, filename, expected):
        with expected:
            load_hr_phase_dict(TEST_FILE_PATH.joinpath(filename))

    @pytest.mark.parametrize(
        "base_path, filename, expected",
        [
            ("load_hr_subject_dict_folder", r"hr_result_(\w+).xlsx", does_not_raise()),
            ("load_hr_subject_dict_folder", "ecg_result.xlsx", pytest.raises(FileNotFoundError)),
            ("test", r"hr_result_(\w+).xlsx", pytest.raises(FileNotFoundError)),
        ],
    )
    def test_load_hr_subject_dict_folder(self, base_path, filename, expected):
        with expected:
            dict_hr_subject = load_hr_phase_dict_folder(TEST_FILE_PATH.joinpath(base_path), filename)
            assert len(dict_hr_subject) == 2
            assert list(dict_hr_subject.keys()) == ["Vp01", "Vp02"]
            assert all([is_hr_phase_dict(dict_phase, raise_exception=False) for dict_phase in dict_hr_subject.values()])

    @pytest.mark.parametrize(
        "base_path, filename, subfolder, expected",
        [
            ("load_hr_subject_dict_folder", r"hr_result_(\w+).xlsx", "Vp*", does_not_raise()),
            ("load_hr_subject_dict_folder", "hr_result_*.xlsx", "Vp*", does_not_raise()),
            ("load_hr_subject_dict_folder", "hr_result_*.xlsx", "Ab*", pytest.raises(FileNotFoundError)),
        ],
    )
    def test_load_hr_subject_dict_folder_subfolder(self, base_path, filename, subfolder, expected):
        with expected:
            dict_hr_subject = load_hr_phase_dict_folder(
                TEST_FILE_PATH.joinpath(base_path), filename, subfolder_pattern=subfolder
            )
            assert len(dict_hr_subject) == 2
            assert list(dict_hr_subject.keys()) == ["Vp03", "Vp04"]
            assert all([is_hr_phase_dict(dict_phase, raise_exception=False) for dict_phase in dict_hr_subject.values()])

    @pytest.mark.parametrize(
        "base_path, filename, subfolder",
        [
            ("load_hr_subject_dict_folder", "hr_result_*.xlsx", "Pb*"),
        ],
    )
    def test_load_hr_subject_dict_folder_subfolder_multiple_filed(self, base_path, filename, subfolder):
        with pytest.warns(UserWarning):
            dict_hr_subject = load_hr_phase_dict_folder(
                TEST_FILE_PATH.joinpath(base_path), filename, subfolder_pattern=subfolder
            )
            assert len(dict_hr_subject) == 2
            assert list(dict_hr_subject.keys()) == ["Pb01", "Pb02"]
            assert all([is_hr_phase_dict(dict_phase, raise_exception=False) for dict_phase in dict_hr_subject.values()])
            assert all([list(dict_phase.keys()) == ["Part1", "Part2"] for dict_phase in dict_hr_subject.values()])

    @pytest.mark.parametrize(
        "data, filename, expected",
        [
            (hr_phase_dict_correct(), "test.xlsx", does_not_raise()),
            (hr_phase_dict_datetimeindex_correct(), "test.xlsx", does_not_raise()),
            (hr_phase_dict_wrong_index_levels(), "test.xlsx", pytest.raises(ValidationError)),
            (hr_phase_dict_wrong_index_levels(), "", pytest.raises(FileExtensionError)),
            (hr_phase_dict_correct(), "test.csv", pytest.raises(FileExtensionError)),
        ],
    )
    def test_write_hr_subject_dict(self, data, filename, expected, tmp_path):
        with expected:
            write_hr_phase_dict(data, tmp_path.joinpath(filename))
