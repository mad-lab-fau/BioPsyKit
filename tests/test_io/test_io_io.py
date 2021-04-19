from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biopsykit.io import load_subject_condition_list
from biopsykit.utils.exceptions import ValidationError

from biopsykit.io.io import load_time_log
from pandas._testing import assert_frame_equal, assert_index_equal

TEST_FILE_PATH = Path("../data/test_files")


@contextmanager
def does_not_raise():
    yield


def time_log_no_index():
    df = pd.DataFrame(
        columns=["subject", "condition", "Baseline", "Intervention", "Stress", "Recovery", "End"], index=range(0, 2)
    )
    return df


def time_log_correct():
    df = pd.DataFrame(
        columns=["Baseline", "Intervention", "Stress", "Recovery", "End"],
        index=pd.MultiIndex.from_tuples(
            [("Vp01", "Intervention"), ("Vp02", "Control")], names=["subject", "condition"]
        ),
    )
    return df


def time_log_not_continuous_correct():
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([["phase1", "phase2"], ["start", "end"]], names=[None, "time"]),
        index=pd.MultiIndex.from_tuples(
            [("Vp01", "Intervention"), ("Vp02", "Control")], names=["subject", "condition"]
        ),
    )
    return df


def subject_condition_list_correct():
    return pd.DataFrame(
        ["Intervention", "Control", "Control", "Intervention"],
        columns=["condition"],
        index=pd.Index(["Vp01", "Vp02", "Vp03", "Vp04"], name="subject"),
    )


def subject_condition_list_correct_dict():
    return {"Control": ["Vp02", "Vp03"], "Intervention": ["Vp01", "Vp04"]}


class TestIoIo:
    @pytest.mark.parametrize(
        "file_path, continuous_time, index_cols, phase_cols, expected",
        [
            ("time_log.csv", True, None, None, does_not_raise()),
            ("time_log.xlsx", True, None, None, does_not_raise()),
            ("time_log.csv", True, ["time"], None, pytest.raises(ValidationError)),
            ("time_log.csv", True, "subject", None, does_not_raise()),
            ("time_log.csv", True, ["subject", "time"], None, pytest.raises(ValidationError)),
            ("time_log.csv", True, None, ["Baseline", "Intervention", "Stress", "Recovery"], does_not_raise()),
            (
                "time_log.csv",
                True,
                None,
                ["Baseline", "Intervention", "Stress", "Postline"],
                pytest.raises(ValidationError),
            ),
            ("time_log_other_index_names.csv", True, None, None, does_not_raise()),
            ("time_log_other_index_names.csv", True, ["subject", "condition"], None, pytest.raises(ValidationError)),
            (
                "time_log_other_index_names.csv",
                True,
                {"ID": "subject", "Condition": "condition"},
                None,
                does_not_raise(),
            ),
            ("time_log_not_continuous.csv", True, None, None, does_not_raise()),
            ("time_log_not_continuous.csv", False, None, None, does_not_raise()),
            ("time_log.xlsx", False, None, None, pytest.raises(ValidationError)),
            ("time_log_wrong_column_names.csv", False, None, None, pytest.raises(ValidationError)),
        ],
    )
    def test_load_time_log_raises(self, file_path, continuous_time, index_cols, phase_cols, expected):
        with expected:
            load_time_log(
                file_path=TEST_FILE_PATH.joinpath(file_path),
                index_cols=index_cols,
                phase_cols=phase_cols,
                continuous_time=continuous_time,
            )

    @pytest.mark.parametrize(
        "file_path, index_cols, phase_cols, expected",
        [
            ("time_log.csv", None, None, time_log_no_index()),
            ("time_log.csv", ["subject", "condition"], None, time_log_correct()),
            ("time_log_other_index_names.csv", {"ID": "subject", "Condition": "condition"}, None, time_log_correct()),
            (
                "time_log.csv",
                ["subject", "condition"],
                ["Baseline", "Intervention", "Stress", "Recovery", "End"],
                time_log_correct(),
            ),
            (
                "time_log_other_column_names.csv",
                ["subject", "condition"],
                {
                    "Phase1": "Baseline",
                    "Phase2": "Intervention",
                    "Phase3": "Stress",
                    "Phase4": "Recovery",
                    "End": "End",
                },
                time_log_correct(),
            ),
        ],
    )
    def test_load_time_log_index_cols(self, file_path, index_cols, phase_cols, expected):
        data_out = load_time_log(
            file_path=TEST_FILE_PATH.joinpath(file_path), index_cols=index_cols, phase_cols=phase_cols
        )
        assert_index_equal(expected.index, data_out.index)
        assert_index_equal(expected.columns, data_out.columns)

    @pytest.mark.parametrize(
        "file_path, continuous_time, expected",
        [
            ("time_log.csv", True, time_log_correct()),
            ("time_log.xlsx", True, time_log_correct()),
            ("time_log_not_continuous.csv", False, time_log_not_continuous_correct()),
        ],
    )
    def test_load_time_log(self, file_path, continuous_time, expected):
        data_out = load_time_log(
            file_path=TEST_FILE_PATH.joinpath(file_path),
            continuous_time=continuous_time,
            index_cols=["subject", "condition"],
        )
        assert_index_equal(data_out.index, expected.index)
        assert_index_equal(data_out.columns, expected.columns)

    @pytest.mark.parametrize(
        "file_path, subject_col, condition_col, expected",
        [
            ("condition_list.csv", None, None, does_not_raise()),
            ("condition_list.csv", "subject", "condition", does_not_raise()),
            ("condition_list.csv", None, "condition", does_not_raise()),
            ("condition_list.csv", "subject", None, does_not_raise()),
            ("condition_list.csv", "ID", "Group", pytest.raises(ValidationError)),
            ("condition_list_other_column_names.csv", None, None, pytest.raises(ValidationError)),
            ("condition_list_other_column_names.csv", "subject", "condition", pytest.raises(ValidationError)),
            ("condition_list_other_column_names.csv", "ID", "Group", does_not_raise()),
        ],
    )
    def test_load_subject_condition_list_raises(self, file_path, subject_col, condition_col, expected):
        with expected:
            load_subject_condition_list(
                TEST_FILE_PATH.joinpath(file_path), subject_col=subject_col, condition_col=condition_col
            )

    @pytest.mark.parametrize(
        "file_path, return_dict, expected",
        [
            ("condition_list.xlsx", False, subject_condition_list_correct()),
            ("condition_list.csv", False, subject_condition_list_correct()),
            ("condition_list.xlsx", True, subject_condition_list_correct_dict()),
            ("condition_list.csv", True, subject_condition_list_correct_dict()),
        ],
    )
    def test_load_subject_condition_list(self, file_path, return_dict, expected):
        data_out = load_subject_condition_list(TEST_FILE_PATH.joinpath(file_path), return_dict=return_dict)
        if isinstance(data_out, pd.DataFrame):
            assert_frame_equal(data_out, expected)
        else:
            assert data_out.keys() == expected.keys()
            for k1, k2 in zip(data_out, expected):
                assert np.array_equal(data_out[k1], expected[k2])

    @pytest.mark.parametrize(
        "file_path, subject_col, condition_col, expected",
        [
            ("condition_list_other_column_names.csv", "ID", "Group", subject_condition_list_correct()),
        ],
    )
    def test_load_subject_condition_list_other_columns(self, file_path, subject_col, condition_col, expected):
        data_out = load_subject_condition_list(
            TEST_FILE_PATH.joinpath(file_path), subject_col=subject_col, condition_col=condition_col
        )
        assert_frame_equal(data_out, expected)
