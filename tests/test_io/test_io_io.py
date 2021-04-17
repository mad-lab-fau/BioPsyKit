from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from biopsykit.utils.exceptions import ValidationError, FileExtensionError

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
