from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nilspodlib import Dataset
from pandas._testing import assert_frame_equal, assert_index_equal
from pytz import UnknownTimeZoneError

from biopsykit.io import (
    convert_time_log_datetime,
    load_questionnaire_data,
    load_subject_condition_list,
    write_result_dict,
)
from biopsykit.io.io import load_time_log
from biopsykit.utils.exceptions import FileExtensionError, ValidationError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data")


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
        [
            ["12:32:05", "12:33:10", "12:35:00", "12:39:45", "12:42:00"],
            ["12:54:50", "12:55:50", "12:57:55", "12:58:00", "13:02:10"],
        ],
        columns=pd.Index(["Baseline", "Intervention", "Stress", "Recovery", "End"], name="phase"),
        index=pd.MultiIndex.from_tuples(
            [("Vp01", "Intervention"), ("Vp02", "Control")], names=["subject", "condition"]
        ),
    )
    return df


def time_log_not_continuous_correct():
    df = pd.DataFrame(
        [["12:32:05", "12:33:10", "12:35:00", "12:39:45"], ["12:54:50", "12:55:50", "12:57:55", "12:58:00"]],
        columns=pd.MultiIndex.from_product([["phase1", "phase2"], ["start", "end"]], names=["phase", "time"]),
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


def questionnaire_data_no_remove():
    return pd.DataFrame(
        {
            "PSS_01": [4, 2, 3, -66, np.nan],
            "PANAS_01": [2, 1, 4, -66, np.nan],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["Vp01", "Vp02", "Vp03", "Vp04", "Vp05"],
                ["Control", "Intervention", "Intervention", "Control", "Control"],
                ["pre", "pre", "post", "post", "pre"],
            ],
            names=["subject", "condition", "time"],
        ),
    )


def questionnaire_data_replace_missing():
    return pd.DataFrame(
        {
            "PSS_01": [4, 2, 3, np.nan, np.nan],
            "PANAS_01": [2, 1, 4, np.nan, np.nan],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["Vp01", "Vp02", "Vp03", "Vp04", "Vp05"],
                ["Control", "Intervention", "Intervention", "Control", "Control"],
                ["pre", "pre", "post", "post", "pre"],
            ],
            names=["subject", "condition", "time"],
        ),
    )


def questionnaire_data_remove_nan():
    return pd.DataFrame(
        {
            "PSS_01": [4, 2, 3, -66],
            "PANAS_01": [2, 1, 4, -66],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["Vp01", "Vp02", "Vp03", "Vp04"],
                ["Control", "Intervention", "Intervention", "Control"],
                ["pre", "pre", "post", "post"],
            ],
            names=["subject", "condition", "time"],
        ),
    )


def questionnaire_data_replace_missing_remove_nan():
    return pd.DataFrame(
        {
            "PSS_01": [4, 2, 3],
            "PANAS_01": [2, 1, 4],
        },
        index=pd.MultiIndex.from_arrays(
            [["Vp01", "Vp02", "Vp03"], ["Control", "Intervention", "Intervention"], ["pre", "pre", "post"]],
            names=["subject", "condition", "time"],
        ),
    )


def result_dict_correct():
    return {
        "Vp01": pd.DataFrame(columns=["data"], index=pd.Index(range(0, 2), name="time")),
        "Vp02": pd.DataFrame(columns=["data"], index=pd.Index(range(0, 2), name="time")),
        "Vp03": pd.DataFrame(columns=["data"], index=pd.Index(range(0, 2), name="time")),
    }


class TestIoIo:
    @pytest.mark.parametrize(
        "file_path, continuous_time, subject_col, condition_col, additional_index_cols, phase_cols, expected",
        [
            ("time_log.csv", True, None, None, None, None, does_not_raise()),
            ("time_log.xlsx", True, None, None, None, None, does_not_raise()),
            ("time_log.csv", True, "time", None, None, None, pytest.raises(ValidationError)),
            ("time_log.csv", True, "subject", None, None, None, does_not_raise()),
            ("time_log.csv", True, "subject", "condition", None, None, does_not_raise()),
            ("time_log.csv", True, ["subject"], None, None, None, pytest.raises(ValidationError)),
            ("time_log.csv", True, "subject", ["condition"], None, None, pytest.raises(ValidationError)),
            ("time_log.csv", True, "subject", None, ["time"], None, pytest.raises(ValidationError)),
            (
                "time_log.csv",
                True,
                None,
                None,
                None,
                ["Baseline", "Intervention", "Stress", "Recovery"],
                does_not_raise(),
            ),
            (
                "time_log.csv",
                True,
                None,
                None,
                None,
                ["Baseline", "Intervention", "Stress", "Postline"],
                pytest.raises(ValidationError),
            ),
            ("time_log_other_index_names.csv", True, None, None, None, None, pytest.raises(ValidationError)),
            (
                "time_log_other_index_names.csv",
                True,
                "subject",
                "condition",
                None,
                None,
                pytest.raises(ValidationError),
            ),
            (
                "time_log_other_index_names.csv",
                True,
                "ID",
                "Condition",
                None,
                None,
                does_not_raise(),
            ),
            ("time_log_not_continuous.csv", True, None, None, None, None, does_not_raise()),
            ("time_log_not_continuous.csv", False, None, None, None, None, does_not_raise()),
            ("time_log.xlsx", False, None, None, None, None, pytest.raises(ValidationError)),
            ("time_log_wrong_column_names.csv", False, None, None, None, None, pytest.raises(ValidationError)),
        ],
    )
    def test_load_time_log_raises(
        self, file_path, continuous_time, subject_col, condition_col, additional_index_cols, phase_cols, expected
    ):
        with expected:
            load_time_log(
                file_path=TEST_FILE_PATH.joinpath(file_path),
                subject_col=subject_col,
                condition_col=condition_col,
                additional_index_cols=additional_index_cols,
                phase_cols=phase_cols,
                continuous_time=continuous_time,
            )

    @pytest.mark.parametrize(
        "file_path, subject_col, condition_col, additional_index_cols, phase_cols, continuous_time, expected",
        [
            ("time_log.csv", None, None, None, None, True, time_log_correct()),
            ("time_log.csv", "subject", "condition", None, None, True, time_log_correct()),
            (
                "time_log_other_index_names.csv",
                "ID",
                "Condition",
                None,
                None,
                True,
                time_log_correct(),
            ),
            (
                "time_log_not_continuous_other_index_names.csv",
                "ID",
                "Condition",
                None,
                None,
                False,
                time_log_not_continuous_correct(),
            ),
            (
                "time_log.csv",
                "subject",
                "condition",
                None,
                ["Baseline", "Intervention", "Stress", "Recovery", "End"],
                True,
                time_log_correct(),
            ),
            (
                "time_log_other_column_names.csv",
                "subject",
                "condition",
                None,
                {
                    "Phase1": "Baseline",
                    "Phase2": "Intervention",
                    "Phase3": "Stress",
                    "Phase4": "Recovery",
                    "End": "End",
                },
                True,
                time_log_correct(),
            ),
        ],
    )
    def test_load_time_log_index_cols(
        self, file_path, subject_col, condition_col, additional_index_cols, phase_cols, continuous_time, expected
    ):
        data_out = load_time_log(
            file_path=TEST_FILE_PATH.joinpath(file_path),
            subject_col=subject_col,
            condition_col=condition_col,
            additional_index_cols=additional_index_cols,
            phase_cols=phase_cols,
            continuous_time=continuous_time,
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
            subject_col="subject",
            condition_col="condition",
            continuous_time=continuous_time,
        )
        assert_frame_equal(data_out, expected)

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

    @pytest.mark.parametrize(
        "file_path, subject_col, condition_col, additional_index_cols, expected",
        [
            ("questionnaire_data.xlsx", None, None, None, does_not_raise()),
            ("questionnaire_data.csv", None, None, None, does_not_raise()),
            ("questionnaire_data.csv", "subject", None, None, does_not_raise()),
            ("questionnaire_data.csv", "subject", "condition", None, does_not_raise()),
            ("questionnaire_data.csv", "subject", "condition", "time", does_not_raise()),
            ("questionnaire_data.csv", "subject", "condition", ["time"], does_not_raise()),
            ("questionnaire_data_other_column_names.csv", "ID", "Group", None, does_not_raise()),
            ("questionnaire_data_other_column_names.csv", "ID", "condition", None, pytest.raises(ValidationError)),
            ("questionnaire_data_other_column_names.csv", "subject", "Group", None, pytest.raises(ValidationError)),
            ("questionnaire_data_other_column_names.csv", "ID", "Group", "test", pytest.raises(ValidationError)),
            ("questionnaire_data_other_column_names.csv", "ID", "Group", "time", does_not_raise()),
        ],
    )
    def test_load_questionnaire_data_raises(
        self, file_path, subject_col, condition_col, additional_index_cols, expected
    ):
        with expected:
            load_questionnaire_data(
                TEST_FILE_PATH.joinpath(file_path),
                subject_col=subject_col,
                condition_col=condition_col,
                additional_index_cols=additional_index_cols,
            )

    @pytest.mark.parametrize(
        "file_path, replace_missing_vals, remove_nan_rows, expected",
        [
            ("questionnaire_data.csv", False, False, questionnaire_data_no_remove()),
            ("questionnaire_data.csv", True, False, questionnaire_data_replace_missing()),
            ("questionnaire_data.csv", False, True, questionnaire_data_remove_nan()),
            ("questionnaire_data.csv", True, True, questionnaire_data_replace_missing_remove_nan()),
            ("questionnaire_data.xlsx", False, False, questionnaire_data_no_remove()),
            ("questionnaire_data.xlsx", True, False, questionnaire_data_replace_missing()),
            ("questionnaire_data.xlsx", False, True, questionnaire_data_remove_nan()),
            ("questionnaire_data.xlsx", True, True, questionnaire_data_replace_missing_remove_nan()),
        ],
    )
    def test_load_questionnaire_data(self, file_path, replace_missing_vals, remove_nan_rows, expected):
        data_out = load_questionnaire_data(
            file_path=TEST_FILE_PATH.joinpath(file_path),
            subject_col="subject",
            condition_col="condition",
            additional_index_cols="time",
            replace_missing_vals=replace_missing_vals,
            remove_nan_rows=remove_nan_rows,
        )

        assert_frame_equal(data_out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "time_log, dataset, df, date, timezone, expected",
        [
            (time_log_correct(), None, None, None, None, pytest.raises(ValueError)),
            (
                time_log_correct(),
                Dataset.from_bin_file(TEST_FILE_PATH.joinpath("dataset_example.bin")),
                None,
                None,
                None,
                does_not_raise(),
            ),
            (time_log_correct(), None, pd.DataFrame(index=range(0, 10)), None, None, pytest.raises(ValueError)),
            (
                time_log_correct(),
                None,
                pd.DataFrame(index=pd.to_datetime(["00:01 12.02.2021", "00:02 12.02.2021", "00:03 12.02.2021"])),
                None,
                None,
                does_not_raise(),
            ),
            (time_log_correct(), None, None, "hello", None, pytest.raises(ValueError)),
            (time_log_correct(), None, None, "12.02.2021", None, does_not_raise()),
            (time_log_correct(), None, None, "12.02.2021", None, does_not_raise()),
            (time_log_correct(), None, None, "12/02/2021", "Europe/Berlin", does_not_raise()),
            (time_log_correct(), None, None, "12/02/2021", "test", pytest.raises(UnknownTimeZoneError)),
        ],
    )
    def test_convert_time_log_datetime_raises(self, time_log, dataset, df, date, timezone, expected):
        with expected:
            convert_time_log_datetime(time_log=time_log, dataset=dataset, df=df, date=date, timezone=timezone)

    @pytest.mark.parametrize(
        "data, filename, expected",
        [
            (result_dict_correct(), "test.csv", does_not_raise()),
            (result_dict_correct(), "test.xlsx", does_not_raise()),
            (result_dict_correct(), "test.txt", pytest.raises(FileExtensionError)),
        ],
    )
    def test_write_result_dict(self, data, filename, expected, tmp_path):
        with expected:
            write_result_dict(data, tmp_path.joinpath(filename))
