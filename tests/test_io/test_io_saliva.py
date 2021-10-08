from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_index_equal

from biopsykit.io.saliva import load_saliva_plate, load_saliva_wide_format, save_saliva
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
        columns=["Baseline", "Intervention", "Stress", "Recovery", "End"],
        index=pd.MultiIndex.from_tuples(
            [("Vp01", "Intervention"), ("Vp02", "Control")], names=["subject", "condition"]
        ),
    )
    return df


def time_log_not_continuous_correct():
    df = pd.DataFrame(
        [["12:32:05", "12:33:10", "12:35:00", "12:39:45"], ["12:54:50", "12:55:50", "12:57:55", "12:58:00"]],
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


def saliva_data_invalid_format():
    index = pd.MultiIndex.from_product([["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["ID", "value"])
    return pd.DataFrame(
        {"cortisol": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41, 0.12, 0.22, 0.32, 0.42]},
        index=index,
    )


def saliva_data_samples():
    index = pd.MultiIndex.from_product(
        [["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["subject", "sample"]
    )
    return pd.DataFrame(
        {"cortisol": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41, 0.12, 0.22, 0.32, 0.42]},
        index=index,
    )


def saliva_data_samples_days():
    index = pd.MultiIndex.from_product(
        [["Vp01", "Vp02"], ["T1", "T2"], ["S0", "S1", "S2"]],
        names=["subject", "day", "sample"],
    )
    return pd.DataFrame(
        {
            "cortisol": [
                0.1,
                0.2,
                0.3,
                0.15,
                0.25,
                0.35,
                0.11,
                0.21,
                0.31,
                0.16,
                0.26,
                0.36,
            ]
        },
        index=index,
    )


def saliva_data_samples_time():
    index = pd.MultiIndex.from_product(
        [["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["subject", "sample"]
    )
    return pd.DataFrame(
        {"cortisol": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41, 0.12, 0.22, 0.32, 0.42], "time": [0, 10, 20, 30] * 3},
        index=index,
    )


def saliva_data_samples_condition():
    index = pd.MultiIndex.from_tuples(
        [
            ("Control", "Vp01", "S0"),
            ("Control", "Vp01", "S1"),
            ("Control", "Vp01", "S2"),
            ("Control", "Vp01", "S3"),
            ("Intervention", "Vp02", "S0"),
            ("Intervention", "Vp02", "S1"),
            ("Intervention", "Vp02", "S2"),
            ("Intervention", "Vp02", "S3"),
            ("Control", "Vp03", "S0"),
            ("Control", "Vp03", "S1"),
            ("Control", "Vp03", "S2"),
            ("Control", "Vp03", "S3"),
        ],
        names=["condition", "subject", "sample"],
    )
    return pd.DataFrame(
        {"cortisol": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41, 0.12, 0.22, 0.32, 0.42]},
        index=index,
    )


def saliva_data_samples_time_condition():
    index = pd.MultiIndex.from_tuples(
        [
            ("Control", "Vp01", "S0"),
            ("Control", "Vp01", "S1"),
            ("Control", "Vp01", "S2"),
            ("Control", "Vp01", "S3"),
            ("Intervention", "Vp02", "S0"),
            ("Intervention", "Vp02", "S1"),
            ("Intervention", "Vp02", "S2"),
            ("Intervention", "Vp02", "S3"),
            ("Control", "Vp03", "S0"),
            ("Control", "Vp03", "S1"),
            ("Control", "Vp03", "S2"),
            ("Control", "Vp03", "S3"),
        ],
        names=["condition", "subject", "sample"],
    )
    return pd.DataFrame(
        {"cortisol": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41, 0.12, 0.22, 0.32, 0.42], "time": [0, 10, 20, 30] * 3},
        index=index,
    )


class TestIoSaliva:
    @pytest.mark.parametrize(
        "file_path, saliva_type, sample_id_col, data_col, id_col_names, regex_str, "
        "sample_times, condition_list, expected",
        [
            ("cortisol_plate_samples.xlsx", "cortisol", None, None, None, None, None, None, does_not_raise()),
            ("cortisol_plate_samples.xlsx", "cortisol", "sample ID", None, None, None, None, None, does_not_raise()),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample ID",
                "cortisol (nmol/l)",
                None,
                None,
                None,
                None,
                does_not_raise(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample ID",
                "cortisol (nmol/l)",
                ["subject", "sample"],
                r"(Vp\d+) (S\d)",
                None,
                None,
                does_not_raise(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample ID",
                "cortisol (nmol/l)",
                None,
                r"(Vp\d+) (S\d)",
                None,
                None,
                does_not_raise(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample ID",
                "cortisol (nmol/l)",
                None,
                r"(Vp\d+) (S\d)",
                [0, 10, 20, 30],
                None,
                does_not_raise(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample ID",
                "cortisol (nmol/l)",
                None,
                r"(Vp\d+) (S\d)",
                None,
                ["Control", "Condition", "Control"],
                does_not_raise(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample ID",
                "cortisol (nmol/l)",
                None,
                r"(Vp\d+) (S\d)",
                None,
                ["Control", "Condition", "Control", "Condition"],
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples.csv",
                "cortisol",
                None,
                None,
                None,
                None,
                None,
                None,
                pytest.raises(FileExtensionError),
            ),
            ("cortisol_plate_samples.xlsx", "amylase", None, None, None, None, None, None, pytest.raises(ValueError)),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                "sample",
                None,
                None,
                None,
                None,
                None,
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                None,
                "data",
                None,
                None,
                None,
                None,
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                None,
                None,
                None,
                "test",
                None,
                None,
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                None,
                None,
                None,
                r"Vp\d{2} S(\d)",
                None,
                None,
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                None,
                None,
                None,
                None,
                [0, 10, 20, 30, 40],
                None,
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples.xlsx",
                "cortisol",
                None,
                None,
                ["subject", "day", "sample"],
                None,
                None,
                None,
                pytest.raises(ValueError),
            ),
            (
                "cortisol_plate_samples_wrong_number.xlsx",
                "cortisol",
                None,
                None,
                None,
                None,
                None,
                None,
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_load_saliva_plate_raises(
        self,
        file_path,
        saliva_type,
        sample_id_col,
        data_col,
        id_col_names,
        regex_str,
        sample_times,
        condition_list,
        expected,
    ):
        with expected:
            load_saliva_plate(
                file_path=TEST_FILE_PATH.joinpath(file_path),
                saliva_type=saliva_type,
                sample_id_col=sample_id_col,
                data_col=data_col,
                id_col_names=id_col_names,
                regex_str=regex_str,
                sample_times=sample_times,
                condition_list=condition_list,
            )

    @pytest.mark.parametrize(
        "file_path, id_col_names, regex_str, expected",
        [
            (
                "cortisol_plate_samples.xlsx",
                ["subject", "sample"],
                r"(Vp\d+) (S\d)",
                pd.MultiIndex.from_product(
                    [["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["subject", "sample"]
                ),
            ),
            (
                "cortisol_plate_samples.xlsx",
                ["subject", "sample"],
                r"(Vp\w+) (S\w)",
                pd.MultiIndex.from_product(
                    [["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["subject", "sample"]
                ),
            ),
            (
                "cortisol_plate_samples.xlsx",
                ["subject", "sample"],
                r"(Vp\d{2}) (S\d)",
                pd.MultiIndex.from_product(
                    [["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["subject", "sample"]
                ),
            ),
            (
                "cortisol_plate_samples.xlsx",
                ["subject", "sample"],
                r"Vp(\w+) S(\d)",
                pd.MultiIndex.from_product([["01", "02", "03"], ["0", "1", "2", "3"]], names=["subject", "sample"]),
            ),
            (
                "cortisol_plate_samples_days.xlsx",
                ["subject", "day", "sample"],
                r"(Vp\d+) (T\d) (S\d)",
                pd.MultiIndex.from_product(
                    [["Vp01", "Vp02"], ["T1", "T2"], ["S0", "S1", "S2"]],
                    names=["subject", "day", "sample"],
                ),
            ),
            (
                "cortisol_plate_samples_days.xlsx",
                ["subject", "day", "sample"],
                r"Vp(\d+) (T\d) S(\d)",
                pd.MultiIndex.from_product(
                    [["01", "02"], ["T1", "T2"], ["0", "1", "2"]],
                    names=["subject", "day", "sample"],
                ),
            ),
        ],
    )
    def test_load_saliva_plate_regex(self, file_path, id_col_names, regex_str, expected):
        data_out = load_saliva_plate(
            TEST_FILE_PATH.joinpath(file_path), saliva_type="cortisol", id_col_names=id_col_names, regex_str=regex_str
        )
        assert_index_equal(data_out.index, expected)

    @pytest.mark.parametrize(
        "file_path, id_col_names, sample_times, condition_list, regex_str, expected",
        [
            ("cortisol_plate_samples.xlsx", None, None, None, r"(Vp\d+) (S\d)", saliva_data_samples()),
            ("cortisol_plate_samples.xlsx", None, [0, 10, 20, 30], None, r"(Vp\d+) (S\d)", saliva_data_samples_time()),
            (
                "cortisol_plate_samples.xlsx",
                None,
                [0, 10, 20, 30],
                ["Control", "Intervention", "Control"],
                r"(Vp\d+) (S\d)",
                saliva_data_samples_time_condition(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                None,
                [0, 10, 20, 30],
                {"Control": ["Vp01", "Vp03"], "Intervention": ["Vp02"]},
                r"(Vp\d+) (S\d)",
                saliva_data_samples_time_condition(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                None,
                [0, 10, 20, 30],
                pd.DataFrame(
                    {"subject": ["Vp01", "Vp02", "Vp03"], "condition": ["Control", "Intervention", "Control"]}
                ).set_index("subject"),
                r"(Vp\d+) (S\d)",
                saliva_data_samples_time_condition(),
            ),
            (
                "cortisol_plate_samples.xlsx",
                None,
                None,
                ["Control", "Intervention", "Control"],
                r"(Vp\d+) (S\d)",
                saliva_data_samples_condition(),
            ),
            ("cortisol_plate_samples_days.xlsx", None, None, None, r"(Vp\d+) (T\d) (S\d)", saliva_data_samples_days()),
        ],
    )
    def test_load_saliva_plate(self, file_path, id_col_names, sample_times, condition_list, regex_str, expected):
        data_out = load_saliva_plate(
            TEST_FILE_PATH.joinpath(file_path),
            saliva_type="cortisol",
            id_col_names=id_col_names,
            regex_str=regex_str,
            sample_times=sample_times,
            condition_list=condition_list,
        )
        assert_frame_equal(data_out, expected)

    @pytest.mark.parametrize(
        "file_path, expected",
        [
            ("cortisol_plate_samples_empty.xlsx", pytest.raises(ValueError)),
        ],
    )
    def test_load_saliva_plate_empty_sample(self, file_path, expected):
        with expected:
            load_saliva_plate(
                TEST_FILE_PATH.joinpath(file_path),
                saliva_type="cortisol",
            )

    @pytest.mark.parametrize(
        "input_data, saliva_type, file_path, expected",
        [
            (saliva_data_samples(), "cortisol", "test_saliva.csv", does_not_raise()),
            (saliva_data_samples(), "cortisol", "test_saliva.xlsx", does_not_raise()),
            (saliva_data_samples(), "cortisol", "test_saliva.xls", does_not_raise()),
            (saliva_data_samples_time(), "cortisol", "test_saliva_time.csv", does_not_raise()),
            (saliva_data_samples_days(), "cortisol", "test_saliva_days.csv", does_not_raise()),
            (saliva_data_samples_condition(), "cortisol", "test_saliva_condition.csv", does_not_raise()),
            (saliva_data_samples_time_condition(), "cortisol", "test_saliva_condition.csv", does_not_raise()),
            (saliva_data_samples(), "amylase", "test_saliva.csv", pytest.raises(ValidationError)),
            (saliva_data_samples(), "cortisol", "test_saliva.txt", pytest.raises(FileExtensionError)),
            (saliva_data_invalid_format(), "cortisol", "test_saliva.csv", pytest.raises(ValidationError)),
        ],
    )
    def test_save_saliva_raises(self, input_data, saliva_type, file_path, expected, tmp_path):
        with expected:
            save_saliva(tmp_path.joinpath(file_path), input_data, saliva_type=saliva_type)

    @pytest.mark.parametrize(
        "file_path, subject_col, condition_col, additional_index_cols, sample_times, expected",
        [
            ("cortisol_wide_samples.xlsx", None, None, None, None, does_not_raise()),
            ("cortisol_wide_samples.csv", None, None, None, None, does_not_raise()),
            ("cortisol_wide_samples.csv", None, None, [], None, does_not_raise()),
            ("cortisol_wide_samples.csv", "subject", None, None, None, does_not_raise()),
            ("cortisol_wide_samples.csv", "subject", None, None, [0, 10, 20, 30], does_not_raise()),
            ("cortisol_wide_samples.csv", "ID", None, None, None, pytest.raises(ValidationError)),
            ("cortisol_wide_samples.csv", None, None, "day", None, pytest.raises(ValidationError)),
            ("cortisol_wide_samples_other_names.csv", None, None, None, None, pytest.raises(ValidationError)),
            ("cortisol_wide_samples_other_names.csv", "ID", None, None, None, does_not_raise()),
            ("cortisol_wide_samples.csv", None, "condition", None, None, pytest.raises(ValidationError)),
            ("cortisol_wide_samples.csv", None, None, None, [0, 10, 20, 30, 40], pytest.raises(ValueError)),
            ("cortisol_wide_samples.csv", None, None, None, [10, 0, 20, 30, 40], pytest.raises(ValueError)),
            ("cortisol_wide_samples_condition.csv", None, None, None, None, does_not_raise()),
            ("cortisol_wide_samples_condition.csv", None, "condition", None, None, does_not_raise()),
            ("cortisol_wide_samples_condition.csv", None, "Group", None, None, pytest.raises(ValidationError)),
            (
                "cortisol_wide_samples_condition_other_names.csv",
                None,
                None,
                None,
                None,
                does_not_raise(),
            ),  # works but result is not in the correct format
            (
                "cortisol_wide_samples_condition_other_names.csv",
                None,
                "condition",
                None,
                None,
                pytest.raises(ValidationError),
            ),
            (
                "cortisol_wide_samples_condition_other_names.csv",
                None,
                "Group",
                None,
                None,
                does_not_raise(),
            ),
            (
                "cortisol_wide_samples_days.csv",
                None,
                None,
                None,
                None,
                does_not_raise(),
            ),  # works but result is not in the correct format
            ("cortisol_wide_samples_days.csv", None, None, "day", None, does_not_raise()),
            ("cortisol_wide_samples_days.csv", None, None, ["day"], None, does_not_raise()),
        ],
    )
    def test_load_saliva_wide_format_raises(
        self, file_path, subject_col, condition_col, additional_index_cols, sample_times, expected
    ):
        with expected:
            load_saliva_wide_format(
                TEST_FILE_PATH.joinpath(file_path),
                saliva_type="cortisol",
                subject_col=subject_col,
                condition_col=condition_col,
                sample_times=sample_times,
                additional_index_cols=additional_index_cols,
            )

    @pytest.mark.parametrize(
        "file_path, subject_col, condition_col, sample_times, additional_index_cols, expected",
        [
            ("cortisol_wide_samples.xlsx", None, None, None, None, saliva_data_samples()),
            ("cortisol_wide_samples.csv", None, None, None, None, saliva_data_samples()),
            ("cortisol_wide_samples.csv", "subject", None, None, None, saliva_data_samples()),
            ("cortisol_wide_samples.csv", "subject", None, [0, 10, 20, 30], None, saliva_data_samples_time()),
            ("cortisol_wide_samples_other_names.csv", "ID", None, None, None, saliva_data_samples()),
            ("cortisol_wide_samples_condition.csv", None, None, None, None, saliva_data_samples_condition()),
            ("cortisol_wide_samples_condition.csv", None, "condition", None, None, saliva_data_samples_condition()),
            (
                "cortisol_wide_samples_condition_other_names.csv",
                None,
                "Group",
                None,
                None,
                saliva_data_samples_condition(),
            ),
            ("cortisol_wide_samples_days.csv", None, None, None, "day", saliva_data_samples_days()),
        ],
    )
    def test_load_saliva_wide_format(
        self, file_path, subject_col, condition_col, sample_times, additional_index_cols, expected
    ):

        data_out = load_saliva_wide_format(
            TEST_FILE_PATH.joinpath(file_path),
            saliva_type="cortisol",
            subject_col=subject_col,
            condition_col=condition_col,
            sample_times=sample_times,
            additional_index_cols=additional_index_cols,
        )
        assert_frame_equal(data_out, expected)
