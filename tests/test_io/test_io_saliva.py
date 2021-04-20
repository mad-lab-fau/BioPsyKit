from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from biopsykit.io.saliva import load_saliva_plate
from biopsykit.utils.exceptions import FileExtensionError, ValidationError
from pandas._testing import assert_index_equal, assert_frame_equal

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


def saliva_data_samples():
    index = pd.MultiIndex.from_product(
        [["Vp01", "Vp02", "Vp03"], ["S0", "S1", "S2", "S3"]], names=["subject", "sample"]
    )
    return pd.DataFrame({"cortisol": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41, 0.12, 0.22, 0.32, 0.42]}, index=index)


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
        "file_path, id_col_names, regex_str, expected",
        [
            ("cortisol_plate_samples.xlsx", None, r"(Vp\d+) (S\d)", saliva_data_samples()),
            ("cortisol_plate_samples_days.xlsx", None, r"(Vp\d+) (T\d) (S\d)", saliva_data_samples_days()),
        ],
    )
    def test_load_saliva_plate(self, file_path, id_col_names, regex_str, expected):
        data_out = load_saliva_plate(
            TEST_FILE_PATH.joinpath(file_path), saliva_type="cortisol", id_col_names=id_col_names, regex_str=regex_str
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
