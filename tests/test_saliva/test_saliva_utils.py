import datetime
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

import biopsykit.saliva.utils as utils
from biopsykit.utils.exceptions import ValidationError


@contextmanager
def does_not_raise():
    yield


def saliva_cols_none():
    return pd.DataFrame(columns=[])


def saliva_cols_cort():
    return pd.DataFrame(columns=["scorts0", "scorts1", "scorts2"])


def saliva_cols_cort_multi_days():
    return pd.DataFrame(columns=["scorts0d1", "scorts1d1", "scorts2d1", "scorts0d2", "scorts1d2", "scorts2d2"])


def saliva_cols_cort_wrong():
    return pd.DataFrame(columns=["scort", "scort", "scort"])


def saliva_cols_amy():
    return pd.DataFrame(columns=["samy1", "samy2", "samy3"])


def saliva_cols_all():
    columns = [
        "pasa1_t1",
        "pasa2_t1",
        "pasa3_t1",
        "pasa4_t1",
        "scort0",
        "scort1",
        "scort2",
        "scort3",
        "scort4",
        "lnscort0",
        "lnscort1",
        "lnscort2",
        "lnscort3",
        "lnscort4",
        "lgscort_inc1",
        "lgscort_inc2",
        "lgcort_hab",
        "COPE_1",
        "COPE_2",
        "COPE_3",
        "COPE_4",
        "PSS_4",
        "PSS_5",
        "PSS_6",
        "PSS_7",
        "PSS_8",
        "PSS_9",
        "PSS_10",
        "THQ_1",
        "THQ_1_1",
        "THQ_1_2",
        "THQ_1_3",
        "amys0",
        "amys1",
        "amys2",
        "amys3",
        "amys4",
        "amyls0",
        "amyls1",
        "amyls2",
        "amyls3",
        "amyls4",
        "amylinc1",
        "amylinc2",
        "hil6s0",
        "hil6s1",
        "hil6s2",
        "hil6s3",
        "x_amylinc",
        "amylhab",
        "erq_reapp_p",
        "erq_supp_p",
        "lgscort0",
        "lgscort1",
        "lgscort2",
        "lgscort3",
        "lgscort4",
        "cortmaxinc1",
        "cortmaxinc2",
        "corthab",
    ]
    return pd.DataFrame(columns=columns)


def sample_times_series_wrong_input():
    return pd.Series(
        np.concatenate(
            [
                [0, 15, 30],
                [0, 15, 30],
                [0, 15, 30],
                [0, 15, 30],
            ]
        ),
        index=pd.MultiIndex.from_product([range(0, 4), ["T1", "T2", "T3"]], names=["subject", "sample"]),
    )


def sample_times_series_seconds_missing():
    return pd.Series(
        np.concatenate(
            [
                ["09:00", "09:15", "09:30"],
                ["09:00", "09:15", "09:30"],
                ["09:00", "09:15", "09:30"],
                ["09:00", "09:15", "09:30"],
            ]
        ),
        index=pd.MultiIndex.from_product([range(0, 4), ["T1", "T2", "T3"]], names=["subject", "sample"]),
    )


def sample_times_series_str_correct():
    return pd.Series(
        np.concatenate(
            [
                ["09:00:00", "09:15:00", "09:30:00"],
                ["09:00:00", "09:15:00", "09:30:00"],
                ["09:00:00", "09:15:00", "09:30:00"],
                ["09:00:00", "09:15:00", "09:30:00"],
            ]
        ),
        index=pd.MultiIndex.from_product([range(0, 4), ["T1", "T2", "T3"]], names=["subject", "sample"]),
    )


def sample_times_series_timedelta_correct():
    return pd.Series(
        np.concatenate(
            [
                [pd.Timedelta(minutes=0), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30)],
                [pd.Timedelta(minutes=0), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30)],
                [pd.Timedelta(minutes=0), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30)],
                [pd.Timedelta(minutes=0), pd.Timedelta(minutes=15), pd.Timedelta(minutes=30)],
            ]
        ),
        index=pd.MultiIndex.from_product([range(0, 4), ["T1", "T2", "T3"]], names=["subject", "sample"]),
    )


def sample_times_series_datetime_correct():
    return pd.Series(
        np.concatenate(
            [
                [
                    datetime.datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 16:45", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 17:00", "%d/%m/%y %H:%M"),
                ],
                [
                    datetime.datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 16:45", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 17:00", "%d/%m/%y %H:%M"),
                ],
                [
                    datetime.datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 16:45", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 17:00", "%d/%m/%y %H:%M"),
                ],
                [
                    datetime.datetime.strptime("21/11/06 16:30", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 16:45", "%d/%m/%y %H:%M"),
                    datetime.datetime.strptime("21/11/06 17:00", "%d/%m/%y %H:%M"),
                ],
            ]
        ),
        index=pd.MultiIndex.from_product([range(0, 4), ["T1", "T2", "T3"]], names=["subject", "sample"]),
    )


def sample_times_series_correct_output():
    return pd.Series(
        [0.0, 15.0, 30.0] * 4,
        index=pd.MultiIndex.from_product([range(0, 4), ["T1", "T2", "T3"]], names=["subject", "sample"]),
    )


class TestSalivaUtils:
    @pytest.mark.parametrize(
        "data_in, saliva_type, expected",
        [
            (None, "cortisol", pytest.raises(ValidationError)),
            (pd.Series(dtype="float64"), "cortisol", pytest.raises(ValidationError)),
            (None, "adrenaline", pytest.raises(ValidationError)),
            (pd.DataFrame(), "adrenaline", pytest.raises(ValueError)),
            (pd.DataFrame(), "cortisol", does_not_raise()),
        ],
    )
    def test_get_saliva_column_suggestions_raise(self, data_in, saliva_type, expected):
        with expected:
            utils.get_saliva_column_suggestions(data_in, saliva_type)

    @pytest.mark.parametrize(
        "data_in, saliva_type, expected",
        [
            (saliva_cols_none(), "cortisol", []),
            (saliva_cols_cort(), "cortisol", [r"^scorts(\d)$"]),
            (saliva_cols_cort(), "amylase", []),
            (saliva_cols_cort_wrong(), "cortisol", []),
            (saliva_cols_cort_multi_days(), "cortisol", [r"^scorts(\d)d(\d)$"]),
            (saliva_cols_amy(), "cortisol", []),
            (saliva_cols_amy(), "amylase", [r"^samy(\d)$"]),
            (saliva_cols_all(), "cortisol", [r"^scort(\d)$"]),
            (saliva_cols_all(), "amylase", [r"^amyls(\d)$", r"^amys(\d)$"]),
            (saliva_cols_all(), "il6", [r"^hil6s(\d)$"]),
            (saliva_cols_all(), ["cortisol"], {"cortisol": [r"^scort(\d)$"]}),
            (
                saliva_cols_all(),
                ["cortisol", "amylase"],
                {"cortisol": [r"^scort(\d)$"], "amylase": [r"^amyls(\d)$", r"^amys(\d)$"]},
            ),
        ],
    )
    def test_get_saliva_column_suggestions(self, data_in, saliva_type, expected):
        assert expected == utils.get_saliva_column_suggestions(data_in, saliva_type)

    @pytest.mark.parametrize(
        "data_in, saliva_type, col_pattern, expected",
        [
            (
                saliva_cols_all(),
                "cortisol",
                None,
                does_not_raise(),
            ),
            (
                saliva_cols_all(),
                "cortisol",
                r"scort(\d)",
                does_not_raise(),
            ),
            (
                saliva_cols_all(),
                "amylase",
                None,
                pytest.raises(ValueError),
            ),
            (
                saliva_cols_all(),
                "amylase",
                r"amys(\d)",
                does_not_raise(),
            ),
            (
                saliva_cols_all(),
                "il6",
                r"hil6(\d)",
                does_not_raise(),
            ),
        ],
    )
    def test_extract_saliva_columns_raise(self, data_in, saliva_type, col_pattern, expected):
        with expected:
            utils.extract_saliva_columns(data=data_in, saliva_type=saliva_type, col_pattern=col_pattern)

    @pytest.mark.parametrize(
        "data_in, saliva_type, col_pattern, expected",
        [
            (
                saliva_cols_all(),
                "cortisol",
                None,
                pd.DataFrame(columns=["scort0", "scort1", "scort2", "scort3", "scort4"]),
            ),
            (
                saliva_cols_all(),
                "il6",
                None,
                pd.DataFrame(
                    columns=[
                        "hil6s0",
                        "hil6s1",
                        "hil6s2",
                        "hil6s3",
                    ]
                ),
            ),
            (
                saliva_cols_all(),
                "amylase",
                r"amyls(\d)",
                pd.DataFrame(
                    columns=[
                        "amyls0",
                        "amyls1",
                        "amyls2",
                        "amyls3",
                        "amyls4",
                    ]
                ),
            ),
        ],
    )
    def test_extract_saliva_columns(self, data_in, saliva_type, col_pattern, expected):
        assert_frame_equal(
            utils.extract_saliva_columns(data=data_in, saliva_type=saliva_type, col_pattern=col_pattern), expected
        )

    @pytest.mark.parametrize(
        "saliva_type, col_pattern, expected",
        [
            (["cortisol", "amylase"], [], pytest.raises(ValueError)),
            ([], [r"scort(\d)"], pytest.raises(ValueError)),
            (["cortisol", "amylase"], [r"scort(\d)", r"amys(\d)", r"hil6(\d)"], pytest.raises(ValueError)),
            (["cortisol", "amylase"], [r"scort(\d)", r"amys(\d)"], does_not_raise()),
            (["cortisol", "amylase"], None, pytest.raises(ValueError)),
            (["cortisol", "il6"], None, does_not_raise()),
        ],
    )
    def test_extract_saliva_columns_multi_saliva_types_raise(self, saliva_type, col_pattern, expected):
        with expected:
            utils.extract_saliva_columns(data=saliva_cols_all(), saliva_type=saliva_type, col_pattern=col_pattern)

    def test_extract_saliva_columns_multi_saliva_types(self):
        saliva_type = ["cortisol", "amylase"]
        col_pattern = [r"scort(\d)", r"amys(\d)"]
        data_out = utils.extract_saliva_columns(
            data=saliva_cols_all(), saliva_type=saliva_type, col_pattern=col_pattern
        )

        for saliva, col in zip(saliva_type, col_pattern):
            assert_frame_equal(
                data_out[saliva],
                utils.extract_saliva_columns(data=saliva_cols_all(), saliva_type=saliva, col_pattern=col),
            )

    @pytest.mark.parametrize(
        "data_in, expected",
        [
            (
                sample_times_series_seconds_missing(),
                pytest.raises(ValueError),
            ),
            (
                sample_times_series_wrong_input(),
                pytest.raises(ValueError),
            ),
            (
                sample_times_series_str_correct(),
                does_not_raise(),
            ),
        ],
    )
    def test_sample_times_series_raises(self, data_in, expected):
        with expected:
            utils.sample_times_datetime_to_minute(data_in)

    @pytest.mark.parametrize(
        "data_in, expected",
        [
            (
                sample_times_series_str_correct(),
                sample_times_series_correct_output(),
            ),
            (
                sample_times_series_timedelta_correct(),
                sample_times_series_correct_output(),
            ),
            (
                sample_times_series_datetime_correct(),
                sample_times_series_correct_output(),
            ),
        ],
    )
    def test_sample_times_series(self, data_in, expected):
        assert_series_equal(utils.sample_times_datetime_to_minute(data_in), expected)

    @pytest.mark.parametrize(
        "data_in, expected",
        [
            (
                sample_times_series_str_correct().unstack("sample"),
                sample_times_series_correct_output().unstack("sample"),
            ),
            (
                sample_times_series_timedelta_correct().unstack("sample"),
                sample_times_series_correct_output().unstack("sample"),
            ),
            (
                sample_times_series_datetime_correct().unstack("sample"),
                sample_times_series_correct_output().unstack("sample"),
            ),
        ],
    )
    def test_sample_times_dataframe(self, data_in, expected):
        assert_frame_equal(utils.sample_times_datetime_to_minute(data_in), expected)
