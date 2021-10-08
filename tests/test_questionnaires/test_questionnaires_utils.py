from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Optional
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal, assert_series_equal

from biopsykit.questionnaires.utils import (
    bin_scale,
    compute_scores,
    convert_scale,
    crop_scale,
    find_cols,
    get_supported_questionnaires,
    invert,
    to_idx,
    wide_to_long,
    zero_pad_columns,
)
from biopsykit.utils.exceptions import ValidationError, ValueRangeError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/questionnaires")


@contextmanager
def does_not_raise():
    yield


def data_complete_correct() -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_correct.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def data_pre_post() -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_pre_post.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def data_results_compute_scores() -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_results_compute_scores.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def data_compute_scores() -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_compute_scores.csv"))
    data = data.set_index(["subject", "condition"])
    return data


class TestQuestionnairesUtils:
    @pytest.mark.parametrize(
        "data, expected",
        [(pd.Series(dtype="float64"), pytest.raises(ValidationError)), (pd.DataFrame(), does_not_raise())],
    )
    def test_find_cols_raise(self, data, expected):
        with expected:
            find_cols(data)

    @pytest.mark.parametrize(
        "data, regex_str, starts_with, ends_with, contains, zero_pad_numbers, expected",
        [
            (
                data_complete_correct(),
                None,
                "ADSL",
                None,
                None,
                False,
                ["ADSL_{}".format(i) for i in range(1, 21)],
            ),
            (
                data_complete_correct(),
                None,
                "ADSL",
                None,
                None,
                True,
                ["ADSL_{:02d}".format(i) for i in range(1, 21)],
            ),
            (
                data_complete_correct(),
                r"ADSL_(\d+)",
                None,
                None,
                None,
                True,
                ["ADSL_{:02d}".format(i) for i in range(1, 21)],
            ),
            (
                data_complete_correct(),
                None,
                "FEE",
                None,
                None,
                True,
                ["FEE_{}_{}".format(i, j) for i, j in product(range(1, 25), ["Mutter", "Vater"])],
            ),
            (
                data_complete_correct(),
                None,
                "FEE",
                "Vater",
                None,
                True,
                ["FEE_{}_Vater".format(i) for i in range(1, 25)],
            ),
            (
                data_complete_correct(),
                r"FEE_(\d+)_Mutter",
                None,
                None,
                None,
                True,
                ["FEE_{}_Mutter".format(i) for i in range(1, 25)],
            ),
            (
                data_complete_correct(),
                r"FEE_(\d+)_Mutter",
                None,
                "Vater",
                None,
                True,
                ["FEE_{}_Mutter".format(i) for i in range(1, 25)],
            ),
            (
                data_complete_correct(),
                None,
                "FEE",
                "Vater",
                "COPE",
                True,
                [],
            ),
            (
                data_complete_correct(),
                None,
                None,
                None,
                "COPE",
                True,
                ["Brief_COPE_{:02d}".format(i) for i in range(1, 29)],
            ),
            (
                data_complete_correct(),
                None,
                None,
                None,
                "COPE",
                False,
                ["Brief_COPE_{}".format(i) for i in range(1, 29)],
            ),
        ],
    )
    def test_find_cols(self, data, regex_str, starts_with, ends_with, contains, zero_pad_numbers, expected):
        data_out, cols = find_cols(
            data=data,
            regex_str=regex_str,
            starts_with=starts_with,
            ends_with=ends_with,
            contains=contains,
            zero_pad_numbers=zero_pad_numbers,
        )

        TestCase().assertListEqual(list(cols), expected)
        TestCase().assertListEqual(list(data_out.columns), expected)

    @pytest.mark.parametrize(
        "data, inplace, expected_in, expected_out",
        [
            (
                pd.DataFrame(columns=["ABC_1", "ABC_2", "ABC_3"]),
                False,
                pd.DataFrame(columns=["ABC_1", "ABC_2", "ABC_3"]),
                pd.DataFrame(columns=["ABC_01", "ABC_02", "ABC_03"]),
            ),
            (
                pd.DataFrame(columns=["ABC_1", "ABC_2", "ABC_3"]),
                True,
                pd.DataFrame(columns=["ABC_01", "ABC_02", "ABC_03"]),
                None,
            ),
        ],
    )
    def test_zero_pad_columns_inplace(self, data, inplace, expected_in, expected_out):
        out = zero_pad_columns(data=data, inplace=inplace)

        assert_frame_equal(data, expected_in)
        if expected_out is not None:
            assert_frame_equal(out, expected_out)

    @pytest.mark.parametrize(
        "col_idxs, expected",
        [
            ([1, 2, 3, 4], np.array([0, 1, 2, 3])),
            (np.array([1, 2, 3, 4]), np.array([0, 1, 2, 3])),
        ],
    )
    def test_to_idx(self, col_idxs, expected):
        out = to_idx(col_idxs=col_idxs)
        assert_array_equal(out, expected)

    @pytest.mark.parametrize(
        "data, score_range, cols, expected",
        [
            (np.array([[1, 2], [3, 4], [5, 6]]), [1, 0], None, pytest.raises(ValidationError)),
            (pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}), [1, 2, 3], None, pytest.raises(ValidationError)),
            (pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}), [1, 3], None, does_not_raise()),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], None, pytest.raises(ValueRangeError)),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], ["A"], pytest.raises(ValueRangeError)),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], ["A", "B"], pytest.raises(ValueRangeError)),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], ["B"], does_not_raise()),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], ["B", "C"], does_not_raise()),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], [0], pytest.raises(ValueRangeError)),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], [0, 1], pytest.raises(ValueRangeError)),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], [1], does_not_raise()),
            (pd.DataFrame({"A": [1, 4], "B": [2, 3], "C": [1, 3]}), [1, 3], [1, 2], does_not_raise()),
            (pd.Series([1, 2, 1, 2, 3]), [1, 3], [1, 2], does_not_raise()),
            (pd.Series([1, 2, 1, 2, 3]), [1, 3], None, does_not_raise()),
            (pd.Series([1, 2, 1, 4, 3]), [1, 3], None, pytest.raises(ValueRangeError)),
        ],
    )
    def test_invert_raises(self, data, score_range, cols, expected):
        with expected:
            invert(data=data, score_range=score_range, cols=cols)

    @pytest.mark.parametrize(
        "data, score_range, cols, inplace, expected_in, expected_out",
        [
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                None,
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [3, 2], "B": [2, 1], "C": [3, 1]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                None,
                True,
                pd.DataFrame({"A": [3, 2], "B": [2, 1], "C": [3, 1]}),
                None,
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 5]}),
                [0, 5],
                None,
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 5]}),
                pd.DataFrame({"A": [4, 3], "B": [3, 2], "C": [4, 0]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 5]}),
                [0, 5],
                None,
                True,
                pd.DataFrame({"A": [4, 3], "B": [3, 2], "C": [4, 0]}),
                None,
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                ["A", "B"],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [3, 2], "B": [2, 1], "C": [1, 3]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                [0, 1],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [3, 2], "B": [2, 1], "C": [1, 3]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                ["A"],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [3, 2], "B": [2, 3], "C": [1, 3]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                ["A"],
                True,
                pd.DataFrame({"A": [3, 2], "B": [2, 3], "C": [1, 3]}),
                None,
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                [1, 2],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [1, 2], "B": [2, 1], "C": [3, 1]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                [1, 3],
                [1, 2],
                True,
                pd.DataFrame({"A": [1, 2], "B": [2, 1], "C": [3, 1]}),
                None,
            ),
        ],
    )
    def test_invert(self, data, score_range, cols, inplace, expected_in, expected_out):
        out = invert(data=data, score_range=score_range, cols=cols, inplace=inplace)

        assert_frame_equal(data, expected_in)
        if expected_out is not None:
            assert_frame_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, score_range, cols, inplace, expected_in, expected_out",
        [
            (
                pd.Series([1, 2, 3, 2, 2, 1]),
                [1, 3],
                None,
                False,
                pd.Series([1, 2, 3, 2, 2, 1]),
                pd.Series([3, 2, 1, 2, 2, 3]),
            ),
            (
                pd.Series([1, 2, 3, 2, 2, 1]),
                [1, 3],
                None,
                True,
                pd.Series([3, 2, 1, 2, 2, 3]),
                None,
            ),
            (
                pd.Series([1, 2, 3, 2, 2, 1]),
                [1, 3],
                ["A"],
                False,
                pd.Series([1, 2, 3, 2, 2, 1]),
                pd.Series([3, 2, 1, 2, 2, 3]),
            ),
        ],
    )
    def test_invert_series(self, data, score_range, cols, inplace, expected_in, expected_out):
        out = invert(data=data, score_range=score_range, cols=cols, inplace=inplace)

        assert_series_equal(data, expected_in)
        if expected_out is not None:
            assert_series_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, offset, cols, inplace, expected_in, expected_out",
        [
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                -1,
                None,
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [0, 1], "B": [1, 2], "C": [0, 2]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                4,
                None,
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [5, 6], "B": [6, 7], "C": [5, 7]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                -1,
                None,
                True,
                pd.DataFrame({"A": [0, 1], "B": [1, 2], "C": [0, 2]}),
                None,
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                -1,
                ["A"],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [0, 1], "B": [2, 3], "C": [1, 3]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                -1,
                ["A"],
                True,
                pd.DataFrame({"A": [0, 1], "B": [2, 3], "C": [1, 3]}),
                None,
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                -1,
                [0],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [0, 1], "B": [2, 3], "C": [1, 3]}),
            ),
            (
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                -1,
                [1, 2],
                False,
                pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}),
                pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [0, 2]}),
            ),
        ],
    )
    def test_convert_scale(self, data, offset, cols, inplace, expected_in, expected_out):
        out = convert_scale(data=data, offset=offset, cols=cols, inplace=inplace)

        assert_frame_equal(data, expected_in)
        if expected_out is not None:
            assert_frame_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, offset, cols, inplace, expected_in, expected_out",
        [
            (
                pd.Series([1, 2, 3, 2, 1, 3]),
                -1,
                None,
                False,
                pd.Series([1, 2, 3, 2, 1, 3]),
                pd.Series([0, 1, 2, 1, 0, 2]),
            ),
            (
                pd.Series([1, 2, 3, 2, 1, 3]),
                -1,
                ["A"],
                False,
                pd.Series([1, 2, 3, 2, 1, 3]),
                pd.Series([0, 1, 2, 1, 0, 2]),
            ),
            (
                pd.Series([1, 2, 3, 2, 1, 3]),
                -1,
                None,
                True,
                pd.Series([0, 1, 2, 1, 0, 2]),
                None,
            ),
        ],
    )
    def test_convert_scale_series(self, data, offset, cols, inplace, expected_in, expected_out):
        out = convert_scale(data=data, offset=offset, cols=cols, inplace=inplace)

        assert_series_equal(data, expected_in)
        if expected_out is not None:
            assert_series_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, score_range, set_nan, expected",
        [
            (np.array([[1, 2], [3, 4], [5, 6]]), [1, 0], None, pytest.raises(ValidationError)),
            (pd.DataFrame({"A": [1, 2], "B": [2, 3], "C": [1, 3]}), [1, 2, 3], False, pytest.raises(ValidationError)),
            (pd.DataFrame({"A": [1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}), [1, 5], False, does_not_raise()),
        ],
    )
    def test_crop_scale_raises(self, data, score_range, set_nan, expected):
        with expected:
            crop_scale(data=data, score_range=score_range, set_nan=set_nan)

    @pytest.mark.parametrize(
        "data, score_range, set_nan, inplace, expected_in, expected_out",
        [
            (
                pd.DataFrame({"A": [-1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}),
                [1, 5],
                False,
                False,
                pd.DataFrame({"A": [-1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}),
                pd.DataFrame({"A": [1, 4, 5], "B": [2, 3, 5], "C": [1, 3, 5]}),
            ),
            (
                pd.DataFrame({"A": [-1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}),
                [1, 5],
                False,
                True,
                pd.DataFrame({"A": [1, 4, 5], "B": [2, 3, 5], "C": [1, 3, 5]}),
                None,
            ),
            (
                pd.DataFrame({"A": [-1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}),
                [1, 5],
                True,
                False,
                pd.DataFrame({"A": [-1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}),
                pd.DataFrame({"A": [np.nan, 4, np.nan], "B": [2, 3, np.nan], "C": [1, 3, np.nan]}),
            ),
            (
                pd.DataFrame({"A": [-1, 4, 8], "B": [2, 3, 7], "C": [1, 3, 6]}),
                [1, 5],
                True,
                True,
                pd.DataFrame({"A": [np.nan, 4, np.nan], "B": [2, 3, np.nan], "C": [1, 3, np.nan]}),
                None,
            ),
        ],
    )
    def test_crop_scale(self, data, score_range, set_nan, inplace, expected_in, expected_out):
        out = crop_scale(data=data, score_range=score_range, inplace=inplace, set_nan=set_nan)

        assert_frame_equal(data, expected_in)
        if expected_out is not None:
            assert_frame_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, score_range, set_nan, inplace, expected_in, expected_out",
        [
            (
                pd.Series([-1, 4, 8, 2, 3, 7, 1, 3, 6]),
                [1, 5],
                False,
                False,
                pd.Series([-1, 4, 8, 2, 3, 7, 1, 3, 6]),
                pd.Series([1, 4, 5, 2, 3, 5, 1, 3, 5]),
            ),
            (
                pd.Series([-1, 4, 8, 2, 3, 7, 1, 3, 6]),
                [1, 5],
                False,
                True,
                pd.Series([1, 4, 5, 2, 3, 5, 1, 3, 5]),
                None,
            ),
            (
                pd.Series([-1, 4, 8, 2, 3, 7, 1, 3, 6]),
                [1, 5],
                True,
                False,
                pd.Series([-1, 4, 8, 2, 3, 7, 1, 3, 6]),
                pd.Series([np.nan, 4, np.nan, 2, 3, np.nan, 1, 3, np.nan]),
            ),
            (
                pd.Series([-1, 4, 8, 2, 3, 7, 1, 3, 6]),
                [1, 5],
                True,
                True,
                pd.Series([np.nan, 4, np.nan, 2, 3, np.nan, 1, 3, np.nan]),
                None,
            ),
        ],
    )
    def test_crop_scale_series(self, data, score_range, set_nan, inplace, expected_in, expected_out):
        out = crop_scale(data=data, score_range=score_range, inplace=inplace, set_nan=set_nan)

        assert_series_equal(data, expected_in)
        if expected_out is not None:
            assert_series_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, bins, cols, first_min, last_max, inplace, expected_in, expected_out",
        [
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                False,
                False,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [np.nan, np.nan, 0, 7, 1, 0, 6, np.nan],
                        "B": [2, 5, np.nan, 4, 4, 6, 1, np.nan],
                        "C": [5, 1, np.nan, np.nan, 0, 1, 1, np.nan],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                False,
                True,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [np.nan, np.nan, 0, 7, 1, 0, 6, 8],
                        "B": [2, 5, np.nan, 4, 4, 6, 1, np.nan],
                        "C": [5, 1, 8, np.nan, 0, 1, 1, np.nan],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                True,
                False,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [0, 0, 1, 8, 2, 1, 7, np.nan],
                        "B": [3, 6, 0, 5, 5, 7, 2, 0],
                        "C": [6, 2, np.nan, 0, 1, 2, 2, 0],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                True,
                True,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [0, 0, 1, 8, 2, 1, 7, 9],
                        "B": [3, 6, 0, 5, 5, 7, 2, 0],
                        "C": [6, 2, 9, 0, 1, 2, 2, 0],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                [5, 14, 25, 45],
                None,
                False,
                False,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [np.nan, 0, 0, np.nan, 1, 1, np.nan, np.nan],
                        "B": [2, np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan],
                        "C": [np.nan, 1, np.nan, 0, 0, 2, 1, 0],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                [5, 14, 25, 45],
                None,
                True,
                False,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [0, 1, 1, np.nan, 2, 2, np.nan, np.nan],
                        "B": [3, np.nan, 0, np.nan, np.nan, np.nan, 2, 0],
                        "C": [np.nan, 1, np.nan, 0, 0, 2, 1, 0],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                [5, 14, 25, 45],
                None,
                True,
                True,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [0, 1, 1, 4, 2, 2, 4, 4],
                        "B": [3, 4, 0, 4, 4, 4, 2, 0],
                        "C": [3, 1, 3, 0, 0, 2, 1, 0],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                [5, 14, 25, 45],
                None,
                True,
                True,
                True,
                pd.DataFrame(
                    {
                        "A": [0, 1, 1, 4, 2, 2, 4, 4],
                        "B": [3, 4, 0, 4, 4, 4, 2, 0],
                        "C": [3, 1, 3, 0, 0, 2, 1, 0],
                    }
                ),
                None,
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 7],
                    }
                ),
                5,
                None,
                True,
                True,
                True,
                pd.DataFrame(
                    {
                        "A": [0, 0, 0, 4, 1, 0, 3, 4],
                        "B": [2, 4, 0, 3, 3, 4, 1, 0],
                        "C": [3, 1, 4, 0, 0, 1, 1, 0],
                    }
                ),
                None,
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                ["A"],
                False,
                True,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [np.nan, np.nan, 0, 7, 1, 0, 6, 8],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                ["B", "C"],
                False,
                True,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [2, 5, np.nan, 4, 4, 6, 1, np.nan],
                        "C": [5, 1, 8, np.nan, 0, 1, 1, np.nan],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                [1, 2],
                False,
                True,
                False,
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [2, 5, np.nan, 4, 4, 6, 1, np.nan],
                        "C": [5, 1, 8, np.nan, 0, 1, 1, np.nan],
                    }
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                ["A"],
                True,
                False,
                True,
                pd.DataFrame(
                    {
                        "A": [0, 0, 1, 8, 2, 1, 7, np.nan],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                None,
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                "A",
                True,
                False,
                True,
                pd.DataFrame(
                    {
                        "A": [0, 0, 1, 8, 2, 1, 7, np.nan],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                None,
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                [0],
                True,
                False,
                True,
                pd.DataFrame(
                    {
                        "A": [0, 0, 1, 8, 2, 1, 7, np.nan],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                None,
            ),
            (
                pd.DataFrame(
                    {
                        "A": [1, 10, 14, 90, 24, 16, 73, 97],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                0,
                True,
                False,
                True,
                pd.DataFrame(
                    {
                        "A": [0, 0, 1, 8, 2, 1, 7, np.nan],
                        "B": [34, 64, 2, 58, 54, 76, 23, 5],
                        "C": [65, 24, 95, 6, 12, 26, 24, 0],
                    }
                ),
                None,
            ),
        ],
    )
    def test_bin_scale(self, data, bins, cols, first_min, last_max, inplace, expected_in, expected_out):
        out = bin_scale(data=data, bins=bins, cols=cols, first_min=first_min, last_max=last_max, inplace=inplace)

        assert_frame_equal(data, expected_in)
        if expected_out is not None:
            assert_frame_equal(out, expected_out)

    @pytest.mark.parametrize(
        "data, bins, cols, first_min, last_max, inplace, expected_in, expected_out",
        [
            (
                pd.Series([1, 10, 14, 90, 24, 16, 73, 97]),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                False,
                False,
                False,
                pd.Series([1, 10, 14, 90, 24, 16, 73, 97]),
                pd.Series([np.nan, np.nan, 0, 7, 1, 0, 6, np.nan]),
            ),
            (
                pd.Series([34, 64, 2, 58, 54, 76, 23, 5]),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                False,
                False,
                False,
                pd.Series([34, 64, 2, 58, 54, 76, 23, 5]),
                pd.Series([2, 5, np.nan, 4, 4, 6, 1, np.nan]),
            ),
            (
                pd.Series([65, 24, 95, 6, 12, 26, 24, 0]),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                False,
                False,
                False,
                pd.Series([65, 24, 95, 6, 12, 26, 24, 0]),
                pd.Series([5, 1, np.nan, np.nan, 0, 1, 1, np.nan]),
            ),
            (
                pd.Series([1, 10, 14, 90, 24, 16, 73, 97]),
                [10, 20, 30, 40, 50, 60, 70, 80, 90],
                None,
                False,
                True,
                False,
                pd.Series([1, 10, 14, 90, 24, 16, 73, 97]),
                pd.Series([np.nan, np.nan, 0, 7, 1, 0, 6, 8]),
            ),
            (
                pd.Series([1, 10, 14, 90, 24, 16, 73, 97]),
                [5, 14, 25, 45],
                None,
                True,
                False,
                False,
                pd.Series([1, 10, 14, 90, 24, 16, 73, 97]),
                pd.Series([0, 1, 1, np.nan, 2, 2, np.nan, np.nan]),
            ),
        ],
    )
    def test_bin_scale_series(self, data, bins, cols, first_min, last_max, inplace, expected_in, expected_out):
        out = bin_scale(data=data, bins=bins, cols=cols, first_min=first_min, last_max=last_max, inplace=inplace)

        assert_series_equal(data, expected_in)
        if expected_out is not None:
            assert_series_equal(out, expected_out)

    def test_wide_to_long_warning(self):
        # just make sure that DeprecationWarning is issued, functionality will be tested in other functions
        with pytest.warns(DeprecationWarning):
            wide_to_long(
                pd.DataFrame({"A_Pre": [0, 1], "A_Post": [0, 1]}, index=pd.Index([0, 1], name="subject")),
                quest_name="A",
                levels="time",
            )

    def test_get_supported_questionnaires(self):
        quests = get_supported_questionnaires()
        assert all(isinstance(s, str) for s in quests.keys())
        assert all(isinstance(s, str) for s in quests.values())

    @pytest.mark.parametrize(
        "data, quest_dict, quest_kwargs, expected",
        [
            (
                data_complete_correct(),
                {"abc": ["ADSL_{}".format(i) for i in range(1, 21)]},
                None,
                pytest.raises(ValueError),
            ),
            (
                data_complete_correct(),
                {"ads_l": ["ADSL_{}".format(i) for i in range(1, 21)]},
                {"ads_l": {"subscales": []}},
                pytest.raises(TypeError),
            ),
            (data_complete_correct(), {"ads_l": ["ADSL_{}".format(i) for i in range(1, 21)]}, None, does_not_raise()),
            (
                data_complete_correct(),
                {"panas": ["PANAS_{}".format(i) for i in range(1, 21)]},
                {"panas": {"subscales": []}},
                pytest.raises(TypeError),
            ),
            (
                data_complete_correct(),
                {"panas": ["PANAS_{}".format(i) for i in range(1, 21)]},
                {"panas": {"language": "english"}},
                does_not_raise(),
            ),
            (
                data_complete_correct(),
                {"FEE": ["FEE_{}_{}".format(i, j) for i, j in product(range(1, 25), ["Vater", "Mutter"])]},
                {"FEE": {"language": "german"}},
                does_not_raise(),
            ),
            (
                data_complete_correct(),
                {"fee": ["FEE_{}_{}".format(i, j) for i, j in product(range(1, 25), ["Vater", "Mutter"])]},
                {"fee": {"language": "german"}},
                does_not_raise(),
            ),
            (
                data_complete_correct(),
                {"FEE": ["FEE_{}_{}".format(i, j) for i, j in product(range(1, 25), ["Vater", "Mutter"])]},
                {"fee": {"language": "german"}},
                pytest.raises(ValidationError),
            ),
            (
                data_complete_correct(),
                {"svf_120": ["SVF120_{}".format(i) for i in range(1, 121)]},
                {"svf_120": {"subscales": {"Bag": [10, 31, 50, 67, 88, 106]}}},
                does_not_raise(),
            ),
            (
                data_pre_post(),
                {
                    "panas-pre": ["PANAS_{}_Pre".format(i) for i in range(1, 21)],
                    "panas-post": ["PANAS_{}_Post".format(i) for i in range(1, 21)],
                },
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_get_compute_scores_raises(self, data, quest_dict, quest_kwargs, expected):
        with expected:
            compute_scores(data=data, quest_dict=quest_dict, quest_kwargs=quest_kwargs)

    @pytest.mark.parametrize(
        "data, quest_dict, quest_kwargs, expected",
        [
            (
                data_compute_scores(),
                {
                    "pss": ["PSS_{}".format(i) for i in range(1, 11)],
                    "fee": ["FEE_{}_{}".format(i, j) for i, j in product(range(1, 25), ["Vater", "Mutter"])],
                    "panas-pre": ["PANAS_{}_Pre".format(i) for i in range(1, 21)],
                    "panas-post": ["PANAS_{}_Post".format(i) for i in range(1, 21)],
                    "svf_120": ["SVF120_{}".format(i) for i in range(1, 121)],
                },
                {"fee": {"language": "german"}, "svf_120": {"subscales": {"Bag": [10, 31, 50, 67, 88, 106]}}},
                data_results_compute_scores(),
            )
        ],
    )
    def test_get_compute_scores(self, data, quest_dict, quest_kwargs, expected):
        out = compute_scores(data=data, quest_dict=quest_dict, quest_kwargs=quest_kwargs)
        assert_frame_equal(expected, out)
