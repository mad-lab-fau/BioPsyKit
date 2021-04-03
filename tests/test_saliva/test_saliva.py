from contextlib import contextmanager
from typing import Optional

import pytest

import pandas as pd
import numpy as np
import scipy.stats as ss
from pandas.testing import assert_frame_equal, assert_series_equal
import biopsykit.saliva as saliva


@contextmanager
def does_not_raise():
    yield


def saliva_none():
    return None


def saliva_multiindex_wrong_name():
    return pd.DataFrame(
        index=pd.MultiIndex.from_product([range(0, 5), range(0, 5)], names=["subject", "id"]),
        columns=["cortisol", "time"],
    )


def saliva_no_multiindex():
    return pd.DataFrame(index=range(0, 5), columns=["cortisol", "time"])


def saliva_no_time():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=["cortisol"],
    )
    data["cortisol"] = np.concatenate(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [10, 2, 4, 4, 6],
            [-6, 2, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [2, np.nan, 8, 4, 6],
        ]
    )
    return data


def saliva_wrong_time_01():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=["cortisol", "time"],
    )
    data["cortisol"] = np.concatenate(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [10, 2, 4, 4, 6],
            [-6, 2, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [2, np.nan, 8, 4, 6],
        ]
    )
    # value '20' occurs twice
    data["time"] = [-10, 10, 20, 20, 40] * 8
    return data


def saliva_wrong_time_02():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=["cortisol", "time"],
    )
    data["cortisol"] = np.concatenate(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [10, 2, 4, 4, 6],
            [-6, 2, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [2, np.nan, 8, 4, 6],
        ]
    )
    # values are not increasing
    data["time"] = [60, 50, 40, 30, 20] * 8
    return data


def saliva_time(biomarker_type: Optional[str] = "cortisol"):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=[biomarker_type, "time"],
    )
    data[biomarker_type] = np.concatenate(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [10, 2, 4, 4, 6],
            [-6, 2, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [2, np.nan, 8, 4, 6],
        ]
    )
    data["time"] = [-10, 0, 10, 20, 30] * 8
    return data


def saliva_time_multiple_pre():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=["cortisol", "time"],
    )
    data["cortisol"] = np.concatenate(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [10, 2, 4, 4, 6],
            [-6, 2, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [2, np.nan, 8, 4, 6],
        ]
    )
    data["time"] = [-20, -10, 0, 10, 20] * 8
    return data


def saliva_pruessner_2003():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 3), range(0, 5)], names=["subject", "sample"]),
        columns=["cortisol", "time"],
    )
    data["cortisol"] = np.concatenate([[3.5, 7, 14, 7, 10], [3.5, 7, 14, 7, 10]])
    data["time"] = np.concatenate([[1, 2, 3, 4, 5], [0, 10, 15, 30, 45]])
    return data


def saliva_multi_days(biomarker_type: Optional[str] = "cortisol"):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 5), range(0, 2), range(0, 5)], names=["subject", "day", "sample"]),
        columns=["cortisol", "time"],
    )
    data["cortisol"] = np.concatenate(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4],
            [5, 4, 3, 2, 1],
            [10, 2, 4, 4, 6],
            [-6, 2, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [2, np.nan, 8, 4, 6],
        ]
    )
    data["time"] = [-10, 0, 10, 20, 30] * 8
    return data


params_max_increase = [
    (
        True,
        pd.DataFrame(
            [0, 0, 3, -1, 4, 4, 12, np.nan],
            columns=pd.Index(["cortisol_max_inc"], name="biomarker"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        pd.DataFrame(
            [0, 0, 4, -1, -4, 12, -6, 6],
            columns=pd.Index(["cortisol_max_inc"], name="biomarker"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_max_increase_percent = [
    (
        True,
        pd.DataFrame(
            [np.nan, 0, 300.0, -25.0, 200.0, 200.0, 200.0, np.nan],
            columns=pd.Index(["cortisol_max_inc_percent"], name="biomarker"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        pd.DataFrame(
            [np.nan, 0.0, np.inf, -20.0, -40.0, 200.0, -50.0, 300.0],
            columns=pd.Index(["cortisol_max_inc_percent"], name="biomarker"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_auc = [
    (
        False,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 40, 80, 120, 180, 100, 110, np.nan],
                "cortisol_auc_i": [0, 0, 80, -80, -220, 340, -370, np.nan],
                "cortisol_auc_i_post": [0, 0, 20, -20, 10, 10, 10, -50],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="biomarker",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        True,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 30, 75, 75, 120, 120, 80, np.nan],
                "cortisol_auc_i": [0, 0, 45, -45, 60, 60, 260, np.nan],
                "cortisol_auc_i_post": [0, 0, 20, -20, 10, 10, 10, -50],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="biomarker",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_auc_mutiple_pre = [
    (
        False,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 40, 80, 120, 180, 100, 110, np.nan],
                "cortisol_auc_i": [0, 0, 80, -80, -220, 340, -370, np.nan],
                "cortisol_auc_i_post": [0, 0, 5, -5, 10, 10, 10, 10],
            },
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="biomarker",
            ),
        ),
    ),
    (
        True,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 30, 75, 75, 120, 120, 80, np.nan],
                "cortisol_auc_i": [0, 0, 45, -45, 60, 60, 260, np.nan],
                "cortisol_auc_i_post": [0, 0, 5, -5, 10, 10, 10, 10],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="biomarker",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_slope = [
    (
        (0, 1),
        pd.DataFrame(
            {"cortisol_slope01": [0.0, 0.0, 0.1, -0.1, -0.8, 0.8, -1.8, np.nan]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slope01"], name="biomarker"),
        ),
    ),
    (
        (1, 2),
        pd.DataFrame(
            {"cortisol_slope12": [0.0, 0.0, 0.1, -0.1, 0.2, 0.2, 1.0, np.nan]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slope12"], name="biomarker"),
        ),
    ),
    (
        (0, 4),
        pd.DataFrame(
            {"cortisol_slope04": [0.0, 0.0, 0.1, -0.1, -0.1, 0.3, -0.15, 0.1]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slope04"], name="biomarker"),
        ),
    ),
    (
        (0, -1),
        pd.DataFrame(
            {"cortisol_slope04": [0.0, 0.0, 0.1, -0.1, -0.1, 0.3, -0.15, 0.1]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slope04"], name="biomarker"),
        ),
    ),
]


class TestSaliva:
    @pytest.mark.parametrize(
        "input_data",
        [saliva_multiindex_wrong_name(), saliva_no_multiindex(), saliva_none()],
    )
    def test_check_data_format_invalid(self, input_data):
        with pytest.raises(ValueError):
            saliva.max_increase(input_data)

    @pytest.mark.parametrize(
        "biomarker_type, expectation",
        [
            ("cortisol", does_not_raise()),
            ("amylase", pytest.raises(ValueError)),
            ("il6", pytest.raises(ValueError)),
            (None, pytest.raises(ValueError)),
        ],
    )
    def test_max_increase_raises_biomarker_type(self, biomarker_type, expectation):
        data = saliva_time()
        with expectation:
            saliva.max_increase(data, biomarker_type=biomarker_type)

    @pytest.mark.parametrize(
        "biomarker_type, expected_columns",
        [
            ("cortisol", ["cortisol_max_inc"]),
            ("amylase", ["amylase_max_inc"]),
            ("il6", ["il6_max_inc"]),
        ],
    )
    def test_max_increase_columns(self, biomarker_type, expected_columns):
        data_in = saliva_time(biomarker_type)
        data_out = saliva.max_increase(data_in, biomarker_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize("remove_s0, expected", params_max_increase)
    def test_max_increase(self, remove_s0, expected):
        out = saliva.max_increase(saliva_no_time(), remove_s0=remove_s0)
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize("remove_s0, expected", params_max_increase_percent)
    def test_max_increase_percent(self, remove_s0, expected):
        out = saliva.max_increase(saliva_no_time(), remove_s0=remove_s0, percent=True)
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize("remove_s0, expected", params_max_increase)
    def test_max_increase_multi_days(self, remove_s0, expected):
        out = saliva.max_increase(saliva_multi_days(), remove_s0=remove_s0)
        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "biomarker_type, expected_columns",
        [
            ("cortisol", ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"]),
            ("amylase", ["amylase_auc_g", "amylase_auc_i", "amylase_auc_i_post"]),
            ("il6", ["il6_auc_g", "il6_auc_i", "il6_auc_i_post"]),
        ],
    )
    def test_auc_columns(self, biomarker_type, expected_columns):
        data_in = saliva_time(biomarker_type)
        data_out = saliva.auc(data_in, biomarker_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize(
        "data",
        [saliva_no_time(), saliva_wrong_time_01(), saliva_wrong_time_02()],
        ids=["notime", "sametime", "descreasingtime"],
    )
    def test_auc_raises_saliva_times(self, data):
        with pytest.raises(ValueError):
            saliva.auc(data)

    @pytest.mark.parametrize(
        "biomarker_type, expectation",
        [
            ("cortisol", does_not_raise()),
            ("amylase", pytest.raises(ValueError)),
            ("il6", pytest.raises(ValueError)),
            (None, pytest.raises(ValueError)),
        ],
    )
    def test_auc_raises_biomarker_type(self, biomarker_type, expectation):
        data = saliva_time()
        with expectation:
            saliva.auc(data, biomarker_type=biomarker_type)

    @pytest.mark.parametrize("remove_s0, expected", params_auc)
    def test_auc(self, remove_s0, expected):
        # check with 'time' column
        assert_frame_equal(saliva.auc(saliva_time(), remove_s0=remove_s0), expected, check_dtype=False)

        # check with 'saliva_time' parameter
        assert_frame_equal(
            saliva.auc(saliva_no_time(), remove_s0=remove_s0, saliva_times=[-10, 0, 10, 20, 30]),
            expected,
            check_dtype=False,
        )

        # check that 'saliva_time' parameter overrides 'time' column => set 'time' to 0
        data = saliva_time()
        data["time"] = 0
        assert_frame_equal(
            saliva.auc(data, remove_s0=remove_s0, saliva_times=[-10, 0, 10, 20, 30]),
            expected,
            check_dtype=False,
        )

    @pytest.mark.parametrize("remove_s0, expected", params_auc_mutiple_pre)
    def test_auc_multiple_pre(self, remove_s0, expected):
        # check with 'time' column
        assert_frame_equal(
            saliva.auc(saliva_time_multiple_pre(), remove_s0=remove_s0),
            expected,
            check_dtype=False,
        )
        # check with 'saliva_time' parameter
        assert_frame_equal(
            saliva.auc(
                saliva_time_multiple_pre(),
                remove_s0=remove_s0,
                saliva_times=[-20, -10, 0, 10, 20],
            ),
            expected,
            check_dtype=False,
        )

    def test_auc_pruessner2003(self):
        """
        Tests the AUC examples in the paper from Pruessner et al. (2003).

        References
        ----------
        Pruessner, J. C., Kirschbaum, C., Meinlschmid, G., & Hellhammer, D. H. (2003). Two formulas for computation of
        the area under the curve represent measures of total hormone concentration versus time-dependent change.
        *Psychoneuroendocrinology*, 28(7), 916–931. https://doi.org/10.1016/S0306-4530(02)00108-7
        """
        expected = pd.DataFrame(
            {"cortisol_auc_g": [34.75, 390], "cortisol_auc_i": [20.75, 232.50]},
            columns=pd.Index(["cortisol_auc_g", "cortisol_auc_i"], name="biomarker"),
            index=pd.Index(range(1, 3), name="subject"),
        )

        # assure that warning is emitted because saliva times are not the same for all subjects
        with pytest.warns(UserWarning):
            data_out = saliva.auc(saliva_pruessner_2003(), remove_s0=False)
            # 'auc_i_post' must not be included because saliva times are not the same for all subjects
            assert list(data_out.columns) == ["cortisol_auc_g", "cortisol_auc_i"]
            assert_frame_equal(
                data_out[["cortisol_auc_g", "cortisol_auc_i"]],
                expected,
                check_dtype=False,
            )

    @pytest.mark.parametrize("remove_s0, expected", params_auc)
    def test_auc_multi_days(self, remove_s0, expected):
        out = saliva.auc(saliva_multi_days(), remove_s0=remove_s0)
        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "biomarker_type, expected_columns",
        [
            ("cortisol", ["cortisol_slope01"]),
            ("amylase", ["amylase_slope01"]),
            ("il6", ["il6_slope01"]),
        ],
    )
    def test_slope_columns_biomarker_type(self, biomarker_type, expected_columns):
        data_in = saliva_time(biomarker_type)
        data_out = saliva.slope(data_in, sample_idx=(0, 1), biomarker_type=biomarker_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize(
        "sample_idx, expected_columns",
        [
            ((0, 1), ["cortisol_slope01"]),
            ((1, 2), ["cortisol_slope12"]),
            ((2, 4), ["cortisol_slope24"]),
            ((0, -1), ["cortisol_slope04"]),
        ],
    )
    def test_slope_columns_sample_idx(self, sample_idx, expected_columns):
        # check with sample_idx = tuple
        data_in = saliva_time(biomarker_type="cortisol")
        data_out = saliva.slope(data_in, sample_idx=sample_idx, biomarker_type="cortisol")
        assert list(data_out.columns) == expected_columns

        # check with sample_idx = list
        sample_idx = list(sample_idx)
        data_out = saliva.slope(data_in, sample_idx=sample_idx, biomarker_type="cortisol")
        assert list(data_out.columns) == expected_columns

        # check with sample_idx = numpy array
        sample_idx = np.array(sample_idx)
        data_out = saliva.slope(data_in, sample_idx=sample_idx, biomarker_type="cortisol")
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize("sample_idx", [(0, 0), (1, 1), (1, 0), (10, 6), (0, 10), (0, 1, 2), (1,)])
    def test_slope_invalid_idx(self, sample_idx):
        with pytest.raises(ValueError):
            saliva.slope(saliva_time(), sample_idx=sample_idx, biomarker_type="cortisol")

    @pytest.mark.parametrize("sample_idx, expected", params_slope)
    def test_slope(self, sample_idx, expected):
        data_in = saliva_time()
        data_out = saliva.slope(data_in, sample_idx=sample_idx)

        assert_frame_equal(data_out, expected)

    @pytest.mark.parametrize("sample_idx, expected", params_slope)
    def test_slope_multi_days(self, sample_idx, expected):
        out = saliva.slope(saliva_multi_days(), sample_idx=sample_idx)
        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "biomarker_type, expected_columns",
        [
            (
                "cortisol",
                [
                    "cortisol_argmax",
                    "cortisol_mean",
                    "cortisol_std",
                    "cortisol_skew",
                    "cortisol_kurt",
                ],
            ),
            (
                "amylase",
                [
                    "amylase_argmax",
                    "amylase_mean",
                    "amylase_std",
                    "amylase_skew",
                    "amylase_kurt",
                ],
            ),
            ("il6", ["il6_argmax", "il6_mean", "il6_std", "il6_skew", "il6_kurt"]),
        ],
    )
    def test_standard_features_columns_biomarker_type(self, biomarker_type, expected_columns):
        data_in = saliva_time(biomarker_type)
        data_out = saliva.standard_features(data_in, biomarker_type=biomarker_type)
        # columns must be Index, not MultiIndex
        assert isinstance(data_out.columns, pd.Index)
        assert not isinstance(data_out.columns, pd.MultiIndex)
        # check column names
        assert list(data_out.columns) == expected_columns

    def test_standard_features(self):
        data_in = saliva_time()
        data_out = saliva.standard_features(data_in)

        expected = pd.DataFrame(
            {
                "cortisol_argmax": [0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 2.0],
                "cortisol_mean": [0.0, 1.0, 2.0, 3.0, 5.2, 2.0, 4.0, 5.0],
                "cortisol_std": [
                    0.0,
                    0.0,
                    np.std([0, 1, 2, 3, 4], ddof=1),
                    np.std([5, 4, 3, 2, 1], ddof=1),
                    np.std([10, 2, 4, 4, 6], ddof=1),
                    np.std([-6, 2, 4, 4, 6], ddof=1),
                    np.std([12, -6, 4, 4, 6], ddof=1),
                    np.nanstd([2, np.nan, 8, 4, 6], ddof=1),
                ],
                "cortisol_skew": [
                    ss.skew([0, 0, 0, 0, 0], bias=False),
                    ss.skew([1, 1, 1, 1, 1], bias=False),
                    ss.skew([0, 1, 2, 3, 4], bias=False),
                    ss.skew([5, 4, 3, 2, 1], bias=False),
                    ss.skew([10, 2, 4, 4, 6], bias=False),
                    ss.skew([-6, 2, 4, 4, 6], bias=False),
                    ss.skew([12, -6, 4, 4, 6], bias=False),
                    0.0,
                ],
                "cortisol_kurt": [
                    ss.kurtosis([0, 0, 0, 0, 0], fisher=False, bias=False),
                    ss.kurtosis([1, 1, 1, 1, 1], fisher=False, bias=False),
                    ss.kurtosis([0, 1, 2, 3, 4], bias=False),
                    ss.kurtosis([5, 4, 3, 2, 1], bias=False),
                    ss.kurtosis([10, 2, 4, 4, 6], bias=False),
                    ss.kurtosis([-6, 2, 4, 4, 6], bias=False),
                    ss.kurtosis([12, -6, 4, 4, 6], bias=False),
                    ss.kurtosis([2, np.nan, 8, 4, 6], bias=False, nan_policy="omit"),
                ],
            },
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(
                [
                    "cortisol_argmax",
                    "cortisol_mean",
                    "cortisol_std",
                    "cortisol_skew",
                    "cortisol_kurt",
                ],
                name="biomarker",
            ),
        )
        assert_frame_equal(data_out, expected, check_dtype=False)

    def test_standard_features_multi_days(self):
        out = saliva.standard_features(saliva_multi_days())

        expected = pd.DataFrame(
            {
                "cortisol_argmax": [0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 2.0],
                "cortisol_mean": [0.0, 1.0, 2.0, 3.0, 5.2, 2.0, 4.0, 5.0],
                "cortisol_std": [
                    0.0,
                    0.0,
                    np.std([0, 1, 2, 3, 4], ddof=1),
                    np.std([5, 4, 3, 2, 1], ddof=1),
                    np.std([10, 2, 4, 4, 6], ddof=1),
                    np.std([-6, 2, 4, 4, 6], ddof=1),
                    np.std([12, -6, 4, 4, 6], ddof=1),
                    np.nanstd([2, np.nan, 8, 4, 6], ddof=1),
                ],
                "cortisol_skew": [
                    ss.skew([0, 0, 0, 0, 0], bias=False),
                    ss.skew([1, 1, 1, 1, 1], bias=False),
                    ss.skew([0, 1, 2, 3, 4], bias=False),
                    ss.skew([5, 4, 3, 2, 1], bias=False),
                    ss.skew([10, 2, 4, 4, 6], bias=False),
                    ss.skew([-6, 2, 4, 4, 6], bias=False),
                    ss.skew([12, -6, 4, 4, 6], bias=False),
                    0.0,
                ],
                "cortisol_kurt": [
                    ss.kurtosis([0, 0, 0, 0, 0], fisher=False, bias=False),
                    ss.kurtosis([1, 1, 1, 1, 1], fisher=False, bias=False),
                    ss.kurtosis([0, 1, 2, 3, 4], bias=False),
                    ss.kurtosis([5, 4, 3, 2, 1], bias=False),
                    ss.kurtosis([10, 2, 4, 4, 6], bias=False),
                    ss.kurtosis([-6, 2, 4, 4, 6], bias=False),
                    ss.kurtosis([12, -6, 4, 4, 6], bias=False),
                    ss.kurtosis([2, np.nan, 8, 4, 6], bias=False, nan_policy="omit"),
                ],
            },
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(
                [
                    "cortisol_argmax",
                    "cortisol_mean",
                    "cortisol_std",
                    "cortisol_skew",
                    "cortisol_kurt",
                ],
                name="biomarker",
            ),
        )

        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)
