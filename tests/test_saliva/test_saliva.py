from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss
from pandas.testing import assert_frame_equal

import biopsykit.saliva as saliva
from biopsykit.utils.exceptions import DataFrameTransformationError, ValidationError


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


def saliva_no_time(saliva_type: Optional[str] = "cortisol"):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=[saliva_type],
    )
    data[saliva_type] = np.concatenate(
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


def saliva_time(saliva_type: Optional[str] = "cortisol"):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=[saliva_type, "time"],
    )
    data[saliva_type] = np.concatenate(
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


def saliva_time_individual(saliva_type: Optional[str] = "cortisol"):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
        columns=[saliva_type, "time"],
    )
    data[saliva_type] = np.concatenate(
        [
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
            [12, -6, 4, 4, 6],
        ]
    )
    data["time"] = np.concatenate(
        [
            [-10, 0, 10, 20, 30],
            [-11, 1, 10, 21, 29],
            [-10, -1, 11, 20, 30],
            [-9, 2, 12, 22, 33],
            [-10, 0, 10, 19, 29],
            [-8, -2, 10, 20, 30],
            [-10, 0, 11, 20, 29],
            [-10, 0, 11, 20, 29],
        ]
    )
    return data


def saliva_idx(saliva_type: Optional[str] = "cortisol"):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [range(1, 9), ["S{}".format(i) for i in range(1, 6)]], names=["subject", "sample"]
        ),
        columns=[saliva_type, "time"],
    )
    data[saliva_type] = np.concatenate(
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


def saliva_idx_multi_types():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [range(1, 9), ["S{}".format(i) for i in range(1, 6)]], names=["subject", "sample"]
        ),
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
    data["amylase"] = np.concatenate(
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


def saliva_multi_types(include_time: Optional[bool] = False):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
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
    data["amylase"] = np.concatenate(
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
    if include_time:
        data["time"] = [-10, 0, 10, 20, 30] * 8
    return data


def saliva_group_col(include_condition: Optional[bool] = False, include_day: Optional[bool] = False):
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product([range(1, 9), range(0, 5)], names=["subject", "sample"]),
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
    if include_condition:
        condition = pd.Series(
            ["Condition", "Control"] * 4, index=pd.Index(range(1, 9), name="subject"), name="condition"
        )
        data = data.join(condition, on="subject")
        data = data.set_index("condition", append=True)
    if include_day:
        day = pd.Series([1, 2] * 4, index=pd.Index(range(1, 9), name="subject"), name="day")
        data = data.join(day, on="subject")
        data = data.set_index("day", append=True)
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


def saliva_multi_days():
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


def saliva_multi_days_idx():
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [range(1, 5), range(0, 2), ["S{}".format(i) for i in range(1, 6)]], names=["subject", "day", "sample"]
        ),
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


params_max_value = [
    (
        True,
        pd.DataFrame(
            [0, 1, 4, 4, 6, 6, 6, 8],
            columns=pd.Index(["cortisol_max_val"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        pd.DataFrame(
            [0, 1, 4, 5, 10, 6, 12, 8],
            columns=pd.Index(["cortisol_max_val"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_initial_value = [
    (
        True,
        pd.DataFrame(
            [0, 1, 1, 4, 2, 2, -6, np.nan],
            columns=pd.Index(["cortisol_ini_val"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        pd.DataFrame(
            [0, 1, 0, 5, 10, -6, 12, 2],
            columns=pd.Index(["cortisol_ini_val"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_max_increase = [
    (
        True,
        pd.DataFrame(
            [0, 0, 3, -1, 4, 4, 12, np.nan],
            columns=pd.Index(["cortisol_max_inc"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        pd.DataFrame(
            [0, 0, 4, -1, -4, 12, -6, 6],
            columns=pd.Index(["cortisol_max_inc"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_max_increase_percent = [
    (
        True,
        pd.DataFrame(
            [np.nan, 0, 300.0, -25.0, 200.0, 200.0, 200.0, np.nan],
            columns=pd.Index(["cortisol_max_inc_percent"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        pd.DataFrame(
            [np.nan, 0.0, np.inf, -20.0, -40.0, 200.0, -50.0, 300.0],
            columns=pd.Index(["cortisol_max_inc_percent"], name="saliva_feature"),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_auc = [
    (
        False,
        True,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 40, 80, 120, 180, 100, 110, np.nan],
                "cortisol_auc_i": [0, 0, 80, -80, -220, 340, -370, np.nan],
                "cortisol_auc_i_post": [0, 0, 45.0, -45.0, 60.0, 60.0, 260.0, np.nan],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="saliva_feature",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        False,
        False,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 40, 80, 120, 180, 100, 110, np.nan],
                "cortisol_auc_i": [0, 0, 80, -80, -220, 340, -370, np.nan],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i"],
                name="saliva_feature",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        True,
        True,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 30, 75, 75, 120, 120, 80, np.nan],
                "cortisol_auc_i": [0, 0, 45, -45, 60, 60, 260, np.nan],
                "cortisol_auc_i_post": [0, 0, 45.0, -45.0, 60.0, 60.0, 260.0, np.nan],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="saliva_feature",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        True,
        False,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 30, 75, 75, 120, 120, 80, np.nan],
                "cortisol_auc_i": [0, 0, 45, -45, 60, 60, 260, np.nan],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i"],
                name="saliva_feature",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_auc_mutiple_pre = [
    (
        False,
        True,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 40, 80, 120, 180, 100, 110, np.nan],
                "cortisol_auc_i": [0, 0, 80, -80, -220, 340, -370, np.nan],
                "cortisol_auc_i_post": [0, 0, 20.0, -20.0, 10.0, 10.0, 10.0, -50.0],
            },
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="saliva_feature",
            ),
        ),
    ),
    (
        False,
        False,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 40, 80, 120, 180, 100, 110, np.nan],
                "cortisol_auc_i": [0, 0, 80, -80, -220, 340, -370, np.nan],
            },
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i"],
                name="saliva_feature",
            ),
        ),
    ),
    (
        True,
        True,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 30, 75, 75, 120, 120, 80, np.nan],
                "cortisol_auc_i": [0, 0, 45, -45, 60, 60, 260, np.nan],
                "cortisol_auc_i_post": [0, 0, 20, -20, 10, 10, 10, -50],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"],
                name="saliva_feature",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
    (
        True,
        False,
        pd.DataFrame(
            {
                "cortisol_auc_g": [0, 30, 75, 75, 120, 120, 80, np.nan],
                "cortisol_auc_i": [0, 0, 45, -45, 60, 60, 260, np.nan],
            },
            columns=pd.Index(
                ["cortisol_auc_g", "cortisol_auc_i"],
                name="saliva_feature",
            ),
            index=pd.Index(range(1, 9), name="subject"),
        ),
    ),
]

params_slope = [
    (
        (0, 1),
        ("S1", "S2"),
        pd.DataFrame(
            {"cortisol_slopeS1S2": [0.0, 0.0, 0.1, -0.1, -0.8, 0.8, -1.8, np.nan]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slopeS1S2"], name="saliva_feature"),
        ),
    ),
    (
        (1, 2),
        ("S2", "S3"),
        pd.DataFrame(
            {"cortisol_slopeS2S3": [0.0, 0.0, 0.1, -0.1, 0.2, 0.2, 1.0, np.nan]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slopeS2S3"], name="saliva_feature"),
        ),
    ),
    (
        (0, 4),
        ("S1", "S5"),
        pd.DataFrame(
            {"cortisol_slopeS1S5": [0.0, 0.0, 0.1, -0.1, -0.1, 0.3, -0.15, 0.1]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slopeS1S5"], name="saliva_feature"),
        ),
    ),
    (
        (0, -1),
        ("S1", "S5"),
        pd.DataFrame(
            {"cortisol_slopeS1S5": [0.0, 0.0, 0.1, -0.1, -0.1, 0.3, -0.15, 0.1]},
            index=pd.Index(range(1, 9), name="subject"),
            columns=pd.Index(["cortisol_slopeS1S5"], name="saliva_feature"),
        ),
    ),
]

# [0, 0, 0, 0, 0],
# [1, 1, 1, 1, 1],
# [0, 1, 2, 3, 4],
# [5, 4, 3, 2, 1],
# [10, 2, 4, 4, 6],
# [-6, 2, 4, 4, 6],
# [12, -6, 4, 4, 6],
# [2, np.nan, 8, 4, 6],

params_mean_se = [
    (
        saliva_no_time("cortisol"),
        False,
        pd.DataFrame(
            {
                "mean": [3.0, 0.57142857, 3.25, 2.75, 3.75],
                "se": [2.06155281, 1.1153688264639723, 0.86085506, 0.55901699, 0.94017476],
            },
            index=pd.Index(list(range(0, 5)), name="sample"),
        ),
    ),
    (
        saliva_no_time("cortisol"),
        True,
        pd.DataFrame(
            {
                "mean": [0.57142857, 3.25, 2.75, 3.75],
                "se": [1.1153688264639723, 0.86085506, 0.55901699, 0.94017476],
            },
            index=pd.Index(list(range(1, 5)), name="sample"),
        ),
    ),
    (
        saliva_time("cortisol"),
        False,
        pd.DataFrame(
            {
                "mean": [3.0, 0.57142857, 3.25, 2.75, 3.75],
                "se": [2.06155281, 1.1153688264639723, 0.86085506, 0.55901699, 0.94017476],
            },
            index=pd.MultiIndex.from_tuples(
                [(i, k) for i, k in enumerate([-10, 0, 10, 20, 30])], names=["sample", "time"]
            ),
        ),
    ),
    (
        saliva_time("cortisol"),
        True,
        pd.DataFrame(
            {
                "mean": [0.57142857, 3.25, 2.75, 3.75],
                "se": [1.1153688264639723, 0.86085506, 0.55901699, 0.94017476],
            },
            index=pd.MultiIndex.from_tuples(
                [(i, k) for i, k in zip(range(1, 5), [0, 10, 20, 30])], names=["sample", "time"]
            ),
        ),
    ),
]


class TestSaliva:
    @pytest.mark.parametrize(
        "input_data",
        [saliva_multiindex_wrong_name(), saliva_no_multiindex(), saliva_none()],
    )
    def test_check_data_format_invalid(self, input_data):
        with pytest.raises(ValidationError):
            saliva.max_increase(input_data)

    @pytest.mark.parametrize(
        "saliva_type, expectation",
        [
            ("cortisol", does_not_raise()),
            ("amylase", pytest.raises(ValidationError)),
            ("il6", pytest.raises(ValidationError)),
            (None, pytest.raises(ValidationError)),
        ],
    )
    def test_max_value_raises_saliva_type(self, saliva_type, expectation):
        data = saliva_time()
        with expectation:
            saliva.max_value(data, saliva_type=saliva_type)

    @pytest.mark.parametrize(
        "saliva_type, expected_columns",
        [
            ("cortisol", ["cortisol_max_val"]),
            ("amylase", ["amylase_max_val"]),
            ("il6", ["il6_max_val"]),
        ],
    )
    def test_max_value_columns(self, saliva_type, expected_columns):
        data_in = saliva_time(saliva_type)
        data_out = saliva.max_value(data_in, saliva_type=saliva_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize("remove_s0, expected", params_max_value)
    def test_max_value(self, remove_s0, expected):
        out = saliva.max_value(saliva_no_time(), remove_s0=remove_s0)
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "include_time",
        [
            True,
            False,
        ],
    )
    def test_max_value_multi_saliva_types(self, include_time):
        data_in = saliva_multi_types(include_time)
        data_out = saliva.max_value(data_in, saliva_type=["cortisol", "amylase"])
        [
            assert_frame_equal(
                saliva.max_value(data_in[[saliva_type]], saliva_type=saliva_type),
                data_out[saliva_type],
            )
            for saliva_type in ["cortisol", "amylase"]
        ]

    @pytest.mark.parametrize(
        "saliva_type, expectation",
        [
            ("cortisol", does_not_raise()),
            ("amylase", pytest.raises(ValidationError)),
            ("il6", pytest.raises(ValidationError)),
            (None, pytest.raises(ValidationError)),
        ],
    )
    def test_initial_value_raises_saliva_type(self, saliva_type, expectation):
        data = saliva_time()
        with expectation:
            saliva.initial_value(data, saliva_type=saliva_type)

    @pytest.mark.parametrize(
        "saliva_type, expected_columns",
        [
            ("cortisol", ["cortisol_ini_val"]),
            ("amylase", ["amylase_ini_val"]),
            ("il6", ["il6_ini_val"]),
        ],
    )
    def test_initial_value_columns(self, saliva_type, expected_columns):
        data_in = saliva_time(saliva_type)
        data_out = saliva.initial_value(data_in, saliva_type=saliva_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize("remove_s0, expected", params_initial_value)
    def test_initial_value(self, remove_s0, expected):
        out = saliva.initial_value(saliva_no_time(), remove_s0=remove_s0)
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "include_time",
        [
            True,
            False,
        ],
    )
    def test_initial_value_multi_saliva_types(self, include_time):
        data_in = saliva_multi_types(include_time)
        data_out = saliva.initial_value(data_in, saliva_type=["cortisol", "amylase"])
        [
            assert_frame_equal(
                saliva.initial_value(data_in[[saliva_type]], saliva_type=saliva_type),
                data_out[saliva_type],
            )
            for saliva_type in ["cortisol", "amylase"]
        ]

    @pytest.mark.parametrize(
        "saliva_type, expectation",
        [
            ("cortisol", does_not_raise()),
            ("amylase", pytest.raises(ValidationError)),
            ("il6", pytest.raises(ValidationError)),
            (None, pytest.raises(ValidationError)),
        ],
    )
    def test_max_increase_raises_saliva_type(self, saliva_type, expectation):
        data = saliva_time()
        with expectation:
            saliva.max_increase(data, saliva_type=saliva_type)

    @pytest.mark.parametrize(
        "saliva_type, expected_columns",
        [
            ("cortisol", ["cortisol_max_inc"]),
            ("amylase", ["amylase_max_inc"]),
            ("il6", ["il6_max_inc"]),
        ],
    )
    def test_max_increase_columns(self, saliva_type, expected_columns):
        data_in = saliva_time(saliva_type)
        data_out = saliva.max_increase(data_in, saliva_type=saliva_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize("remove_s0, expected", params_max_increase)
    def test_max_increase(self, remove_s0, expected):
        out = saliva.max_increase(saliva_no_time(), remove_s0=remove_s0)
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "include_time",
        [
            True,
            False,
        ],
    )
    def test_max_increase_multi_saliva_types(self, include_time):
        data_in = saliva_multi_types(include_time)
        data_out = saliva.max_increase(data_in, saliva_type=["cortisol", "amylase"])
        [
            assert_frame_equal(
                saliva.max_increase(data_in[[saliva_type]], saliva_type=saliva_type),
                data_out[saliva_type],
            )
            for saliva_type in ["cortisol", "amylase"]
        ]

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
        "saliva_type, compute_auc_post, expected_columns",
        [
            ("cortisol", True, ["cortisol_auc_g", "cortisol_auc_i", "cortisol_auc_i_post"]),
            ("cortisol", False, ["cortisol_auc_g", "cortisol_auc_i"]),
            ("amylase", True, ["amylase_auc_g", "amylase_auc_i", "amylase_auc_i_post"]),
            ("amylase", False, ["amylase_auc_g", "amylase_auc_i"]),
            ("il6", True, ["il6_auc_g", "il6_auc_i", "il6_auc_i_post"]),
            ("il6", False, ["il6_auc_g", "il6_auc_i"]),
        ],
    )
    def test_auc_columns(self, saliva_type, compute_auc_post, expected_columns):
        data_in = saliva_time(saliva_type)
        data_out = saliva.auc(data_in, saliva_type, compute_auc_post=compute_auc_post)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize(
        "data, sample_times, expected",
        [
            (saliva_no_time(), None, pytest.raises(ValueError)),
            (saliva_wrong_time_01(), None, pytest.raises(ValueError)),
            (saliva_wrong_time_02(), None, pytest.raises(ValueError)),
            (
                saliva_no_time(),
                [[-10, 0, 10, 20]],
                pytest.raises(ValueError),
            ),
            (
                saliva_no_time(),
                [[-10, 0, 10, 20, 30, 40]],
                pytest.raises(ValueError),
            ),
            (saliva_no_time(), [[-10, 0, 10, 20, 30], [-10, 0, 10, 20, 30]], pytest.raises(ValueError)),
            (
                saliva_no_time(),
                [
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                    [-10, 0, 10, 20, 30, 40],
                ],
                pytest.raises(ValueError),
            ),
            (
                saliva_no_time(),
                [
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                ],
                pytest.raises(ValueError),
            ),
            (
                saliva_no_time(),
                [
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                    [[-10, 0], [0, 10], [10, 20], [20, 30], [30, 40]],
                ],
                pytest.raises(ValueError),
            ),
            (
                saliva_no_time(),
                [
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                    [[-10], [0], [10], [20], [30]],
                ],
                does_not_raise(),
            ),
            (
                saliva_no_time(),
                [
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                    [-10, 0, 10, 20, 30],
                ],
                does_not_raise(),
            ),
        ],
        ids=[
            "no_time",
            "same_time",
            "decreasing_time",
            "wrong_shape01",
            "wrong_shape02",
            "wrong_shape03",
            "wrong_shape04",
            "wrong_shape05",
            "wrong_shape06",
            "wrong_shape_but_correct_after_squeeze",
            "correct_shape",
        ],
    )
    def test_auc_raises_saliva_times(self, data, sample_times, expected):
        with expected:
            saliva.auc(data, sample_times=sample_times)

    @pytest.mark.parametrize(
        "saliva_type, expectation",
        [
            ("cortisol", does_not_raise()),
            ("amylase", pytest.raises(ValidationError)),
            ("il6", pytest.raises(ValidationError)),
            (None, pytest.raises(ValidationError)),
        ],
    )
    def test_auc_raises_saliva_type(self, saliva_type, expectation):
        data = saliva_time()
        with expectation:
            saliva.auc(data, saliva_type=saliva_type)

    @pytest.mark.parametrize("remove_s0, compute_auc_post, expected", params_auc)
    def test_auc(self, remove_s0, compute_auc_post, expected):
        # check with 'time' column
        assert_frame_equal(
            saliva.auc(saliva_time(), remove_s0=remove_s0, compute_auc_post=compute_auc_post),
            expected,
            check_dtype=False,
        )

        # check with 'saliva_time' parameter
        assert_frame_equal(
            saliva.auc(
                saliva_no_time(),
                remove_s0=remove_s0,
                compute_auc_post=compute_auc_post,
                sample_times=[-10, 0, 10, 20, 30],
            ),
            expected,
            check_dtype=False,
        )

        # check that 'saliva_time' parameter overrides 'time' column => set 'time' to 0
        data = saliva_time()
        data["time"] = 0
        assert_frame_equal(
            saliva.auc(data, remove_s0=remove_s0, compute_auc_post=compute_auc_post, sample_times=[-10, 0, 10, 20, 30]),
            expected,
            check_dtype=False,
        )

    def test_auc_individual_time(self):
        # check with 'time' column
        data_out = saliva.auc(saliva_time_individual())
        # only two AUC values should be the same
        assert len(data_out["cortisol_auc_g"]) - 1 == len(data_out["cortisol_auc_g"].unique())
        assert len(data_out["cortisol_auc_i"]) - 1 == len(data_out["cortisol_auc_i"].unique())
        # these two values should be the same
        assert data_out["cortisol_auc_g"].iloc[-1] == data_out["cortisol_auc_g"].iloc[-2]
        assert data_out["cortisol_auc_i"].iloc[-1] == data_out["cortisol_auc_i"].iloc[-2]
        # these two, for example, not
        assert data_out["cortisol_auc_g"].iloc[0] != data_out["cortisol_auc_g"].iloc[1]
        assert data_out["cortisol_auc_i"].iloc[0] != data_out["cortisol_auc_i"].iloc[1]

    @pytest.mark.parametrize("remove_s0, compute_auc_post, expected", params_auc_mutiple_pre)
    def test_auc_multiple_pre(self, remove_s0, compute_auc_post, expected):
        # check with 'time' column
        assert_frame_equal(
            saliva.auc(saliva_time_multiple_pre(), remove_s0=remove_s0, compute_auc_post=compute_auc_post),
            expected,
            check_dtype=False,
        )
        # check with 'saliva_time' parameter
        assert_frame_equal(
            saliva.auc(
                saliva_time_multiple_pre(),
                remove_s0=remove_s0,
                compute_auc_post=compute_auc_post,
                sample_times=[-20, -10, 0, 10, 20],
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
            columns=pd.Index(["cortisol_auc_g", "cortisol_auc_i"], name="saliva_feature"),
            index=pd.Index(range(1, 3), name="subject"),
        )

        # assure that warning is emitted because saliva times are not the same for all subjects
        # if `compute_auc_post` is True
        with pytest.warns(UserWarning):
            data_out = saliva.auc(saliva_pruessner_2003(), compute_auc_post=True, remove_s0=False)
            # 'auc_i_post' must not be included because saliva times are not the same for all subjects
            assert list(data_out.columns) == ["cortisol_auc_g", "cortisol_auc_i"]
            assert_frame_equal(
                data_out[["cortisol_auc_g", "cortisol_auc_i"]],
                expected,
                check_dtype=False,
            )

    @pytest.mark.parametrize("remove_s0, compute_auc_post, expected", params_auc)
    def test_auc_multi_days(self, remove_s0, compute_auc_post, expected):
        out = saliva.auc(saliva_multi_days(), compute_auc_post=compute_auc_post, remove_s0=remove_s0)
        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "include_time, saliva_times",
        [
            (True, None),
            (False, [-10, 0, 10, 20, 30]),
        ],
    )
    def test_auc_multi_saliva_types(self, include_time, saliva_times):
        data_in = saliva_multi_types(include_time)
        data_out = saliva.auc(data_in, saliva_type=["cortisol", "amylase"], sample_times=saliva_times)
        for saliva_type in ["cortisol", "amylase"]:
            if include_time:
                cols = [saliva_type, "time"]
            else:
                cols = [saliva_type]
            assert_frame_equal(
                saliva.auc(data_in[cols], saliva_type=saliva_type, sample_times=saliva_times),
                data_out[saliva_type],
            )

    @pytest.mark.parametrize(
        "saliva_type, expected_columns",
        [
            ("cortisol", ["cortisol_slope01"]),
            ("amylase", ["amylase_slope01"]),
            ("il6", ["il6_slope01"]),
        ],
    )
    def test_slope_columns_saliva_type(self, saliva_type, expected_columns):
        data_in = saliva_time(saliva_type)
        data_out = saliva.slope(data_in, sample_idx=(0, 1), saliva_type=saliva_type)
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize(
        "sample_labels, sample_idx, expected",
        [
            (None, None, pytest.raises(IndexError)),
            ((1, 2), (0, 1), pytest.raises(IndexError)),
            (None, (0, 1), does_not_raise()),
            (("S1", "S2"), None, does_not_raise()),
        ],
    )
    def test_slope_params(self, sample_labels, sample_idx, expected):
        with expected:
            saliva.slope(
                saliva_idx(saliva_type="cortisol"),
                sample_labels=sample_labels,
                sample_idx=sample_idx,
                saliva_type="cortisol",
            )

    @pytest.mark.parametrize(
        "sample_idx, expected_columns",
        [
            ((0, 1), ["cortisol_slopeS1S2"]),
            ((1, 2), ["cortisol_slopeS2S3"]),
            ((2, 4), ["cortisol_slopeS3S5"]),
            ((0, -1), ["cortisol_slopeS1S5"]),
        ],
    )
    def test_slope_columns_sample_idx(self, sample_idx, expected_columns):
        # check with sample_idx = tuple
        data_in = saliva_idx(saliva_type="cortisol")
        data_out = saliva.slope(data_in, sample_idx=sample_idx, saliva_type="cortisol")
        assert list(data_out.columns) == expected_columns

        # check with sample_idx = list
        sample_idx = list(sample_idx)
        data_out = saliva.slope(data_in, sample_idx=sample_idx, saliva_type="cortisol")
        assert list(data_out.columns) == expected_columns

        # check with sample_idx = numpy array
        sample_idx = np.array(sample_idx)
        data_out = saliva.slope(data_in, sample_idx=sample_idx, saliva_type="cortisol")
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize(
        "sample_labels, expected_columns",
        [
            (("S1", "S2"), ["cortisol_slopeS1S2"]),
            (("S2", "S3"), ["cortisol_slopeS2S3"]),
            (("S3", "S5"), ["cortisol_slopeS3S5"]),
        ],
    )
    def test_slope_columns_sample_labels(self, sample_labels, expected_columns):
        # check with sample_idx = tuple
        data_in = saliva_idx(saliva_type="cortisol")
        data_out = saliva.slope(data_in, sample_labels=sample_labels, saliva_type="cortisol")
        assert list(data_out.columns) == expected_columns

        # check with sample_idx = list
        sample_labels = list(sample_labels)
        data_out = saliva.slope(data_in, sample_labels=sample_labels, saliva_type="cortisol")
        assert list(data_out.columns) == expected_columns

        # check with sample_idx = numpy array
        sample_labels = np.array(sample_labels)
        data_out = saliva.slope(data_in, sample_labels=sample_labels, saliva_type="cortisol")
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize(
        "sample_idx", [(0, 0), (1, 1), (-1, -1), (1, 0), (-2, 1), (10, 6), (0, 10), (0, 1, 2), (1,)]
    )
    def test_slope_invalid_idx(self, sample_idx):
        with pytest.raises(IndexError):
            saliva.slope(saliva_time(), sample_idx=sample_idx, saliva_type="cortisol")

    @pytest.mark.parametrize(
        "sample_labels",
        [
            ("S0", "S1"),
            ("S1", "S1"),
            ("S4", "S6"),
            ("S2", "S1"),
            ("S5", "S1"),
            ("S10", "S6"),
            ("S1", "S10"),
            ("S1", "S2", "S3"),
            ("S2",),
        ],
    )
    def test_slope_invalid_labels(self, sample_labels):
        with pytest.raises(IndexError):
            saliva.slope(saliva_idx(), sample_labels=sample_labels, saliva_type="cortisol")

    @pytest.mark.parametrize("sample_idx, sample_labels, expected", params_slope)
    def test_slope(self, sample_idx, sample_labels, expected):
        data_in = saliva_idx()
        data_out = saliva.slope(data_in, sample_idx=sample_idx)
        assert_frame_equal(data_out, expected)
        data_out = saliva.slope(data_in, sample_labels=sample_labels)
        assert_frame_equal(data_out, expected)

    @pytest.mark.parametrize("sample_idx, sample_labels, expected", params_slope)
    def test_slope_multi_days(self, sample_idx, sample_labels, expected):
        out = saliva.slope(saliva_multi_days_idx(), sample_idx=sample_idx)
        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

        out = saliva.slope(saliva_multi_days_idx(), sample_labels=sample_labels)
        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize("sample_idx, sample_labels, expected", params_slope)
    def test_slope_multi_saliva_types(self, sample_idx, sample_labels, expected):
        data_in = saliva_idx_multi_types()
        data_out = saliva.slope(data_in, saliva_type=["cortisol", "amylase"], sample_idx=sample_idx)
        for saliva_type in ["cortisol", "amylase"]:
            assert_frame_equal(
                saliva.slope(data_in[[saliva_type, "time"]], saliva_type=saliva_type, sample_idx=sample_idx),
                data_out[saliva_type],
            )

        data_out = saliva.slope(data_in, saliva_type=["cortisol", "amylase"], sample_labels=sample_labels)
        for saliva_type in ["cortisol", "amylase"]:
            assert_frame_equal(
                saliva.slope(data_in[[saliva_type, "time"]], saliva_type=saliva_type, sample_labels=sample_labels),
                data_out[saliva_type],
            )

    @pytest.mark.parametrize(
        "saliva_type, expected_columns",
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
    def test_standard_features_columns_saliva_type(self, saliva_type, expected_columns):
        data_in = saliva_time(saliva_type)
        data_out = saliva.standard_features(data_in, saliva_type=saliva_type)
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
                    ss.skew(np.array([0, 0, 0, 0, 0]), bias=False),
                    ss.skew(np.array([1, 1, 1, 1, 1]), bias=False),
                    ss.skew(np.array([0, 1, 2, 3, 4]), bias=False),
                    ss.skew(np.array([5, 4, 3, 2, 1]), bias=False),
                    ss.skew(np.array([10, 2, 4, 4, 6]), bias=False),
                    ss.skew(np.array([-6, 2, 4, 4, 6]), bias=False),
                    ss.skew(np.array([12, -6, 4, 4, 6]), bias=False),
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
                name="saliva_feature",
            ),
        )
        assert_frame_equal(data_out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "group_cols, keep_index, expected",
        [
            (None, True, does_not_raise()),
            (["subject", "day"], True, does_not_raise()),
            (["day"], True, pytest.raises(DataFrameTransformationError)),
            (["subject"], True, pytest.raises(DataFrameTransformationError)),
            (["sample"], True, pytest.raises(DataFrameTransformationError)),
            (None, False, does_not_raise()),
            (["subject", "day"], False, does_not_raise()),
            (["day"], False, pytest.raises(ValidationError)),
            (["subject"], False, does_not_raise()),
            (["sample"], False, pytest.raises(ValidationError)),
            (["condition"], True, pytest.raises(ValueError)),
            (["condition"], False, pytest.raises(ValueError)),
        ],
    )
    def test_standard_features_group_col_raise(self, group_cols, keep_index, expected):
        with expected:
            saliva.standard_features(saliva_multi_days(), group_cols=group_cols, keep_index=keep_index)

    @pytest.mark.parametrize(
        "group_cols, keep_index, expected",
        [
            (None, True, ["subject", "day"]),
            (["subject", "day"], True, ["subject", "day"]),
            (None, False, ["subject", "day"]),
            (["subject", "day"], False, ["subject", "day"]),
            (["subject"], False, ["subject"]),
            ("subject", False, ["subject"]),
        ],
    )
    def test_standard_features_group_col_index(self, group_cols, keep_index, expected):
        out = saliva.standard_features(saliva_multi_days(), group_cols=group_cols, keep_index=keep_index)
        assert list(out.index.names) == list(expected)

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
                    ss.skew(np.array([0, 0, 0, 0, 0]), bias=False),
                    ss.skew(np.array([1, 1, 1, 1, 1]), bias=False),
                    ss.skew(np.array([0, 1, 2, 3, 4]), bias=False),
                    ss.skew(np.array([5, 4, 3, 2, 1]), bias=False),
                    ss.skew(np.array([10, 2, 4, 4, 6]), bias=False),
                    ss.skew(np.array([-6, 2, 4, 4, 6]), bias=False),
                    ss.skew(np.array([12, -6, 4, 4, 6]), bias=False),
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
                name="saliva_feature",
            ),
        )

        # set correct index to the expected output
        expected.index = pd.MultiIndex.from_product([range(1, 5), range(0, 2)], names=["subject", "day"])
        assert_frame_equal(out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "include_time",
        [
            True,
            False,
        ],
    )
    def test_standard_features_multi_saliva_types(self, include_time):
        data_in = saliva_multi_types(include_time)
        data_out = saliva.standard_features(data_in, saliva_type=["cortisol", "amylase"])
        for saliva_type in ["cortisol", "amylase"]:
            assert_frame_equal(
                saliva.standard_features(data_in[[saliva_type]], saliva_type=saliva_type),
                data_out[saliva_type],
            )

    @pytest.mark.parametrize(
        "include_condition, include_day, expected_index_levels",
        [
            (True, True, ["subject", "condition", "day"]),
            (True, False, ["subject", "condition"]),
            (False, True, ["subject", "day"]),
        ],
    )
    def test_standard_features_group_col(self, include_condition, include_day, expected_index_levels):
        data_in = saliva_group_col(include_condition, include_day)
        group_cols = ["subject"]
        if include_condition:
            group_cols.append("condition")
        if include_day:
            group_cols.append("day")
        data_out = saliva.standard_features(data_in, group_cols=group_cols)
        assert list(data_out.index.names) == expected_index_levels

    @pytest.mark.parametrize(
        "data_func, saliva_type, expected_columns, expected_index",
        [
            (saliva_no_time, "cortisol", ["mean", "se"], list(range(0, 5))),
            (saliva_no_time, "amylase", ["mean", "se"], list(range(0, 5))),
            (saliva_no_time, "il6", ["mean", "se"], list(range(0, 5))),
            (saliva_time, "cortisol", ["mean", "se"], [(i, k) for i, k in enumerate([-10, 0, 10, 20, 30])]),
            (saliva_time, "amylase", ["mean", "se"], [(i, k) for i, k in enumerate([-10, 0, 10, 20, 30])]),
            (saliva_time, "il6", ["mean", "se"], [(i, k) for i, k in enumerate([-10, 0, 10, 20, 30])]),
            (
                saliva_idx,
                "cortisol",
                ["mean", "se"],
                [("S{}".format(i), k) for i, k in zip(range(1, 6), [-10, 0, 10, 20, 30])],
            ),
            (
                saliva_idx,
                "amylase",
                ["mean", "se"],
                [("S{}".format(i), k) for i, k in zip(range(1, 6), [-10, 0, 10, 20, 30])],
            ),
            (
                saliva_idx,
                "il6",
                ["mean", "se"],
                [("S{}".format(i), k) for i, k in zip(range(1, 6), [-10, 0, 10, 20, 30])],
            ),
        ],
    )
    def test_mean_se_columns_saliva_type(self, data_func, saliva_type, expected_columns, expected_index):
        data_in = data_func(saliva_type)
        data_out = saliva.mean_se(data_in, saliva_type=saliva_type)
        # columns must be Index, not MultiIndex
        assert isinstance(data_out.columns, pd.Index)
        assert not isinstance(data_out.columns, pd.MultiIndex)
        # check index names
        assert list(data_out.index) == expected_index
        # check column names
        assert list(data_out.columns) == expected_columns

    @pytest.mark.parametrize("data_in, remove_s0, expected", params_mean_se)
    def test_mean_se(self, data_in, remove_s0, expected):
        data_out = saliva.mean_se(data_in, remove_s0=remove_s0)
        assert_frame_equal(data_out, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "data_in, group_cols, expected",
        [
            (saliva_no_time(), None, does_not_raise()),
            (saliva_no_time(), ["subject"], pytest.raises(DataFrameTransformationError)),
            (saliva_no_time(), ["subject", "sample"], pytest.raises(DataFrameTransformationError)),
            (saliva_no_time(), ["day"], pytest.raises(ValueError)),
            (saliva_multi_days(), None, does_not_raise()),
            (saliva_multi_days(), ["subject", "day"], pytest.raises(DataFrameTransformationError)),
            (saliva_multi_days(), ["day"], pytest.raises(DataFrameTransformationError)),
            (saliva_multi_days(), ["subject"], pytest.raises(DataFrameTransformationError)),
        ],
    )
    def test_mean_se_group_col_raise(self, data_in, group_cols, expected):
        with expected:
            saliva.mean_se(data_in, group_cols=group_cols)

    @pytest.mark.parametrize(
        "data_in, group_cols, expected",
        [
            (saliva_no_time(), None, ["sample"]),
            (saliva_time(), None, ["sample", "time"]),
            (saliva_multi_days(), None, ["day", "sample", "time"]),
            (saliva_no_time(), ["sample"], ["sample"]),
            (saliva_no_time(), "sample", ["sample"]),
            (saliva_time(), ["sample"], ["sample", "time"]),
            (saliva_multi_days(), ["sample"], ["sample", "time"]),
            (saliva_time(), ["sample", "time"], ["sample", "time"]),
            (saliva_multi_days(), ["sample", "time"], ["sample", "time"]),
            (saliva_multi_days(), ["sample", "time", "day"], ["sample", "time", "day"]),
        ],
    )
    def test_mean_se_group_col_index(self, data_in, group_cols, expected):
        out = saliva.mean_se(data_in, group_cols=group_cols)
        assert list(out.index.names) == list(expected)

    @pytest.mark.parametrize(
        "include_time",
        [
            True,
            False,
        ],
    )
    def test_mean_se_multi_saliva_types(self, include_time):
        data_in = saliva_multi_types(include_time)
        data_out = saliva.mean_se(data_in, saliva_type=["cortisol", "amylase"])
        for saliva_type in ["cortisol", "amylase"]:
            if include_time:
                data_test = data_in[[saliva_type, "time"]]
            else:
                data_test = data_in[[saliva_type]]
            assert_frame_equal(
                saliva.mean_se(data_test, saliva_type=saliva_type),
                data_out[saliva_type],
            )
