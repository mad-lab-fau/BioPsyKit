from contextlib import contextmanager

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from biopsykit.metadata import bmi, whr
from biopsykit.utils.exceptions import ValueRangeError


@contextmanager
def does_not_raise():
    yield


def data_complete():
    return pd.DataFrame(
        {
            "weight": [68, 68, 64],
            "height": [165, 178, 190],
            "waist": [76, 92, 0.71],
            "hip": [97, 112, 0.89],
        }
    )


def bmi_correct():
    return pd.DataFrame(
        {
            "weight": [68, 68, 64],
            "height": [165, 178, 190],
        }
    )


def bmi_wrong_order():
    return pd.DataFrame(
        {
            "height": [165, 178],
            "weight": [68, 68],
        }
    )


def whr_correct():
    return pd.DataFrame(
        {
            "waist": [76, 92, 0.71],
            "hip": [97, 112, 0.89],
        }
    )


def whr_wrong_values():
    return pd.DataFrame(
        {
            "hip": [50, 4, 0.5],
            "waist": [100, 20, 0.1],
        }
    )


def bmi_correct_solution():
    return pd.DataFrame({"BMI": [24.98, 21.46, 17.73]})


def whr_correct_solution():
    return pd.DataFrame({"WHR": [0.784, 0.821, 0.798]})


class TestMetadata:
    @pytest.mark.parametrize(
        "input_data, expected",
        [(bmi_correct(), does_not_raise()), (bmi_wrong_order(), pytest.raises(ValueRangeError))],
    )
    def test_bmi_raises(self, input_data, expected):
        with expected:
            bmi(input_data)

    @pytest.mark.parametrize(
        "input_data, columns, expected",
        [(bmi_correct(), None, bmi_correct_solution()), (bmi_correct(), ["weight", "height"], bmi_correct_solution())],
    )
    def test_bmi(self, input_data, columns, expected):
        data_out = bmi(input_data, columns)
        assert_frame_equal(data_out, expected)

    @pytest.mark.parametrize(
        "input_data, expected",
        [(whr_correct(), does_not_raise()), (whr_wrong_values(), pytest.raises(ValueRangeError))],
    )
    def test_whr_raises(self, input_data, expected):
        with expected:
            whr(input_data)

    @pytest.mark.parametrize(
        "input_data, columns, expected",
        [(whr_correct(), None, whr_correct_solution()), (data_complete(), ["waist", "hip"], whr_correct_solution())],
    )
    def test_whr(self, input_data, columns, expected):
        data_out = whr(input_data, columns)
        assert_frame_equal(data_out, expected)
