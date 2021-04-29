from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from unittest import TestCase

import pytest
import pandas as pd
from biopsykit.questionnaires.utils import convert_scale
from biopsykit.utils.exceptions import ValidationError, ValueRangeError

from pandas._testing import assert_frame_equal
from biopsykit.questionnaires import *

from itertools import product


TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/questionnaires")


@contextmanager
def does_not_raise():
    yield


def data_complete_correct() -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_correct.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def data_complete_wrong_range() -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_wrong_range.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def data_filtered_correct(like: Optional[str] = None, regex: Optional[str] = None) -> pd.DataFrame:
    data = data_complete_correct()
    if like is None:
        return data.filter(regex=regex)
    return data.filter(like=like)


def result_filtered(like: Optional[str] = None, regex: Optional[str] = None) -> pd.DataFrame:
    data = pd.read_csv(TEST_FILE_PATH.joinpath("questionnaire_results.csv"))
    data = data.set_index(["subject", "condition"])
    if like is None:
        return data.filter(regex=regex)
    return data.filter(like=like)


def data_filtered_wrong_range(like: Optional[str] = None, regex: Optional[str] = None) -> pd.DataFrame:
    data = data_complete_wrong_range()
    if like is None:
        return data.filter(regex=regex)
    return data.filter(like=like)


def data_subscale(quest_name: str):
    data = pd.read_csv(TEST_FILE_PATH.joinpath("{}_subscale.csv".format(quest_name)))
    data = data.set_index(["subject", "condition"])
    return data


class TestQuestionnaires:
    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("PSS"), None, pytest.raises(ValueRangeError)),
            (data_complete_correct(), ["PSS{}".format(i) for i in range(1, 11)], pytest.raises(ValidationError)),
            (convert_scale(data_filtered_wrong_range("PSS"), -1), None, does_not_raise()),
            (data_filtered_correct("PSS"), None, does_not_raise()),
            (data_filtered_correct("PSS"), ["PSS_{}".format(i) for i in range(1, 11)], does_not_raise()),
            (data_complete_correct(), ["PSS_{}".format(i) for i in range(1, 11)], does_not_raise()),
        ],
    )
    def test_pss_raises(self, data, columns, expected):
        with expected:
            pss(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("PSS"), None, result_filtered("PSS")),
            (data_complete_correct(), ["PSS_{}".format(i) for i in range(1, 11)], result_filtered("PSS")),
            (convert_scale(data_filtered_wrong_range("PSS"), -1), None, result_filtered("PSS")),
        ],
    )
    def test_pss(self, data, columns, result):
        data_out = pss(data, columns)
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range(regex=r"ABI\d"), None, pytest.raises(ValueRangeError)),
            (
                data_filtered_correct(regex=r"ABI\d"),
                ["ABI_{}".format(i) for i in range(1, 11)],
                pytest.raises(ValidationError),
            ),
            (convert_scale(data_filtered_wrong_range(regex=r"ABI\d"), 1), None, does_not_raise()),
            (data_filtered_correct(regex=r"ABI\d"), None, does_not_raise()),
            (
                data_complete_correct(),
                ["ABI{}_{}".format(i, j) for i, j in product(range(1, 9), range(1, 11))],
                does_not_raise(),
            ),
            (
                data_filtered_correct(regex=r"ABI\d"),
                ["ABI{}_{}".format(i, j) for i, j in product(range(1, 9), range(1, 11))],
                does_not_raise(),
            ),
        ],
    )
    def test_abi_raises(self, data, columns, expected):
        with expected:
            abi(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct(regex=r"ABI\d"), None, result_filtered("ABI")),
            (
                data_complete_correct(),
                ["ABI{}_{}".format(i, j) for i, j in product(range(1, 9), range(1, 11))],
                result_filtered("ABI"),
            ),
            (convert_scale(data_filtered_wrong_range(regex=r"ABI\d"), 1), None, result_filtered("ABI")),
        ],
    )
    def test_abi(self, data, columns, result):
        data_out = abi(data, columns)
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("BE"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("BE"), -1), None, None, does_not_raise()),
            (data_filtered_correct("BE"), None, None, does_not_raise()),
            (
                data_filtered_correct("BE"),
                ["BE{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("BE"), ["BE{:02d}".format(i) for i in range(1, 24)], None, does_not_raise()),
            (
                data_filtered_correct("BE"),
                None,
                {
                    "Appearance": [1, 6, 7, 9, 11, 13, 15, 17, 21, 23],
                    "Weight": [3, 4, 8, 10, 16, 18, 19, 22],
                    "Attribution": [2, 5, 12, 14, 20],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("BE"),
                None,
                {
                    "Appearance": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("BE"),
                None,
                {
                    "Appearance": [1, 3, 5, 7, 9, 11, 13],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_besaa_raises(self, data, columns, subscales, expected):
        with expected:
            besaa(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("BE"), None, None, result_filtered("BESAA")),
            (convert_scale(data_filtered_wrong_range("BE"), -1), None, None, result_filtered("BESAA")),
            (
                data_subscale("besaa"),
                None,
                {"Appearance": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                result_filtered("BESAA_Appearance"),
            ),
        ],
    )
    def test_besaa(self, data, columns, subscales, result):
        data_out = besaa(data, columns, subscales)
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("BFI_K"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("BFI_K"), 1), None, None, does_not_raise()),
            (data_filtered_correct("BFI_K"), None, None, does_not_raise()),
            (
                data_filtered_correct("BFI_K"),
                ["BFI_K_{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("BFI_K"), ["BFI_K_{}".format(i) for i in range(1, 22)], None, does_not_raise()),
            (
                data_filtered_correct("BFI_K"),
                None,
                {
                    "E": [1, 6, 11, 16],
                    "A": [2, 7, 12, 17],
                    "C": [3, 8, 13, 18],
                    "N": [4, 9, 14, 19],
                    "O": [5, 10, 15, 20, 21],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("BFI_K"),
                None,
                {
                    "E": [1, 2, 3, 4],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("BFI_K"),
                None,
                {
                    "E": [2, 4],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_bfi_k_raises(self, data, columns, subscales, expected):
        with expected:
            bfi_k(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("BFI_K"), None, None, result_filtered("BFI_K")),
            (convert_scale(data_filtered_wrong_range("BFI_K"), 1), None, None, result_filtered("BFI_K")),
            (
                data_subscale("bfi_k"),
                None,
                {"E": [1, 2, 3, 4]},
                result_filtered("BFI_K_E"),
            ),
        ],
    )
    def test_bfi_k(self, data, columns, subscales, result):
        data_out = bfi_k(data, columns, subscales)
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("BIDR"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("BIDR"), 1), None, None, does_not_raise()),
            (data_filtered_correct("BIDR"), None, None, does_not_raise()),
            (
                data_filtered_correct("BIDR"),
                ["BIDR_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("BIDR"), ["BIDR_{}".format(i) for i in range(1, 21)], None, does_not_raise()),
            (
                data_filtered_correct("BIDR"),
                None,
                {
                    "ST": list(range(1, 11)),
                    "FT": list(range(11, 21)),
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("BIDR"),
                None,
                {
                    "FT": list(range(1, 11)),
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("BIDR"),
                None,
                {"FT": [1, 2, 3, 4, 5, 6, 7]},
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_bidr_raises(self, data, columns, subscales, expected):
        with expected:
            bidr(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("BIDR"), None, None, result_filtered("BIDR")),
            (convert_scale(data_filtered_wrong_range("BIDR"), 1), None, None, result_filtered("BIDR")),
            (
                data_subscale("bidr"),
                None,
                {"FT": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                result_filtered("BIDR_FT"),
            ),
        ],
    )
    def test_bidr(self, data, columns, subscales, result):
        data_out = bidr(data, columns, subscales)
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("Brief_COPE"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("Brief_COPE"), 1), None, None, does_not_raise()),
            (data_filtered_correct("Brief_COPE"), None, None, does_not_raise()),
            (
                data_filtered_correct("Brief_COPE"),
                ["Brief_COPE_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("Brief_COPE"),
                ["Brief_COPE_{}".format(i) for i in range(1, 29)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("Brief_COPE"),
                None,
                {
                    "SelfDistraction": [1, 19],
                    "ActiveCoping": [2, 7],
                    "Denial": [3, 8],
                    "SubstanceUse": [4, 11],
                    "EmotionalSupport": [5, 15],
                    "InstrumentalSupport": [10, 23],
                    "BehavioralDisengagement": [6, 16],
                    "Venting": [9, 21],
                    "PosReframing": [12, 17],
                    "Planning": [14, 25],
                    "Humor": [18, 28],
                    "Acceptance": [20, 24],
                    "Religion": [22, 27],
                    "SelfBlame": [13, 26],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("Brief_COPE"),
                None,
                {
                    "EmotionalSupport": [5, 15],
                    "InstrumentalSupport": [10, 23],
                    "BehavioralDisengagement": [6, 16],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_brief_cope_raises(self, data, columns, subscales, expected):
        with expected:
            brief_cope(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("Brief_COPE"), None, None, result_filtered("Brief_COPE")),
            (convert_scale(data_filtered_wrong_range("Brief_COPE"), 1), None, None, result_filtered("Brief_COPE")),
            (
                data_subscale("brief_cope"),
                None,
                {
                    "EmotionalSupport": [1, 4],
                    "InstrumentalSupport": [3, 6],
                    "BehavioralDisengagement": [2, 5],
                },
                pd.concat(
                    [
                        result_filtered("Brief_COPE_EmotionalSupport"),
                        result_filtered("Brief_COPE_InstrumentalSupport"),
                        result_filtered("Brief_COPE_BehavioralDisengagement"),
                    ],
                    axis=1,
                ),
            ),
        ],
    )
    def test_brief_cope(self, data, columns, subscales, result):
        data_out = brief_cope(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("CESD"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("CESD"), -1), None, does_not_raise()),
            (data_filtered_correct("CESD"), None, does_not_raise()),
            (
                data_filtered_correct("CESD"),
                ["CESD_{}".format(i) for i in range(1, 21)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("CESD"),
                ["CESD{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("CESD"),
                ["CESD{:02d}".format(i) for i in range(1, 21)],
                does_not_raise(),
            ),
        ],
    )
    def test_cesd_raises(self, data, columns, expected):
        with expected:
            cesd(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("CESD"), None, result_filtered("CESD")),
            (convert_scale(data_filtered_wrong_range("CESD"), -1), None, result_filtered("CESD")),
        ],
    )
    def test_cesd(self, data, columns, result):
        data_out = cesd(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)
