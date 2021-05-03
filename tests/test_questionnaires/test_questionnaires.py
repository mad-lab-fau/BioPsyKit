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


def fee_english():
    data = pd.read_csv(TEST_FILE_PATH.joinpath("fee_english.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def fee_wrong():
    data = pd.read_csv(TEST_FILE_PATH.joinpath("fee_wrong.csv"))
    data = data.set_index(["subject", "condition"])
    return data


def panas_results_german():
    data = pd.read_csv(TEST_FILE_PATH.joinpath("panas_results_german.csv"))
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
            (data_filtered_correct("BE"), ["BE{:02d}".format(i) for i in range(1, 24)], None, result_filtered("BESAA")),
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
            (
                data_filtered_correct("BFI_K"),
                ["BFI_K_{}".format(i) for i in range(1, 22)],
                None,
                result_filtered("BFI_K"),
            ),
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
            (data_filtered_correct("BIDR"), ["BIDR_{}".format(i) for i in range(1, 21)], None, result_filtered("BIDR")),
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
            (
                data_filtered_correct("Brief_COPE"),
                ["Brief_COPE_{}".format(i) for i in range(1, 29)],
                None,
                result_filtered("Brief_COPE"),
            ),
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
            (
                data_filtered_correct("CESD"),
                ["CESD{:02d}".format(i) for i in range(1, 21)],
                result_filtered("CESD"),
            ),
            (convert_scale(data_filtered_wrong_range("CESD"), -1), None, result_filtered("CESD")),
        ],
    )
    def test_cesd(self, data, columns, result):
        data_out = cesd(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("CTQ"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("CTQ"), 1), None, None, does_not_raise()),
            (data_filtered_correct("CTQ"), None, None, does_not_raise()),
            (
                data_filtered_correct("CTQ"),
                ["CTQ_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("CTQ"),
                ["CTQ_{}".format(i) for i in range(1, 29)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("CTQ"),
                ["CTQ{:02d}".format(i) for i in range(1, 29)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("CTQ"),
                None,
                {
                    "PhysicalAbuse": [9, 11, 12, 15, 17],
                    "SexualAbuse": [20, 21, 23, 24, 27],
                    "EmotionalNeglect": [5, 7, 13, 19, 28],
                    "PhysicalNeglect": [1, 2, 4, 6, 26],
                    "EmotionalAbuse": [3, 8, 14, 18, 25],
                    "Validity": [10, 16, 22],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("CTQ"),
                None,
                {
                    "PhysicalNeglect": [1, 2, 4],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_ctq_raises(self, data, columns, subscales, expected):
        with expected:
            ctq(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("CTQ"), None, None, result_filtered("CTQ")),
            (data_filtered_correct("CTQ"), ["CTQ_{}".format(i) for i in range(1, 29)], None, result_filtered("CTQ")),
            (convert_scale(data_filtered_wrong_range("CTQ"), 1), None, None, result_filtered("CTQ")),
            (
                data_subscale("ctq"),
                None,
                {
                    "PhysicalNeglect": [1, 2, 3, 4, 5],
                },
                result_filtered("CTQ_PhysicalNeglect"),
            ),
        ],
    )
    def test_ctq(self, data, columns, subscales, result):
        data_out = ctq(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, language, expected",
        [
            (data_complete_correct(), None, None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("FEE"), None, None, "german", pytest.raises(ValueRangeError)),
            (data_filtered_wrong_range("FEE"), None, None, "spanish", pytest.raises(ValueError)),
            (convert_scale(data_filtered_wrong_range("FEE"), 1), None, None, "german", does_not_raise()),
            (data_filtered_correct("FEE"), None, None, "german", does_not_raise()),
            (data_filtered_correct("FEE"), None, None, None, pytest.raises(ValidationError)),
            (fee_english(), None, None, None, does_not_raise()),
            (fee_english(), None, None, "german", pytest.raises(ValidationError)),
            (fee_wrong(), None, None, None, pytest.raises(ValidationError)),
            (
                data_filtered_correct("FEE"),
                ["FEE_{}".format(i) for i in range(1, 25)],
                None,
                "german",
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("FEE"),
                ["FEE_{}_{}".format(i, e) for i, e in product(range(1, 10), ["Mutter", "Vater"])],
                None,
                "german",
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("FEE"),
                ["FEE_{}_{}".format(i, e) for i, e in product(range(1, 25), ["Mutter", "Vater"])],
                None,
                "german",
                does_not_raise(),
            ),
            (
                data_filtered_correct("FEE"),
                None,
                {
                    "RejectionPunishment": [1, 3, 6, 8, 16, 18, 20, 22],
                    "EmotionalWarmth": [2, 7, 9, 12, 14, 15, 17, 24],
                    "ControlOverprotection": [4, 5, 10, 11, 13, 19, 21, 23],
                },
                "german",
                does_not_raise(),
            ),
        ],
    )
    def test_fee_raises(self, data, columns, subscales, language, expected):
        with expected:
            fee(data, columns, subscales, language)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("FEE"), None, None, result_filtered("FEE")),
            (
                data_filtered_correct("FEE"),
                ["FEE_{}_{}".format(i, e) for i, e in product(range(1, 25), ["Mutter", "Vater"])],
                None,
                result_filtered("FEE"),
            ),
            (convert_scale(data_filtered_wrong_range("FEE"), 1), None, None, result_filtered("FEE")),
            (
                data_subscale("fee"),
                None,
                {
                    "RejectionPunishment": [1, 2, 3, 4, 5, 6, 7, 8],
                },
                result_filtered("FEE_RejectionPunishment"),
            ),
        ],
    )
    def test_fee(self, data, columns, subscales, result):
        data_out = fee(data, columns, subscales, language="german")
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("FKK"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("FKK"), 1), None, None, does_not_raise()),
            (data_filtered_correct("FKK"), None, None, does_not_raise()),
            (
                data_filtered_correct("FKK"),
                ["FKK_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("FKK"),
                ["FKK_{}".format(i) for i in range(1, 33)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("FKK"),
                ["FKK_{:02d}".format(i) for i in range(1, 33)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("FKK"),
                None,
                {
                    "SK": [4, 8, 12, 16],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_fkk_raises(self, data, columns, subscales, expected):
        with expected:
            fkk(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("FKK"), None, None, result_filtered("FKK")),
            (
                data_filtered_correct("FKK"),
                ["FKK_{}".format(i) for i in range(1, 33)],
                None,
                result_filtered("FKK"),
            ),
            (convert_scale(data_filtered_wrong_range("FKK"), 1), None, None, result_filtered("FKK")),
            (
                data_subscale("fkk"),
                None,
                {
                    "SK": [1, 2, 3, 4, 5, 6, 7, 8],
                },
                result_filtered(regex="FKK_SK$"),
            ),
            (
                data_subscale("fkk"),
                None,
                {
                    "SK": [1, 2, 3, 4, 5, 6, 7, 8],
                    "I": [9, 10, 11, 12, 13, 14, 15, 16],
                },
                pd.concat(
                    [
                        result_filtered(regex="FKK_SK$"),
                        result_filtered(regex="FKK_I$"),
                        result_filtered(regex="FKK_SKI$"),
                    ],
                    axis=1,
                ),
            ),
            (
                data_filtered_correct("FKK"),
                None,
                {
                    "P": [3, 10, 14, 17, 19, 22, 26, 29],
                    "C": [2, 7, 9, 13, 15, 18, 21, 31],
                },
                pd.concat(
                    [
                        result_filtered(regex="FKK_P$"),
                        result_filtered(regex="FKK_C$"),
                        result_filtered(regex="FKK_PC$"),
                    ],
                    axis=1,
                ),
            ),
        ],
    )
    def test_fkk(self, data, columns, subscales, result):
        data_out = fkk(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("FSCRS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("FSCRS"), -1), None, None, does_not_raise()),
            (data_filtered_correct("FSCRS"), None, None, does_not_raise()),
            (
                data_filtered_correct("FSCRS"),
                ["FSCRS{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("FSCRS"),
                ["FSCRS_{}".format(i) for i in range(1, 23)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("FSCRS"),
                ["FSCRS{:02d}".format(i) for i in range(1, 23)],
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_fscrs_raises(self, data, columns, subscales, expected):
        with expected:
            fscrs(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("FSCRS"), None, None, result_filtered("FSCRS")),
            (
                data_filtered_correct("FSCRS"),
                ["FSCRS{:02d}".format(i) for i in range(1, 23)],
                None,
                result_filtered("FSCRS"),
            ),
            (convert_scale(data_filtered_wrong_range("FSCRS"), -1), None, None, result_filtered("FSCRS")),
            (
                data_subscale("fscrs"),
                None,
                {
                    "InadequateSelf": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                },
                result_filtered("FSCRS_InadequateSelf"),
            ),
        ],
    )
    def test_fscrs(self, data, columns, subscales, result):
        data_out = fscrs(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("GHQ"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("GHQ"), -1), None, does_not_raise()),
            (data_filtered_correct("GHQ"), None, does_not_raise()),
            (
                data_filtered_correct("GHQ"),
                ["GHQ{}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("GHQ"),
                ["GHQ_{}".format(i) for i in range(1, 13)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("GHQ"),
                ["GHQ{:02d}".format(i) for i in range(1, 13)],
                does_not_raise(),
            ),
        ],
    )
    def test_ghq_raises(self, data, columns, expected):
        with expected:
            ghq(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("GHQ"), None, result_filtered("GHQ")),
            (
                data_filtered_correct("GHQ"),
                ["GHQ{:02d}".format(i) for i in range(1, 13)],
                result_filtered("GHQ"),
            ),
            (convert_scale(data_filtered_wrong_range("GHQ"), -1), None, result_filtered("GHQ")),
        ],
    )
    def test_ghq(self, data, columns, result):
        data_out = ghq(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("HADS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("HADS"), -1), None, None, does_not_raise()),
            (data_filtered_correct("HADS"), None, None, does_not_raise()),
            (
                data_filtered_correct("HADS"),
                ["HADS_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("HADS"),
                ["HADS_{}".format(i) for i in range(1, 15)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("HADS"),
                ["HADS{:02d}".format(i) for i in range(1, 15)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("HADS"),
                None,
                {
                    "Depression": [2, 4, 6, 8, 10],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_hads_raises(self, data, columns, subscales, expected):
        with expected:
            hads(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("HADS"), None, None, result_filtered("HADS")),
            (
                data_filtered_correct("HADS"),
                ["HADS{:02d}".format(i) for i in range(1, 15)],
                None,
                result_filtered("HADS"),
            ),
            (convert_scale(data_filtered_wrong_range("HADS"), -1), None, None, result_filtered("HADS")),
            (
                data_subscale("hads"),
                None,
                {
                    "Depression": [1, 2, 3, 4, 5, 6, 7],
                },
                result_filtered("HADS_Depression"),
            ),
        ],
    )
    def test_hads(self, data, columns, subscales, result):
        data_out = hads(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("KKG"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("KKG"), 1), None, None, does_not_raise()),
            (data_filtered_correct("KKG"), None, None, does_not_raise()),
            (
                data_filtered_correct("KKG"),
                ["KKG_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("KKG"),
                ["KKG_{}".format(i) for i in range(1, 22)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("KKG"),
                ["KKG{:02d}".format(i) for i in range(1, 22)],
                None,
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_kkg_raises(self, data, columns, subscales, expected):
        with expected:
            kkg(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("KKG"), None, None, result_filtered("KKG")),
            (
                data_filtered_correct("KKG"),
                ["KKG_{}".format(i) for i in range(1, 22)],
                None,
                result_filtered("KKG"),
            ),
            (convert_scale(data_filtered_wrong_range("KKG"), 1), None, None, result_filtered("KKG")),
            (
                data_subscale("kkg"),
                None,
                {
                    "I": [1, 2, 3, 4, 5, 6, 7],
                },
                result_filtered("KKG_I"),
            ),
        ],
    )
    def test_kkg(self, data, columns, subscales, result):
        data_out = kkg(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("LSQ"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("LSQ"), -1), None, None, does_not_raise()),
            (data_filtered_correct("LSQ"), None, None, does_not_raise()),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}_{}".format(p, i) for p, i in product(["Partner", "Parents", "Child"], range(1, 11))],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parents", "Child"], range(1, 5))],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parents", "Child"], range(1, 11))],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{}".format(p, i) for p, i in product(["Partner", "Parents", "Child"], range(1, 11))],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_Partner{:02d}".format(i) for i in range(1, 11)],
                ["Partner"],
                does_not_raise(),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parents", "Child"], range(1, 11))],
                ["Partner", "Parents"],
                does_not_raise(),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_Partner{:02d}".format(i) for i in range(1, 11)],
                "Partner",
                does_not_raise(),
            ),
        ],
    )
    def test_lsq_raises(self, data, columns, subscales, expected):
        with expected:
            lsq(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("LSQ"), None, None, result_filtered("LSQ")),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parent", "Child"], range(1, 11))],
                None,
                result_filtered("LSQ"),
            ),
            (convert_scale(data_filtered_wrong_range("LSQ"), -1), None, None, result_filtered("LSQ")),
            (data_subscale("lsq"), None, "Partner", result_filtered("LSQ_Partner")),
        ],
    )
    def test_lsq(self, data, columns, subscales, result):
        data_out = lsq(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MBI_GS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MBI_GS"), -1), None, None, does_not_raise()),
            (data_filtered_correct("MBI_GS"), None, None, does_not_raise()),
            (
                data_filtered_correct("MBI_GS"),
                ["MBI_GS_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MBI_GS"),
                ["MBI_GS_{}".format(i) for i in range(1, 17)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("MBI_GS"),
                ["MBI_GS{:02d}".format(i) for i in range(1, 17)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MBI_GS"),
                None,
                {
                    "EE": [1, 2, 3, 4, 5],
                    "PA": [6, 7, 8, 11, 12, 16],
                    "DC": [9, 10, 13, 14, 15],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_mbi_gs_raises(self, data, columns, subscales, expected):
        with expected:
            mbi_gs(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("MBI_GS"), None, None, result_filtered(regex="MBI_GS_(EE|PA|DC)")),
            (
                data_filtered_correct("MBI_GS"),
                ["MBI_GS_{}".format(i) for i in range(1, 17)],
                None,
                result_filtered(regex="MBI_GS_(EE|PA|DC)"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MBI_GS"), -1),
                None,
                None,
                result_filtered(regex="MBI_GS_(EE|PA|DC)"),
            ),
            (data_subscale("mbi"), None, {"PA": [1, 2, 3, 4, 5, 6]}, result_filtered("MBI_GS_PA")),
        ],
    )
    def test_mbi_gs(self, data, columns, subscales, result):
        data_out = mbi_gs(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MBI_Students"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MBI_Students"), -1), None, None, does_not_raise()),
            (data_filtered_correct("MBI_Students"), None, None, does_not_raise()),
            (
                data_filtered_correct("MBI_Students"),
                ["MBI_Students_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MBI_Students"),
                ["MBI_Students_{}".format(i) for i in range(1, 17)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("MBI_Students"),
                ["MBI_Students{:02d}".format(i) for i in range(1, 17)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MBI_Students"),
                None,
                {
                    "EE": [1, 2, 3, 4, 5],
                    "PA": [6, 7, 8, 11, 12, 16],
                    "DC": [9, 10, 13, 14, 15],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_mbi_gss_raises(self, data, columns, subscales, expected):
        with expected:
            mbi_gss(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("MBI_Students"), None, None, result_filtered(regex="MBI_GSS_(EE|PA|DC)")),
            (
                data_filtered_correct("MBI_Students"),
                ["MBI_Students_{}".format(i) for i in range(1, 17)],
                None,
                result_filtered(regex="MBI_GSS_(EE|PA|DC)"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MBI_Students"), -1),
                None,
                None,
                result_filtered(regex="MBI_GSS_(EE|PA|DC)"),
            ),
            (data_subscale("mbi"), None, {"PA": [7, 8, 9, 10, 11, 12]}, result_filtered("MBI_GSS_PA")),
        ],
    )
    def test_mbi_gss(self, data, columns, subscales, result):
        data_out = mbi_gss(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MDBF"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MDBF"), 1), None, None, does_not_raise()),
            (data_filtered_correct("MDBF"), None, None, does_not_raise()),
            (
                data_filtered_correct("MDBF"),
                ["MDBF_{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MDBF"),
                ["MDBF_{:02d}".format(i) for i in range(1, 25)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("MDBF"),
                ["MDBF{}".format(i) for i in range(1, 25)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MDBF"),
                None,
                {
                    "GoodBad": [1, 4, 8, 11, 14, 16, 18, 21],
                    "AwakeTired": [2, 5, 7, 10, 13, 17, 20, 23],
                    "CalmNervous": [3, 6, 9, 12, 15, 19, 22, 24],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("MDBF"),
                None,
                {
                    "GoodBad": [1, 4, 8, 11, 14],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_mdbf_raises(self, data, columns, subscales, expected):
        with expected:
            mdbf(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("MDBF"), None, None, result_filtered("MDBF")),
            (
                data_filtered_correct("MDBF"),
                ["MDBF_{:02}".format(i) for i in range(1, 25)],
                None,
                result_filtered("MDBF"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MDBF"), 1),
                None,
                None,
                result_filtered("MDBF"),
            ),
            (data_subscale("mdbf"), None, {"GoodBad": [1, 2, 3, 4, 5, 6, 7, 8]}, result_filtered("MDBF_GoodBad")),
        ],
    )
    def test_mdbf(self, data, columns, subscales, result):
        data_out = mdbf(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MEQ"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MEQ"), 1), None, does_not_raise()),
            (data_filtered_correct("MEQ"), None, does_not_raise()),
            (
                data_filtered_correct("MEQ"),
                ["MEQ_{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MEQ"),
                ["MEQ_{:02d}".format(i) for i in range(1, 20)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("MEQ"),
                ["MEQ{}".format(i) for i in range(1, 20)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_meq_raises(self, data, columns, expected):
        with expected:
            meq(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("MEQ"), None, result_filtered(regex="(MEQ|Chronotype_*)")),
            (
                data_filtered_correct("MEQ"),
                ["MEQ_{:02}".format(i) for i in range(1, 20)],
                result_filtered(regex="(MEQ|Chronotype_*)"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MEQ"), 1),
                None,
                result_filtered(regex="(MEQ|Chronotype_*)"),
            ),
        ],
    )
    def test_meq(self, data, columns, result):
        data_out = meq(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range(regex=r"MIDI\d+"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range(regex=r"MIDI\d+"), 1), None, does_not_raise()),
            (data_filtered_correct(regex=r"MIDI\d+"), None, does_not_raise()),
            (
                data_filtered_correct(regex=r"MIDI\d+"),
                ["MIDI{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct(regex=r"MIDI\d+"),
                ["MIDI{:02d}".format(i) for i in range(1, 13)],
                does_not_raise(),
            ),
            (
                data_filtered_correct(regex=r"MIDI\d+"),
                ["MIDI_{}".format(i) for i in range(1, 13)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_midi_raises(self, data, columns, expected):
        with expected:
            midi(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct(regex=r"MIDI\d+"), None, result_filtered(regex="^MIDI$")),
            (
                data_filtered_correct(regex=r"MIDI\d+"),
                ["MIDI{:02d}".format(i) for i in range(1, 13)],
                result_filtered(regex="^MIDI$"),
            ),
            (
                convert_scale(data_filtered_wrong_range(regex=r"MIDI\d+"), 1),
                None,
                result_filtered(regex="^MIDI$"),
            ),
        ],
    )
    def test_midi(self, data, columns, result):
        data_out = midi(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MLQ"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MLQ"), 1), None, None, does_not_raise()),
            (data_filtered_correct("MLQ"), None, None, does_not_raise()),
            (
                data_filtered_correct("MLQ"),
                ["MLQ_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MLQ"),
                ["MLQ_{:02d}".format(i) for i in range(1, 11)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MLQ"),
                ["MLQ_{}".format(i) for i in range(1, 11)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("MLQ"),
                None,
                {
                    "PresenceMeaning": [1, 4, 5, 6, 9],
                    "SearchMeaning": [2, 3, 7, 8, 10],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("MLQ"),
                None,
                {
                    "PresenceMeaning": [1, 4, 5],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_mlq_raises(self, data, columns, subscales, expected):
        with expected:
            mlq(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("MLQ"), None, None, result_filtered("MLQ")),
            (
                data_filtered_correct("MLQ"),
                ["MLQ_{}".format(i) for i in range(1, 11)],
                None,
                result_filtered("MLQ"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MLQ"), 1),
                None,
                None,
                result_filtered("MLQ"),
            ),
            (data_subscale("mlq"), None, {"PresenceMeaning": [1, 2, 3, 4, 5]}, result_filtered("MLQ_PresenceMeaning")),
        ],
    )
    def test_mlq(self, data, columns, subscales, result):
        data_out = mlq(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MVES"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MVES"), -1), None, does_not_raise()),
            (data_filtered_correct("MVES"), None, does_not_raise()),
            (
                data_filtered_correct("MVES"),
                ["MVES{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MVES"),
                ["MVES{:02d}".format(i) for i in range(1, 24)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("MVES"),
                ["MVES_{}".format(i) for i in range(1, 24)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_mves_raises(self, data, columns, expected):
        with expected:
            mves(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("MVES"), None, result_filtered("MVES")),
            (
                data_filtered_correct("MVES"),
                ["MVES{:02d}".format(i) for i in range(1, 24)],
                result_filtered("MVES"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MVES"), -1),
                None,
                result_filtered("MVES"),
            ),
        ],
    )
    def test_mves(self, data, columns, result):
        data_out = mves(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, language, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("PANAS"), None, "german", pytest.raises(ValueRangeError)),
            (data_filtered_correct("PANAS"), None, "spanish", pytest.raises(ValueError)),
            (convert_scale(data_filtered_wrong_range("PANAS"), 1), None, "german", does_not_raise()),
            (data_filtered_correct("PANAS"), None, "german", does_not_raise()),
            (
                data_filtered_correct("PANAS"),
                ["PANAS_{}".format(i) for i in range(1, 10)],
                "german",
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PANAS"),
                ["PANAS_{}".format(i) for i in range(1, 10)],
                "german",
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PANAS"),
                ["PANAS_{}".format(i) for i in range(1, 21)],
                "german",
                does_not_raise(),
            ),
            (
                data_filtered_correct("PANAS"),
                ["PANAS_{}".format(i) for i in range(1, 21)],
                "english",
                does_not_raise(),
            ),
        ],
    )
    def test_panas_raises(self, data, columns, language, expected):
        with expected:
            panas(data, columns, language)

    @pytest.mark.parametrize(
        "data, columns, language, result",
        [
            (data_filtered_correct("PANAS"), None, "english", result_filtered("PANAS")),
            (data_filtered_correct("PANAS"), None, "german", panas_results_german()),
            (
                data_filtered_correct("PANAS"),
                ["PANAS_{}".format(i) for i in range(1, 21)],
                None,
                result_filtered("PANAS"),
            ),
            (convert_scale(data_filtered_wrong_range("PANAS"), 1), None, None, result_filtered("PANAS")),
        ],
    )
    def test_panas(self, data, columns, language, result):
        data_out = panas(data, columns, language=language)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("PASA"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("PASA"), 1), None, None, does_not_raise()),
            (data_filtered_correct("PASA"), None, None, does_not_raise()),
            (
                data_filtered_correct("PASA"),
                ["PASA_{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PASA"),
                ["PASA{:02d}".format(i) for i in range(1, 17)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PASA"),
                ["PASA_{:02d}".format(i) for i in range(1, 17)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("PASA"),
                None,
                {
                    "Threat": [1, 9, 5, 13],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("PASA"),
                None,
                {
                    "Threat": [1, 2],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_pasa_raises(self, data, columns, subscales, expected):
        with expected:
            pasa(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("PASA"), None, None, result_filtered("PASA")),
            (
                data_filtered_correct("PASA"),
                ["PASA_{:02d}".format(i) for i in range(1, 17)],
                None,
                result_filtered("PASA"),
            ),
            (
                convert_scale(data_filtered_wrong_range("PASA"), 1),
                None,
                None,
                result_filtered("PASA"),
            ),
            (
                data_subscale("pasa"),
                None,
                {"Threat": [1, 2, 3, 4]},
                result_filtered("PASA_Threat"),
            ),
            (
                data_filtered_correct("PASA"),
                None,
                {"Threat": [1, 5, 9, 13], "Challenge": [2, 6, 10, 14]},
                result_filtered(regex="PASA_(Threat|Challenge|Primary)"),
            ),
            (
                data_filtered_correct("PASA"),
                None,
                {
                    "Threat": [1, 5, 9, 13],
                    "SelfConcept": [3, 7, 11, 15],
                },
                result_filtered(regex="PASA_(Threat|SelfConcept)"),
            ),
            (
                data_filtered_correct("PASA"),
                None,
                {
                    "SelfConcept": [3, 7, 11, 15],
                    "ControlExp": [4, 8, 12, 16],
                },
                result_filtered(regex="PASA_(SelfConcept|ControlExp|Secondary)"),
            ),
        ],
    )
    def test_pasa(self, data, columns, subscales, result):
        data_out = pasa(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)
