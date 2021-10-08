from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Optional
from unittest import TestCase

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from biopsykit.questionnaires import *
from biopsykit.questionnaires.utils import convert_scale
from biopsykit.utils.exceptions import ValidationError, ValueRangeError

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
            (data_filtered_correct(regex=r"ABI\d"), None, result_filtered(regex="ABI_(VIG|KOV)")),
            (
                data_complete_correct(),
                ["ABI{}_{}".format(i, j) for i, j in product(range(1, 9), range(1, 11))],
                result_filtered(regex="ABI_(VIG|KOV)"),
            ),
            (convert_scale(data_filtered_wrong_range(regex=r"ABI\d"), 1), None, result_filtered(regex="ABI_(VIG|KOV)")),
        ],
    )
    def test_abi(self, data, columns, result):
        data_out = abi(data, columns)
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("ADSL"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("ADSL"), -1), None, does_not_raise()),
            (data_filtered_correct("ADSL"), None, does_not_raise()),
            (
                data_filtered_correct("ADSL"),
                ["ADSL{}".format(i) for i in range(1, 21)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ADSL"),
                ["ADSL{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ADSL"),
                ["ADSL_{}".format(i) for i in range(1, 21)],
                does_not_raise(),
            ),
        ],
    )
    def test_ads_l_raises(self, data, columns, expected):
        with expected:
            ads_l(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("ADSL"), None, result_filtered("ADS_L")),
            (
                data_filtered_correct("ADSL"),
                ["ADSL_{}".format(i) for i in range(1, 21)],
                result_filtered("ADS_L"),
            ),
            (convert_scale(data_filtered_wrong_range("ADSL"), -1), None, result_filtered("ADS_L")),
        ],
    )
    def test_ads_l(self, data, columns, result):
        data_out = ads_l(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range(regex=r"ASQ\d{2}"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range(regex=r"ASQ\d{2}"), -1), None, does_not_raise()),
            (data_filtered_correct(regex=r"ASQ\d{2}"), None, does_not_raise()),
            (
                data_filtered_correct(regex=r"ASQ\d{2}"),
                ["ASQ{:02d}".format(i) for i in range(1, 4)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct(regex=r"ASQ\d{2}"),
                ["ASQ{:02d}".format(i) for i in range(1, 5)],
                does_not_raise(),
            ),
            (
                data_filtered_correct(regex=r"ASQ\d{2}"),
                ["ASQ_{}".format(i) for i in range(1, 5)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_asq_raises(self, data, columns, expected):
        with expected:
            asq(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct(regex=r"ASQ\d{2}"), None, result_filtered(regex=r"ASQ$")),
            (
                data_filtered_correct(regex=r"ASQ\d{2}"),
                ["ASQ{:02d}".format(i) for i in range(1, 5)],
                result_filtered(regex=r"ASQ$"),
            ),
            (
                convert_scale(data_filtered_wrong_range(regex=r"ASQ\d{2}"), -1),
                None,
                result_filtered(regex=r"ASQ$"),
            ),
        ],
    )
    def test_asq(self, data, columns, result):
        data_out = asq(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("ASQ_MOD"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("ASQ_MOD"), -1), None, does_not_raise()),
            (data_filtered_correct("ASQ_MOD"), None, does_not_raise()),
            (
                data_filtered_correct("ASQ_MOD"),
                ["ASQ_MOD{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ASQ_MOD"),
                ["ASQ_MOD{:02d}".format(i) for i in range(1, 5)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("ASQ_MOD"),
                ["ASQ_MOD_{}".format(i) for i in range(1, 5)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_asq_mod_raises(self, data, columns, expected):
        with expected:
            asq_mod(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("ASQ_MOD"), None, result_filtered("ASQ_MOD")),
            (
                data_filtered_correct("ASQ_MOD"),
                ["ASQ_MOD{:02d}".format(i) for i in range(1, 5)],
                result_filtered("ASQ_MOD"),
            ),
            (
                convert_scale(data_filtered_wrong_range("ASQ_MOD"), -1),
                None,
                result_filtered("ASQ_MOD"),
            ),
        ],
    )
    def test_asq_mod(self, data, columns, result):
        data_out = asq_mod(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
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
            (data_filtered_wrong_range("IE4"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("IE4"), 1), None, None, does_not_raise()),
            (data_filtered_correct("IE4"), None, None, does_not_raise()),
            (
                data_filtered_correct("IE4"),
                ["IE4_{}".format(i) for i in range(1, 4)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("IE4"),
                ["IE4_{}".format(i) for i in range(1, 5)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("IE4"),
                ["IE4_{:02d}".format(i) for i in range(1, 5)],
                None,
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_ie_4_raises(self, data, columns, subscales, expected):
        with expected:
            ie_4(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("IE4"), None, None, result_filtered("IE4")),
            (
                data_filtered_correct("IE4"),
                ["IE4_{}".format(i) for i in range(1, 5)],
                None,
                result_filtered("IE4"),
            ),
            (convert_scale(data_filtered_wrong_range("IE4"), 1), None, None, result_filtered("IE4")),
            (
                data_subscale("ie4"),
                None,
                {
                    "Extern": [1, 2],
                },
                result_filtered("IE4_Extern"),
            ),
        ],
    )
    def test_ie4(self, data, columns, subscales, result):
        data_out = ie_4(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result, check_dtype=False)

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
                ["LSQ_{}_{}".format(p, i) for p, i in product(["Partner", "Parent", "Child"], range(1, 11))],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parent", "Child"], range(1, 5))],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parent", "Child"], range(1, 11))],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("LSQ"),
                ["LSQ_{}{}".format(p, i) for p, i in product(["Partner", "Parent", "Child"], range(1, 11))],
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
                ["LSQ_{}{:02d}".format(p, i) for p, i in product(["Partner", "Parent", "Child"], range(1, 11))],
                ["Partner", "Parent"],
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
        assert_frame_equal(data_out, result, check_dtype=False)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("PEAT"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("PEAT"), -1), None, does_not_raise()),
            (data_filtered_correct("PEAT"), None, does_not_raise()),
            (
                data_filtered_correct("PEAT"),
                ["PEAT{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PEAT"),
                ["PEAT{:02d}".format(i) for i in range(1, 11)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("PEAT"),
                ["PEAT_{}".format(i) for i in range(1, 11)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_peat_raises(self, data, columns, expected):
        with expected:
            peat(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("PEAT"), None, result_filtered("PEAT")),
            (
                data_filtered_correct("PEAT"),
                ["PEAT{:02d}".format(i) for i in range(1, 11)],
                result_filtered("PEAT"),
            ),
            (
                convert_scale(data_filtered_wrong_range("PEAT"), -1),
                None,
                result_filtered("PEAT"),
            ),
        ],
    )
    def test_peat(self, data, columns, result):
        data_out = peat(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("PFB"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("PFB"), 1), None, None, does_not_raise()),
            (data_filtered_correct("PFB"), None, None, does_not_raise()),
            (
                data_filtered_correct("PFB"),
                ["PFB{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PFB"),
                ["PFB{:02d}".format(i) for i in range(1, 32)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("PFB"),
                ["PFB_{}".format(i) for i in range(1, 32)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_wrong_range("PFB"),
                None,
                {
                    "Zaertlichkeit": [2, 3, 5, 9, 13, 14, 16, 23, 27, 28],
                },
                pytest.raises(ValueRangeError),
            ),
            (
                data_filtered_correct("PFB"),
                None,
                {
                    "Zaertlichkeit": [2, 3, 5, 9, 13, 14, 16, 23, 27, 28],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("PFB"),
                None,
                {
                    "Streitverhalten": [1, 6, 8, 17, 18],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_pfb_raises(self, data, columns, subscales, expected):
        with expected:
            pfb(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("PFB"), None, None, result_filtered("PFB")),
            (
                data_filtered_correct("PFB"),
                ["PFB{:02d}".format(i) for i in range(1, 32)],
                None,
                result_filtered("PFB"),
            ),
            (
                convert_scale(data_filtered_wrong_range("PFB"), 1),
                None,
                None,
                result_filtered("PFB"),
            ),
            (
                data_subscale("pfb"),
                None,
                {"Streitverhalten": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                result_filtered("PFB_Streitverhalten"),
            ),
        ],
    )
    def test_pfb(self, data, columns, subscales, result):
        data_out = pfb(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

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
            (data_filtered_wrong_range("PL"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("PL"), 1), None, does_not_raise()),
            (data_filtered_correct("PL"), None, does_not_raise()),
            (
                data_filtered_correct("PL"),
                ["PL{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PL"),
                ["PL{:02d}".format(i) for i in range(1, 11)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("PL"),
                ["PL_{}".format(i) for i in range(1, 11)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_purpose_life_raises(self, data, columns, expected):
        with expected:
            purpose_life(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("PL"), None, result_filtered("PurposeLife")),
            (
                data_filtered_correct("PL"),
                ["PL{:02d}".format(i) for i in range(1, 11)],
                result_filtered("PurposeLife"),
            ),
            (
                convert_scale(data_filtered_wrong_range("PL"), 1),
                None,
                result_filtered("PurposeLife"),
            ),
        ],
    )
    def test_purpose_life(self, data, columns, result):
        data_out = purpose_life(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("RMIDI"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("RMIDI"), 1), None, None, does_not_raise()),
            (data_filtered_correct("RMIDI"), None, None, does_not_raise()),
            (
                data_filtered_correct("RMIDI"),
                ["RMIDIPS{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RMIDI"),
                ["RMIDIPS{:02d}".format(i) for i in range(1, 32)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("RMIDI"),
                ["RMIDIPS_{}".format(i) for i in range(1, 32)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RMIDI"),
                None,
                {
                    "Conscientiousness": [4, 9, 16, 24, 31],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("RMIDI"),
                None,
                {
                    "Conscientiousness": [4, 9, 16],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_rmidi_raises(self, data, columns, subscales, expected):
        with expected:
            rmidi(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("RMIDI"), None, None, result_filtered("RMIDI")),
            (
                data_filtered_correct("RMIDI"),
                ["RMIDIPS{:02d}".format(i) for i in range(1, 32)],
                None,
                result_filtered("RMIDI"),
            ),
            (
                convert_scale(data_filtered_wrong_range("RMIDI"), 1),
                None,
                None,
                result_filtered("RMIDI"),
            ),
            (
                data_subscale("rmidi"),
                None,
                {"Conscientiousness": [1, 2, 3, 4, 5]},
                result_filtered("RMIDI_Conscientiousness"),
            ),
        ],
    )
    def test_rmidi(self, data, columns, subscales, result):
        data_out = rmidi(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("RSE"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("RSE"), -1), None, does_not_raise()),
            (data_filtered_correct("RSE"), None, does_not_raise()),
            (
                data_filtered_correct("RSE"),
                ["RSE{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RSE"),
                ["RSE{:02d}".format(i) for i in range(1, 11)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("RSE"),
                ["RSE_{}".format(i) for i in range(1, 11)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_rse_raises(self, data, columns, expected):
        with expected:
            rse(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("RSE"), None, result_filtered("RSE")),
            (
                data_filtered_correct("RSE"),
                ["RSE{:02d}".format(i) for i in range(1, 11)],
                result_filtered("RSE"),
            ),
            (
                convert_scale(data_filtered_wrong_range("RSE"), -1),
                None,
                result_filtered("RSE"),
            ),
        ],
    )
    def test_rse(self, data, columns, result):
        data_out = rse(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("RS_"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("RS_"), 1), None, does_not_raise()),
            (data_filtered_correct("RS_"), None, does_not_raise()),
            (
                data_filtered_correct("RS_"),
                ["RS_{:01d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RS_"),
                ["RS{}".format(i) for i in range(1, 12)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RS_"),
                ["RS_{:01d}".format(i) for i in range(1, 12)],
                does_not_raise(),
            ),
        ],
    )
    def test_resilience_raises(self, data, columns, expected):
        with expected:
            resilience(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct(regex=r"^RS_\d+"), None, result_filtered(regex=r"^RS$")),
            (
                data_filtered_correct(regex=r"^RS_\d+"),
                ["RS_{:01d}".format(i) for i in range(1, 12)],
                result_filtered(regex=r"^RS$"),
            ),
            (
                convert_scale(data_filtered_wrong_range(regex=r"^RS_\d+"), 1),
                None,
                result_filtered(regex=r"^RS$"),
            ),
        ],
    )
    def test_resilience(self, data, columns, result):
        data_out = resilience(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("RSQ"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("RSQ"), 1), None, None, does_not_raise()),
            (data_filtered_correct("RSQ"), None, None, does_not_raise()),
            (
                data_filtered_correct("RSQ"),
                ["RSQ_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RSQ"),
                ["RSQ{:02d}".format(i) for i in range(1, 33)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("RSQ"),
                ["RSQ_{}".format(i) for i in range(1, 33)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("RSQ"),
                None,
                {
                    "SymptomRumination": [2, 3, 4, 8, 11, 12, 13, 25],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_rsq_raises(self, data, columns, subscales, expected):
        with expected:
            rsq(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("RSQ"), None, None, result_filtered("RSQ")),
            (
                data_filtered_correct("RSQ"),
                ["RSQ_{}".format(i) for i in range(1, 33)],
                None,
                result_filtered("RSQ"),
            ),
            (
                convert_scale(data_filtered_wrong_range("RSQ"), 1),
                None,
                None,
                result_filtered("RSQ"),
            ),
            (
                data_subscale("rsq"),
                None,
                {"SymptomRumination": [1, 2, 3, 4, 5, 6, 7, 8]},
                result_filtered("RSQ_SymptomRumination"),
            ),
        ],
    )
    def test_rsq(self, data, columns, subscales, result):
        data_out = rsq(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SCS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SCS"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SCS"), None, None, does_not_raise()),
            (
                data_filtered_correct("SCS"),
                ["SCS{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SCS"), ["SCS{:02d}".format(i) for i in range(1, 27)], None, does_not_raise()),
            (
                data_filtered_correct("SCS"),
                ["SCS_{}".format(i) for i in range(1, 27)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SCS"),
                None,
                {
                    "SelfJudgment": [1, 8, 11, 16, 21],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("SCS"),
                None,
                {
                    "SelfJudgment": [1, 8, 11, 16],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_scs_raises(self, data, columns, subscales, expected):
        with expected:
            scs(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SCS"), None, None, result_filtered("SCS")),
            (
                data_filtered_correct("SCS"),
                ["SCS{:02d}".format(i) for i in range(1, 27)],
                None,
                result_filtered("SCS"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SCS"), 1),
                None,
                None,
                result_filtered("SCS"),
            ),
            (
                data_subscale("scs"),
                None,
                {"SelfKindness": [1, 2, 3, 4, 5]},
                result_filtered("SCS_SelfKindness"),
            ),
        ],
    )
    def test_scs(self, data, columns, subscales, result):
        data_out = scs(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SSGS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SSGS"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SSGS"), None, None, does_not_raise()),
            (
                data_filtered_correct("SSGS"),
                ["SSGS{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SSGS"), ["SSGS{:02d}".format(i) for i in range(1, 16)], None, does_not_raise()),
            (
                data_filtered_correct("SSGS"),
                ["SSGS_{}".format(i) for i in range(1, 16)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SSGS"),
                None,
                {
                    "Pride": [1, 4, 7, 10, 13],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_ssgs_raises(self, data, columns, subscales, expected):
        with expected:
            ssgs(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SSGS"), None, None, result_filtered("SSGS")),
            (
                data_filtered_correct("SSGS"),
                ["SSGS{:02d}".format(i) for i in range(1, 16)],
                None,
                result_filtered("SSGS"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SSGS"), 1),
                None,
                None,
                result_filtered("SSGS"),
            ),
            (
                data_subscale("ssgs"),
                None,
                {"Pride": [1, 2, 3, 4, 5]},
                result_filtered("SSGS_Pride"),
            ),
        ],
    )
    def test_ssgs(self, data, columns, subscales, result):
        data_out = ssgs(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SSS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SSS"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SSS"), None, None, does_not_raise()),
            (
                data_filtered_correct("SSS"),
                ["SSS_U_{}".format(i) for i in range(1, 2)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SSS"), ["SSS_U_{}".format(i) for i in range(1, 3)], None, does_not_raise()),
            (
                data_filtered_correct("SSS"),
                ["SSS_U{:02d}".format(i) for i in range(1, 3)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SSS"),
                None,
                {
                    "SocioeconomicStatus": [1],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_sss_raises(self, data, columns, subscales, expected):
        with expected:
            sss(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SSS"), None, None, result_filtered("SSS")),
            (
                data_filtered_correct("SSS"),
                ["SSS_U_{}".format(i) for i in range(1, 3)],
                None,
                result_filtered("SSS"),
            ),
            (
                data_subscale("sss"),
                None,
                {"Community": [1]},
                result_filtered("SSS_Community"),
            ),
        ],
    )
    def test_sss(self, data, columns, subscales, result):
        data_out = sss(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, stadi_type, expected",
        [
            (data_complete_correct(), None, None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("STADI"), None, None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("STADI"), 1), None, None, None, does_not_raise()),
            (data_filtered_correct("STADI"), None, None, None, does_not_raise()),
            (
                data_filtered_correct("STADI"),
                ["STADI_{}_{}".format(s, i) for s, i in product(["S", "T"], range(1, 10))],
                None,
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI{:02d}".format(i) for i in range(1, 21)],
                None,
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_{}_{}".format(s, i) for s, i in product(["S", "T"], range(1, 21))],
                None,
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_{}_{}".format(s, i) for s, i in product(["S", "T"], range(1, 21))],
                None,
                "state_trait",
                does_not_raise(),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_S_{}".format(i) for i in range(1, 21)],
                None,
                "state",
                does_not_raise(),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_T_{}".format(i) for i in range(1, 21)],
                None,
                "trait",
                does_not_raise(),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_{}_{}".format(s, i) for s, i in product(["S", "T"], range(1, 21))],
                None,
                "trate",
                pytest.raises(ValueError),
            ),
            (
                data_filtered_correct("STADI"),
                None,
                {
                    "AU": [1, 5, 9, 13, 17],
                },
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_stadi_raises(self, data, columns, subscales, stadi_type, expected):
        with expected:
            stadi(data, columns, subscales, stadi_type)

    @pytest.mark.parametrize(
        "data, columns, subscales, stadi_type, result",
        [
            (data_filtered_correct("STADI"), None, None, None, result_filtered("STADI")),
            (data_filtered_correct("STADI"), None, None, "state_trait", result_filtered("STADI")),
            (data_filtered_correct("STADI_S"), None, None, "state", result_filtered("STADI_State")),
            (data_filtered_correct("STADI_T"), None, None, "trait", result_filtered("STADI_Trait")),
            (
                data_filtered_correct("STADI"),
                ["STADI_{}_{}".format(s, i) for s, i in product(["S", "T"], range(1, 21))],
                None,
                None,
                result_filtered("STADI"),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_S_{}".format(i) for i in range(1, 21)],
                None,
                "state",
                result_filtered("STADI_State"),
            ),
            (
                data_filtered_correct("STADI"),
                ["STADI_T_{}".format(i) for i in range(1, 21)],
                None,
                "trait",
                result_filtered("STADI_Trait"),
            ),
            (
                convert_scale(data_filtered_wrong_range("STADI"), 1),
                None,
                None,
                None,
                result_filtered("STADI"),
            ),
            (
                data_subscale("stadi"),
                None,
                {"AU": [1, 2, 3, 4, 5]},
                None,
                result_filtered(regex="STADI_(State|Trait)_AU"),
            ),
            (
                data_filtered_correct("STADI"),
                None,
                {"AU": [1, 5, 9, 13, 17], "BE": [2, 6, 10, 14, 18]},
                None,
                result_filtered(regex="STADI_(State|Trait)_(AU|BE|Anxiety)"),
            ),
            (
                data_filtered_correct("STADI"),
                None,
                {
                    "EU": [3, 7, 11, 15, 19],
                    "DY": [4, 8, 12, 16, 20],
                },
                None,
                result_filtered(regex="STADI_(State|Trait)_(EU|DY|Depression)"),
            ),
            (
                data_filtered_correct("STADI"),
                None,
                {
                    "AU": [1, 5, 9, 13, 17],
                    "EU": [3, 7, 11, 15, 19],
                },
                None,
                result_filtered(regex="STADI_(State|Trait)_(AU|EU)"),
            ),
            (
                data_filtered_correct("STADI_T"),
                None,
                {"AU": [1, 5, 9, 13, 17], "BE": [2, 6, 10, 14, 18]},
                "trait",
                result_filtered(regex="STADI_Trait_(AU|BE|Anxiety)"),
            ),
        ],
    )
    def test_stadi(self, data, columns, subscales, stadi_type, result):
        data_out = stadi(data, columns, subscales, stadi_type)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("StateRumination"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("StateRumination"), -1), None, does_not_raise()),
            (data_filtered_correct("StateRumination"), None, does_not_raise()),
            (
                data_filtered_correct("StateRumination"),
                ["StateRumination{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("StateRumination"),
                ["StateRumination{:02d}".format(i) for i in range(1, 28)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("StateRumination"),
                ["StateRumination_{}".format(i) for i in range(1, 28)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_state_rumination_raises(self, data, columns, expected):
        with expected:
            state_rumination(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("StateRumination"), None, result_filtered("StateRumination")),
            (
                data_filtered_correct("StateRumination"),
                ["StateRumination{:02d}".format(i) for i in range(1, 28)],
                result_filtered("StateRumination"),
            ),
            (
                convert_scale(data_filtered_wrong_range("StateRumination"), -1),
                None,
                result_filtered("StateRumination"),
            ),
        ],
    )
    def test_state_rumination(self, data, columns, result):
        data_out = state_rumination(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SVF120"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SVF120"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SVF120"), None, None, does_not_raise()),
            (
                data_filtered_correct("SVF120"),
                ["SVF120_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SVF120"),
                ["SVF120{:03d}".format(i) for i in range(1, 121)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SVF120"),
                ["SVF120_{}".format(i) for i in range(1, 121)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("SVF120"),
                None,
                {"Bag": [10, 31, 50, 67, 88, 106]},
                does_not_raise(),
            ),
        ],
    )
    def test_svf_120_raises(self, data, columns, subscales, expected):
        with expected:
            svf_120(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SVF120"), None, None, result_filtered("SVF120")),
            (
                data_filtered_correct("SVF120"),
                ["SVF120_{}".format(i) for i in range(1, 121)],
                None,
                result_filtered("SVF120"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SVF120"), 1),
                None,
                None,
                result_filtered("SVF120"),
            ),
            (
                data_subscale("svf_120"),
                None,
                {
                    "Bag": [1, 3, 5, 7, 9, 11],
                    "Her": [2, 4, 6, 8, 10, 12],
                },
                result_filtered(regex="SVF120_(Bag|Her)"),
            ),
            (
                data_filtered_correct("SVF120"),
                None,
                {
                    "Bag": [10, 31, 50, 67, 88, 106],
                    "Her": [17, 38, 52, 77, 97, 113],
                    "Schab": [5, 30, 43, 65, 104, 119],
                },
                result_filtered(regex="SVF120_(Bag|Her|Schab|Pos1)"),
            ),
            (
                data_filtered_correct("SVF120"),
                None,
                {
                    "Bag": [10, 31, 50, 67, 88, 106],  # Bagatellisierung
                    "Her": [17, 38, 52, 77, 97, 113],  # Herunterspielen
                    "Schab": [5, 30, 43, 65, 104, 119],  # Schuldabwehr
                    "Abl": [1, 20, 45, 86, 101, 111],  # Ablenkung
                    "Ers": [22, 36, 64, 74, 80, 103],  # Ersatzbefriedigung
                    "Sebest": [34, 47, 59, 78, 95, 115],  # Selbstbestätigung
                    "Entsp": [12, 28, 58, 81, 99, 114],  # Entspannung
                    "Sitkon": [11, 18, 39, 66, 91, 116],  # Situationskontrolle
                    "Rekon": [2, 26, 54, 68, 85, 109],  # Reaktionskontrolle
                    "Posi": [15, 37, 56, 71, 83, 96],  # Positive Selbstinstruktion
                },
                result_filtered(
                    regex="SVF120_(Bag|Her|Schab|Abl|Ers|Sebest|Entsp|Sitkon|Rekon|Posi|Pos1|Pos2|Pos3|PosGesamt)"
                ),
            ),
            (
                data_filtered_correct("SVF120"),
                None,
                {
                    "Bag": [10, 31, 50, 67, 88, 106],  # Bagatellisierung
                    "Her": [17, 38, 52, 77, 97, 113],  # Herunterspielen
                    "Schab": [5, 30, 43, 65, 104, 119],  # Schuldabwehr
                    "Abl": [1, 20, 45, 86, 101, 111],  # Ablenkung
                    "Ers": [22, 36, 64, 74, 80, 103],  # Ersatzbefriedigung
                    "Sebest": [34, 47, 59, 78, 95, 115],  # Selbstbestätigung
                    "Entsp": [12, 28, 58, 81, 99, 114],  # Entspannung
                    "Sitkon": [11, 18, 39, 66, 91, 116],  # Situationskontrolle
                    "Rekon": [2, 26, 54, 68, 85, 109],  # Reaktionskontrolle
                },
                result_filtered(regex="SVF120_(Bag|Her|Schab|Abl|Ers|Sebest|Entsp|Sitkon|Rekon|Pos1|Pos2)"),
            ),
        ],
    )
    def test_svf_120(self, data, columns, subscales, result):
        data_out = svf_120(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SVF42"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SVF42"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SVF42"), None, None, does_not_raise()),
            (
                data_filtered_correct("SVF42"),
                ["SVF42_{}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SVF42"),
                ["SVF42{:03d}".format(i) for i in range(1, 43)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SVF42"),
                ["SVF42_{}".format(i) for i in range(1, 43)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("SVF42"),
                None,
                {"Bag": [7, 22]},
                does_not_raise(),
            ),
        ],
    )
    def test_svf_42_raises(self, data, columns, subscales, expected):
        with expected:
            svf_42(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SVF42"), None, None, result_filtered("SVF42")),
            (
                data_filtered_correct("SVF42"),
                ["SVF42_{}".format(i) for i in range(1, 43)],
                None,
                result_filtered("SVF42"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SVF42"), 1),
                None,
                None,
                result_filtered("SVF42"),
            ),
            (
                data_subscale("svf_42"),
                None,
                {
                    "Bag": [1, 3],  # Bagatellisierung
                    "Her": [2, 4],  # Herunterspielen
                },
                result_filtered(regex="SVF42_(Bag|Her)"),
            ),
            (
                data_filtered_correct("SVF42"),
                None,
                {
                    "Bag": [7, 22],  # Bagatellisierung
                    "Her": [11, 35],  # Herunterspielen
                    "Schab": [2, 34],  # Schuldabwehr
                },
                result_filtered(regex="SVF42_(Bag|Her|Schab)"),
            ),
            (
                data_filtered_correct("SVF42"),
                None,
                {
                    "Verm": [6, 30],  # Vermeidung
                    "Flu": [16, 40],  # Flucht
                    "Soza": [20, 29],  # Soziale Abkapselung
                },
                result_filtered(regex="SVF42_(Verm|Flu|Soza|Denial)"),
            ),
            (
                data_filtered_correct("SVF42"),
                None,
                {
                    "Ers": [12, 42],  # Ersatzbefriedigung
                    "Entsp": [13, 26],  # Entspannung
                    "Sozube": [14, 27],  # Soziales Unterstützungsbedürfnis
                },
                result_filtered(regex="SVF42_(Ers|Entsp|Sozube|Distraction)"),
            ),
            (
                data_filtered_correct("SVF42"),
                None,
                {
                    "Bag": [7, 22],  # Bagatellisierung
                    "Her": [11, 35],  # Herunterspielen
                    "Posi": [9, 24],  # Positive Selbstinstruktion
                },
                result_filtered(regex="SVF42_(Bag|Her|Posi|Stressordevaluation)"),
            ),
            (
                data_filtered_correct("SVF42"),
                None,
                {
                    "Verm": [6, 30],  # Vermeidung
                    "Flu": [16, 40],  # Flucht
                },
                result_filtered(regex="SVF42_(Verm|Flu)"),
            ),
        ],
    )
    def test_svf_42(self, data, columns, subscales, result):
        data_out = svf_42(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("TICS_L"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("TICS_L"), -1), None, None, does_not_raise()),
            (data_filtered_correct("TICS_L"), None, None, does_not_raise()),
            (
                data_filtered_correct("TICS_L"),
                ["TICS_L_{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TICS_L"),
                ["TICS_L_{}".format(i) for i in range(1, 58)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TICS_L"),
                ["TICS_L_{:02d}".format(i) for i in range(1, 58)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("TICS_L"),
                None,
                {"WorkOverload": [50, 38, 44, 54, 17, 4, 27, 1]},
                does_not_raise(),
            ),
        ],
    )
    def test_tics_l_raises(self, data, columns, subscales, expected):
        with expected:
            tics_l(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("TICS_L"), None, None, result_filtered("TICS_L")),
            (
                data_filtered_correct("TICS_L"),
                ["TICS_L_{:02d}".format(i) for i in range(1, 58)],
                None,
                result_filtered("TICS_L"),
            ),
            (
                convert_scale(data_filtered_wrong_range("TICS_L"), -1),
                None,
                None,
                result_filtered("TICS_L"),
            ),
            (
                data_filtered_correct("TICS_L"),
                None,
                {
                    "WorkOverload": [1, 4, 17, 27, 38, 44, 50, 54],  # Arbeitsüberlastung
                    "SocialOverload": [7, 19, 28, 39, 49, 57],  # Soziale Überlastung
                    "PressureToPerform": [8, 12, 14, 22, 23, 30, 32, 43, 40],  # Erfolgsdruck
                    "WorkDiscontent": [5, 10, 13, 21, 37, 41, 48, 53],  # Unzufriedenheit mit der Arbeit
                    "DemandsWork": [3, 20, 24, 35, 47, 55],  # Überforderung bei der Arbeit
                    "LackSocialRec": [2, 18, 31, 46],  # Mangel an sozialer Anerkennung
                    "SocialTension": [6, 15, 26, 33, 45, 52],  # Soziale Spannungen
                    "SocialIsolation": [11, 29, 34, 42, 51, 56],  # Soziale Isolation
                    "ChronicWorry": [9, 16, 25, 36],  # Chronische Besorgnis
                },
                result_filtered(regex="TICS_L"),
            ),
            (
                data_subscale("tics_l"),
                None,
                {"WorkOverload": [1, 2, 3, 4, 5, 6, 7, 8]},
                result_filtered(regex="TICS_L_WorkOverload"),
            ),
        ],
    )
    def test_tics_l(self, data, columns, subscales, result):
        data_out = tics_l(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("TICS_S"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("TICS_S"), -1), None, None, does_not_raise()),
            (data_filtered_correct("TICS_S"), None, None, does_not_raise()),
            (
                data_filtered_correct("TICS_S"),
                ["TICS_S_{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TICS_S"),
                ["TICS_S_{}".format(i) for i in range(1, 31)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TICS_S"),
                ["TICS_S_{:02d}".format(i) for i in range(1, 31)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("TICS_S"),
                None,
                {"WorkOverload": [1, 3, 21]},
                does_not_raise(),
            ),
        ],
    )
    def test_tics_s_raises(self, data, columns, subscales, expected):
        with expected:
            tics_s(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("TICS_S"), None, None, result_filtered("TICS_S")),
            (
                data_filtered_correct("TICS_S"),
                ["TICS_S_{:02d}".format(i) for i in range(1, 31)],
                None,
                result_filtered("TICS_S"),
            ),
            (
                convert_scale(data_filtered_wrong_range("TICS_S"), -1),
                None,
                None,
                result_filtered("TICS_S"),
            ),
            (
                data_filtered_correct("TICS_S"),
                None,
                {
                    "WorkOverload": [1, 3, 21],
                    "SocialOverload": [11, 18, 28],
                    "PressureToPerform": [5, 14, 29],
                    "WorkDiscontent": [8, 13, 24],
                    "DemandsWork": [12, 16, 27],
                    "PressureSocial": [6, 15, 22],
                    "LackSocialRec": [2, 20, 23],
                    "SocialTension": [4, 9, 26],
                    "SocialIsolation": [19, 25, 30],
                    "ChronicWorry": [7, 10, 17],
                },
                result_filtered(regex="TICS_S"),
            ),
            (
                data_subscale("tics_s"),
                None,
                {"WorkOverload": [1, 2, 3]},
                result_filtered(regex="TICS_S_WorkOverload"),
            ),
        ],
    )
    def test_tics_s(self, data, columns, subscales, result):
        data_out = tics_s(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("TB"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("TB"), 1), None, None, does_not_raise()),
            (data_filtered_correct("TB"), None, None, does_not_raise()),
            (
                data_filtered_correct("TB"),
                ["TB_{}".format(i) for i in range(1, 11)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TB"),
                ["TB{:02d}".format(i) for i in range(1, 13)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TB"),
                ["TB_{}".format(i) for i in range(1, 13)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("TB"),
                None,
                {"TechComp": [5, 6, 7, 8]},
                does_not_raise(),
            ),
        ],
    )
    def test_tb_raises(self, data, columns, subscales, expected):
        with expected:
            tb(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("TB"), None, None, result_filtered("TB")),
            (
                data_filtered_correct("TB"),
                ["TB_{}".format(i) for i in range(1, 13)],
                None,
                result_filtered("TB"),
            ),
            (
                convert_scale(data_filtered_wrong_range("TB"), 1),
                None,
                None,
                result_filtered("TB"),
            ),
            (
                data_filtered_correct("TB"),
                None,
                {"TechAcc": [1, 2, 3, 4], "TechComp": [5, 6, 7, 8], "TechControl": [9, 10, 11, 12]},
                result_filtered("TB"),
            ),
            (
                data_subscale("tb"),
                None,
                {"TechComp": [1, 2, 3, 4]},
                result_filtered(regex="TB_TechComp"),
            ),
        ],
    )
    def test_tb(self, data, columns, subscales, result):
        data_out = tb(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("TraitRumination"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("TraitRumination"), -1), None, does_not_raise()),
            (data_filtered_correct("TraitRumination"), None, does_not_raise()),
            (
                data_filtered_correct("TraitRumination"),
                ["TraitRumination{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TraitRumination"),
                ["TraitRumination{:02d}".format(i) for i in range(1, 15)],
                does_not_raise(),
            ),
            (
                data_filtered_correct("TraitRumination"),
                ["TraitRumination_{}".format(i) for i in range(1, 15)],
                pytest.raises(ValidationError),
            ),
        ],
    )
    def test_trait_rumination_raises(self, data, columns, expected):
        with expected:
            trait_rumination(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("TraitRumination"), None, result_filtered("TraitRumination")),
            (
                data_filtered_correct("TraitRumination"),
                ["TraitRumination{:02d}".format(i) for i in range(1, 15)],
                result_filtered("TraitRumination"),
            ),
            (
                convert_scale(data_filtered_wrong_range("TraitRumination"), -1),
                None,
                result_filtered("TraitRumination"),
            ),
        ],
    )
    def test_trait_rumination(self, data, columns, result):
        data_out = trait_rumination(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("TSGS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("TSGS"), 1), None, None, does_not_raise()),
            (data_filtered_correct("TSGS"), None, None, does_not_raise()),
            (
                data_filtered_correct("TSGS"),
                ["TSGS{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("TSGS"), ["TSGS{:02d}".format(i) for i in range(1, 16)], None, does_not_raise()),
            (
                data_filtered_correct("TSGS"),
                ["TSGS_{}".format(i) for i in range(1, 16)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TSGS"),
                None,
                {
                    "Pride": [1, 4, 7, 10, 13],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_tsgs_raises(self, data, columns, subscales, expected):
        with expected:
            tsgs(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("TSGS"), None, None, result_filtered("TSGS")),
            (
                data_filtered_correct("TSGS"),
                ["TSGS{:02d}".format(i) for i in range(1, 16)],
                None,
                result_filtered("TSGS"),
            ),
            (
                convert_scale(data_filtered_wrong_range("TSGS"), 1),
                None,
                None,
                result_filtered("TSGS"),
            ),
            (
                data_subscale("tsgs"),
                None,
                {"Pride": [1, 2, 3, 4, 5]},
                result_filtered("TSGS_Pride"),
            ),
        ],
    )
    def test_tsgs(self, data, columns, subscales, result):
        data_out = tsgs(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("Type_D"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("Type_D"), -1), None, None, does_not_raise()),
            (data_filtered_correct("Type_D"), None, None, does_not_raise()),
            (
                data_filtered_correct("Type_D"),
                ["Type_D{:02d}".format(i) for i in range(1, 10)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("Type_D"), ["Type_D{:02d}".format(i) for i in range(1, 15)], None, does_not_raise()),
            (
                data_filtered_correct("Type_D"),
                ["Type_D_{}".format(i) for i in range(1, 15)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("Type_D"),
                None,
                {
                    "SocialInhibition": [1, 3, 6, 8, 10, 11, 14],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("Type_D"),
                None,
                {
                    "SocialInhibition": [1],
                },
                pytest.raises(IndexError),
            ),
        ],
    )
    def test_type_d_scale_raises(self, data, columns, subscales, expected):
        with expected:
            type_d(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("Type_D"), None, None, result_filtered("Type_D")),
            (
                data_filtered_correct("Type_D"),
                ["Type_D{:02d}".format(i) for i in range(1, 15)],
                None,
                result_filtered("Type_D"),
            ),
            (
                convert_scale(data_filtered_wrong_range("Type_D"), -1),
                None,
                None,
                result_filtered("Type_D"),
            ),
            (
                data_subscale("type_d"),
                None,
                {"SocialInhibition": [1, 2, 3, 4, 5, 6, 7]},
                result_filtered("Type_D_SocialInhibition"),
            ),
        ],
    )
    def test_type_d_scale(self, data, columns, subscales, result):
        data_out = type_d(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("KAB"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("KAB"), 1), None, does_not_raise()),
            (data_filtered_correct("KAB"), None, does_not_raise()),
            (
                data_filtered_correct("KAB"),
                ["T0_KAB_{:01d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("KAB"), ["T0_KAB_{:01d}".format(i) for i in range(1, 7)], does_not_raise()),
            (
                data_filtered_correct("KAB"),
                ["T0_KAB_{:02d}".format(i) for i in range(1, 7)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("KAB"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_kab_raises(self, data, columns, expected):
        with expected:
            kab(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("KAB"), None, result_filtered("KAB")),
            (
                data_filtered_correct("KAB"),
                ["T0_KAB_{:01d}".format(i) for i in range(1, 7)],
                result_filtered("KAB"),
            ),
            (
                convert_scale(data_filtered_wrong_range("KAB"), 1),
                None,
                result_filtered("KAB"),
            ),
        ],
    )
    def test_kab(self, data, columns, result):
        data_out = kab(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, stai_type, expected",
        [
            (data_complete_correct(), None, ["state"], pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SAI"), None, ["state"], pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SAI"), 1), None, ["state"], does_not_raise()),
            (data_filtered_correct("SAI"), None, ["state"], does_not_raise()),
            (
                data_filtered_correct("SAI"),
                ["SAI_{:01d}".format(i) for i in range(1, 15)],
                ["state"],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SAI"),
                ["SAI_{:01d}".format(i) for i in range(1, 11)],
                ["state"],
                does_not_raise(),
            ),
            (
                data_filtered_correct("SAI"),
                ["SAI_{:02d}".format(i) for i in range(1, 10)],
                ["state"],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SAI"),
                None,
                ["state"],
                does_not_raise(),
            ),
        ],
    )
    def test_sai_raises(self, data, columns, stai_type, expected):
        with expected:
            stai_short(data, columns, stai_type)

    @pytest.mark.parametrize(
        "data, columns, result, stai_type",
        [
            (data_filtered_correct("SAI"), None, result_filtered("SAI"), ["state"]),
            (
                data_filtered_correct("SAI"),
                ["SAI_{:01d}".format(i) for i in range(1, 11)],
                result_filtered("SAI"),
                ["state"],
            ),
            (
                convert_scale(data_filtered_wrong_range("SAI"), 1),
                None,
                result_filtered("SAI"),
                ["state"],
            ),
        ],
    )
    def test_sai(self, data, columns, result, stai_type):
        data_out = stai_short(data, columns, stai_type)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, stai_type, expected",
        [
            (data_complete_correct(), None, ["trait"], pytest.raises(ValidationError)),
            (data_filtered_wrong_range("TAI"), None, ["trait"], pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("TAI"), 1), None, ["trait"], does_not_raise()),
            (data_filtered_correct("TAI"), None, ["trait"], does_not_raise()),
            (
                data_filtered_correct("TAI"),
                ["TAI_{:01d}".format(i) for i in range(1, 15)],
                ["trait"],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TAI"),
                ["TAI_{:01d}".format(i) for i in range(1, 11)],
                ["trait"],
                does_not_raise(),
            ),
            (
                data_filtered_correct("TAI"),
                ["TAI_{:02d}".format(i) for i in range(1, 10)],
                ["trait"],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("TAI"),
                None,
                ["trait"],
                does_not_raise(),
            ),
        ],
    )
    def test_tai_raises(self, data, columns, stai_type, expected):
        with expected:
            stai_short(data, columns, stai_type)

    @pytest.mark.parametrize(
        "data, columns, result, stai_type",
        [
            (data_filtered_correct("TAI"), None, result_filtered("TAI"), ["trait"]),
            (
                data_filtered_correct("TAI"),
                ["TAI_{:01d}".format(i) for i in range(1, 11)],
                result_filtered("TAI"),
                ["trait"],
            ),
            (
                convert_scale(data_filtered_wrong_range("TAI"), 1),
                None,
                result_filtered("TAI"),
                ["trait"],
            ),
        ],
    )
    def test_tai(self, data, columns, result, stai_type):
        data_out = stai_short(data, columns, stai_type)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("CLQ"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("CLQ"), 1), None, None, does_not_raise()),
            (data_filtered_correct("CLQ"), None, None, does_not_raise()),
            (
                data_filtered_correct("CLQ"),
                ["CLQ_{:01d}".format(i) for i in range(1, 28)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("CLQ"), ["CLQ_{:01d}".format(i) for i in range(1, 27)], None, does_not_raise()),
            (
                data_filtered_correct("CLQ"),
                ["CLQ_{:02d}".format(i) for i in range(1, 27)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("CLQ"),
                None,
                {
                    "SS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_clq_raises(self, data, columns, subscales, expected):
        with expected:
            clq(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("CLQ"), None, None, result_filtered("CLQ")),
            (
                data_filtered_correct("CLQ"),
                ["CLQ_{:01d}".format(i) for i in range(1, 27)],
                None,
                result_filtered("CLQ"),
            ),
            (
                convert_scale(data_filtered_wrong_range("CLQ"), 1),
                None,
                None,
                result_filtered("CLQ"),
            ),
            (
                data_subscale("clq"),
                None,
                {"SS": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]},
                result_filtered("CLQ_SS"),
            ),
        ],
    )
    def test_clq(self, data, columns, subscales, result):
        data_out = clq(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SOP"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SOP"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SOP"), None, None, does_not_raise()),
            (
                data_filtered_correct("SOP"),
                ["SOP_{:01d}".format(i) for i in range(1, 11)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SOP"), ["SOP_{:01d}".format(i) for i in range(1, 10)], None, does_not_raise()),
            (
                data_filtered_correct("SOP"),
                ["SOP_{:02d}".format(i) for i in range(1, 11)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SOP"),
                None,
                {
                    "SW": [1, 3, 5, 7, 8],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_sop_raises(self, data, columns, subscales, expected):
        with expected:
            sop(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SOP"), None, None, result_filtered("SOP")),
            (
                data_filtered_correct("SOP"),
                ["SOP_{:01d}".format(i) for i in range(1, 10)],
                None,
                result_filtered("SOP"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SOP"), 1),
                None,
                None,
                result_filtered("SOP"),
            ),
            (
                data_subscale("sop"),
                None,
                {"SE": [1, 2, 3, 4, 5]},
                result_filtered("SOP_SE"),
            ),
        ],
    )
    def test_sop(self, data, columns, subscales, result):
        data_out = sop(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result, check_dtype=False)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("BFI10"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("BFI10"), 1), None, None, does_not_raise()),
            (data_filtered_correct("BFI10"), None, None, does_not_raise()),
            (
                data_filtered_correct("BFI10"),
                ["BFI10_{:01d}".format(i) for i in range(1, 12)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("BFI10"),
                ["BFI10_{}{}".format(l, i) for l in ["N", "O", "E", "V", "G"] for i in range(1, 3)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("BFI10"),
                ["BFI10_{:02d}".format(i) for i in range(1, 11)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("BFI10"),
                None,
                {
                    "E": [1, 2],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_bfi_10_raises(self, data, columns, subscales, expected):
        with expected:
            bfi_10(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("BFI10"), None, None, result_filtered("BFI_10")),
            (
                data_filtered_correct("BFI10"),
                ["BFI10_{}{}".format(l, i) for i in range(1, 3) for l in ["E", "V", "G", "N", "O"]],
                None,
                result_filtered("BFI_10"),
            ),
            (
                convert_scale(data_filtered_wrong_range("BFI10"), 1),
                None,
                None,
                result_filtered("BFI_10"),
            ),
            (
                data_subscale("bfi10"),
                None,
                {"E": [1, 2]},
                result_filtered("BFI_10_E"),
            ),
        ],
    )
    def test_bfi_10(self, data, columns, subscales, result):
        data_out = bfi_10(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("MKHAI"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("MKHAI"), -1), None, does_not_raise()),
            (data_filtered_correct("MKHAI"), None, does_not_raise()),
            (
                data_filtered_correct("MKHAI"),
                ["MKAHI_{:01d}".format(i) for i in range(1, 16)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("MKHAI"), ["MKHAI_{:01d}".format(i) for i in range(1, 15)], does_not_raise()),
            (
                data_filtered_correct("MKHAI"),
                ["MKHAI_{:02d}".format(i) for i in range(1, 15)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("MKHAI"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_mkhai_raises(self, data, columns, expected):
        with expected:
            mkhai(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("MKHAI"), None, result_filtered("MKHAI")),
            (
                data_filtered_correct("MKHAI"),
                ["MKHAI_{:01d}".format(i) for i in range(1, 15)],
                result_filtered("MKHAI"),
            ),
            (
                convert_scale(data_filtered_wrong_range("MKHAI"), -1),
                None,
                result_filtered("MKHAI"),
            ),
        ],
    )
    def test_mkhai(self, data, columns, result):
        data_out = mkhai(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SWB"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SWB"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SWB"), None, None, does_not_raise()),
            (
                data_filtered_correct("SWB"),
                ["SWB_{:01d}".format(i) for i in range(1, 15)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SWB"), ["SWB_{:01d}".format(i) for i in range(1, 14)], None, does_not_raise()),
            (
                data_filtered_correct("SWB"),
                ["SWB_{:02d}".format(i) for i in range(1, 14)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SWB"),
                None,
                {
                    "SN": [2, 5, 8, 10, 11, 13],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_swb_raises(self, data, columns, subscales, expected):
        with expected:
            swb(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, invert_score, result",
        [
            (data_filtered_correct("SWB"), None, None, False, result_filtered(regex="SWB_(SN|ALZ)$")),
            (
                data_filtered_correct("SWB"),
                ["SWB_{:01d}".format(i) for i in range(1, 14)],
                None,
                False,
                result_filtered(regex="SWB_(SN|ALZ)$"),
            ),
            (
                data_filtered_correct("SWB"),
                ["SWB_{:01d}".format(i) for i in range(1, 14)],
                None,
                True,
                result_filtered(regex=r"SWB_inv_(SN|ALZ)$"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SWB"), 1),
                None,
                None,
                False,
                result_filtered(regex="SWB_(SN|ALZ)$"),
            ),
            (
                data_subscale("swb"),
                None,
                {"SN": [1, 2, 3, 4, 5, 6]},
                False,
                result_filtered("SWB_SN"),
            ),
        ],
    )
    def test_swb(self, data, columns, subscales, invert_score, result):
        data_out = swb(data, columns, subscales, invert_score)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("ABIMS"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("ABIMS"), 1), None, None, does_not_raise()),
            (data_filtered_correct("ABIMS"), None, None, does_not_raise()),
            (
                data_filtered_correct("ABIMS"),
                ["ABIMS_{:01d}_{:01d}".format(i, j) for i in range(1, 6) for j in range(1, 9)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ABIMS"),
                ["ABIMS_{:01d}_{:01d}".format(i, j) for i in range(1, 5) for j in range(1, 9)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("ABIMS"),
                ["ABIMS_{:02d}_{:01d}".format(i, j) for i in range(1, 6) for j in range(1, 9)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ABIMS"),
                None,
                {
                    "KOV1": [1, 4, 5, 7],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_abi_ms_raises(self, data, columns, subscales, expected):
        with expected:
            abi_ms(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("ABIMS"), None, None, result_filtered("ABI_MS")),
            (
                data_filtered_correct("ABIMS"),
                ["ABIMS_{:01d}_{:01d}".format(i, j) for i in range(1, 5) for j in range(1, 9)],
                None,
                result_filtered("ABI_MS"),
            ),
            (
                convert_scale(data_filtered_wrong_range("ABIMS"), 1),
                None,
                None,
                result_filtered("ABI_MS"),
            ),
            (
                data_subscale("abims"),
                None,
                {"KOV1": [1, 2, 3, 4]},
                result_filtered("ABI_MS_KOV1"),
            ),
        ],
    )
    def test_abi_ms(self, data, columns, subscales, result):
        data_out = abi_ms(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("ASI"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("ASI"), -1), None, None, does_not_raise()),
            (data_filtered_correct("ASI"), None, None, does_not_raise()),
            (
                data_filtered_correct("ASI"),
                ["ASI_{:01d}".format(i) for i in range(1, 14)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("ASI"), ["ASI_{:01d}".format(i) for i in range(1, 13)], None, does_not_raise()),
            (
                data_filtered_correct("ASI"),
                ["ASI_{:02d}".format(i) for i in range(1, 13)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ASI"),
                None,
                {
                    "BSM": [3, 4, 7, 8, 12],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_asi_raises(self, data, columns, subscales, expected):
        with expected:
            asi(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("ASI"), None, None, result_filtered("ASI")),
            (
                data_filtered_correct("ASI"),
                ["ASI_{:01d}".format(i) for i in range(1, 13)],
                None,
                result_filtered("ASI"),
            ),
            (
                convert_scale(data_filtered_wrong_range("ASI"), -1),
                None,
                None,
                result_filtered("ASI"),
            ),
            (
                data_subscale("asi"),
                None,
                {"BSM": [1, 2, 3, 4, 5]},
                result_filtered("ASI_BSM"),
            ),
        ],
    )
    def test_asi(self, data, columns, subscales, result):
        data_out = asi(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SCI"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SCI"), 1), None, None, does_not_raise()),
            (data_filtered_correct("SCI"), None, None, does_not_raise()),
            (
                data_filtered_correct("SCI"),
                ["SCI_{:01d}_{:1d}".format(i, j) for i in range(1, 6) for j in range(1, 8)]
                + ["SCI_4_{:01d}".format(i) for i in range(8, 14)]
                + ["SCI_5_{:01d}".format(i) for i in range(1, 21)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SCI"),
                ["SCI_{:01d}_{:1d}".format(i, j) for i in range(1, 5) for j in range(1, 8)]
                + ["SCI_4_{:01d}".format(i) for i in range(8, 14)]
                + ["SCI_5_{:01d}".format(i) for i in range(1, 21)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("SCI"),
                ["SCI_{:01d}_{:02d}".format(i, j) for i in range(1, 5) for j in range(1, 8)]
                + ["SCI_4_{:01d}".format(i) for i in range(8, 14)]
                + ["SCI_5_{:01d}".format(i) for i in range(1, 21)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SCI"),
                None,
                {
                    "Stress2": [1, 2, 3, 4, 5, 6, 7],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_sci_raises(self, data, columns, subscales, expected):
        with expected:
            sci(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("SCI"), None, None, result_filtered("SCI")),
            (
                data_filtered_correct("SCI"),
                ["SCI_{:01d}_{:1d}".format(i, j) for i in range(1, 5) for j in range(1, 8)]
                + ["SCI_4_{:01d}".format(i) for i in range(8, 14)]
                + ["SCI_5_{:01d}".format(i) for i in range(1, 21)],
                None,
                result_filtered("SCI"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SCI"), 1),
                None,
                None,
                result_filtered("SCI"),
            ),
            (
                data_subscale("sci"),
                None,
                {"Stress2": [1, 2, 3, 4, 5, 6, 7]},
                result_filtered("SCI_Stress2"),
            ),
        ],
    )
    def test_sci(self, data, columns, subscales, result):
        data_out = sci(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("ERQ"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("ERQ"), 1), None, None, does_not_raise()),
            (data_filtered_correct("ERQ"), None, None, does_not_raise()),
            (
                data_filtered_correct("ERQ"),
                ["ERQ_{:01d}".format(i) for i in range(1, 12)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ERQ"),
                ["ERQ_{:01d}".format(i) for i in range(1, 11)],
                None,
                does_not_raise(),
            ),
            (
                data_filtered_correct("ERQ"),
                ["ERQ_{:02d}".format(i) for i in range(1, 11)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ERQ"),
                None,
                {
                    "Suppr": [2, 4, 6, 9],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_erq_raises(self, data, columns, subscales, expected):
        with expected:
            erq(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("ERQ"), None, None, result_filtered("ERQ")),
            (
                data_filtered_correct("ERQ"),
                ["ERQ_{:01d}".format(i) for i in range(1, 11)],
                None,
                result_filtered("ERQ"),
            ),
            (
                convert_scale(data_filtered_wrong_range("ERQ"), 1),
                None,
                None,
                result_filtered("ERQ"),
            ),
            (
                data_subscale("erq"),
                None,
                {"Suppr": [1, 2, 3, 4]},
                result_filtered("ERQ_Suppr"),
            ),
        ],
    )
    def test_erq(self, data, columns, subscales, result):
        data_out = erq(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("PHQ"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("PHQ"), 1), None, does_not_raise()),
            (data_filtered_correct("PHQ"), None, does_not_raise()),
            (
                data_filtered_correct("PHQ"),
                ["PHQ_{:01d}".format(i) for i in range(1, 11)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("PHQ"), ["PHQ_{:01d}".format(i) for i in range(1, 10)], does_not_raise()),
            (
                data_filtered_correct("PHQ"),
                ["PHQ_{:02d}".format(i) for i in range(1, 10)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("PHQ"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_phq_raises(self, data, columns, expected):
        with expected:
            phq(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("PHQ"), None, result_filtered("PHQ")),
            (
                data_filtered_correct("PHQ"),
                ["PHQ_{:01d}".format(i) for i in range(1, 10)],
                result_filtered("PHQ"),
            ),
            (
                convert_scale(data_filtered_wrong_range("PHQ"), 1),
                None,
                result_filtered("PHQ"),
            ),
        ],
    )
    def test_phq(self, data, columns, result):
        data_out = phq(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SDS_"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SDS_"), 1), None, does_not_raise()),
            (data_filtered_correct("SDS_"), None, does_not_raise()),
            (
                data_filtered_correct("SDS_"),
                ["SDS_{:01d}".format(i) for i in range(1, 6)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SDS_"), ["SDS_{:01d}".format(i) for i in range(1, 5)], does_not_raise()),
            (
                data_filtered_correct("SDS_"),
                ["SDS_{:02d}".format(i) for i in range(1, 5)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SDS_"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_sds_raises(self, data, columns, expected):
        with expected:
            sds(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("SDS_"), None, result_filtered("SDS")),
            (
                data_filtered_correct("SDS_"),
                ["SDS_{:01d}".format(i) for i in range(1, 5)],
                result_filtered("SDS"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SDS_"), 1),
                None,
                result_filtered("SDS"),
            ),
        ],
    )
    def test_sds(self, data, columns, result):
        data_out = sds(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("EV_"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("EV_"), 1), None, does_not_raise()),
            (data_filtered_correct("EV_"), None, does_not_raise()),
            (
                data_filtered_correct("EV_"),
                ["EV_{:01d}".format(i) for i in range(1, 5)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("EV_"), ["EV_{:01d}".format(i) for i in range(1, 4)], does_not_raise()),
            (
                data_filtered_correct("EV_"),
                ["EV_{:02d}".format(i) for i in range(1, 4)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("EV_"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_eval_clinic_raises(self, data, columns, expected):
        with expected:
            eval_clinic(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("EV_"), None, result_filtered("EvalClinic")),
            (
                data_filtered_correct("EV_"),
                ["EV_{:01d}".format(i) for i in range(1, 4)],
                result_filtered("EvalClinic"),
            ),
            (
                convert_scale(data_filtered_wrong_range("EV_"), 1),
                None,
                result_filtered("EvalClinic"),
            ),
        ],
    )
    def test_eval_clinic(self, data, columns, result):
        data_out = eval_clinic(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("ASKU"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("ASKU"), 1), None, does_not_raise()),
            (data_filtered_correct("ASKU"), None, does_not_raise()),
            (
                data_filtered_correct("ASKU"),
                ["ASKU_{:01d}".format(i) for i in range(1, 5)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("ASKU"), ["ASKU_{:01d}".format(i) for i in range(1, 4)], does_not_raise()),
            (
                data_filtered_correct("ASKU"),
                ["ASKU_{:02d}".format(i) for i in range(1, 4)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("ASKU"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_asku_raises(self, data, columns, expected):
        with expected:
            asku(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("ASKU"), None, result_filtered("ASKU")),
            (
                data_filtered_correct("ASKU"),
                ["ASKU_{:01d}".format(i) for i in range(1, 4)],
                result_filtered("ASKU"),
            ),
            (
                convert_scale(data_filtered_wrong_range("ASKU"), 1),
                None,
                result_filtered("ASKU"),
            ),
        ],
    )
    def test_asku(self, data, columns, result):
        data_out = asku(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("SWLS"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("SWLS"), 1), None, does_not_raise()),
            (data_filtered_correct("SWLS"), None, does_not_raise()),
            (
                data_filtered_correct("SWLS"),
                ["SWLS_{:01d}".format(i) for i in range(1, 7)],
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("SWLS"), ["SWLS_{:01d}".format(i) for i in range(1, 6)], does_not_raise()),
            (
                data_filtered_correct("SWLS"),
                ["SWLS_{:02d}".format(i) for i in range(1, 6)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("SWLS"),
                None,
                does_not_raise(),
            ),
        ],
    )
    def test_swls_raises(self, data, columns, expected):
        with expected:
            swls(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("SWLS"), None, result_filtered("SWLS")),
            (
                data_filtered_correct("SWLS"),
                ["SWLS_{:01d}".format(i) for i in range(1, 6)],
                result_filtered("SWLS"),
            ),
            (
                convert_scale(data_filtered_wrong_range("SWLS"), 1),
                None,
                result_filtered("SWLS"),
            ),
        ],
    )
    def test_swls(self, data, columns, result):
        data_out = swls(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, subscales, expected",
        [
            (data_complete_correct(), None, None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("WPI"), None, None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("WPI"), 1), None, None, does_not_raise()),
            (data_filtered_correct("WPI"), None, None, does_not_raise()),
            (
                data_filtered_correct("WPI"),
                ["WPI_{}".format(i) for i in range(1, 30)],
                None,
                pytest.raises(ValidationError),
            ),
            (data_filtered_correct("WPI"), ["WPI_{}".format(i) for i in range(1, 36)], None, does_not_raise()),
            (
                data_filtered_correct("WPI"),
                ["WPI{:02d}".format(i) for i in range(1, 36)],
                None,
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("WPI"),
                None,
                {
                    "AccessTreatment": [2, 3, 4, 5, 6],
                    "StaffCompetence": [11, 12],
                    "EffectTreatment": [22, 23, 24],
                    "StationEquipment": [8, 9, 10],
                    "Relation": [7, 15, 16, 17, 18, 19, 21],
                    "Information": [13, 14, 20],
                    "OverallSatisfaction": [1],
                    "TreatmentInterventions": [25, 28, 29, 30, 31],
                    "Education": [26, 27],
                    "Support": [32, 33, 34, 35],
                },
                does_not_raise(),
            ),
            (
                data_filtered_correct("WPI"),
                None,
                {
                    "Information": [13, 14, 20],
                },
                does_not_raise(),
            ),
        ],
    )
    def test_wpi_raises(self, data, columns, subscales, expected):
        with expected:
            wpi(data, columns, subscales)

    @pytest.mark.parametrize(
        "data, columns, subscales, result",
        [
            (data_filtered_correct("WPI"), None, None, result_filtered("WPI")),
            (
                data_filtered_correct("WPI"),
                ["WPI_{}".format(i) for i in range(1, 36)],
                None,
                result_filtered("WPI"),
            ),
            (
                convert_scale(data_filtered_wrong_range("WPI"), 1),
                None,
                None,
                result_filtered("WPI"),
            ),
            (
                data_subscale("wpi"),
                None,
                {"Relation": [1, 2, 3, 4, 5, 6, 7]},
                result_filtered("WPI_Relation"),
            ),
        ],
    )
    def test_wpi(self, data, columns, subscales, result):
        data_out = wpi(data, columns, subscales)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result, check_dtype=False)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("IDQ_PRE"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("IDQ_PRE"), 1), None, does_not_raise()),
            (data_filtered_correct("IDQ_PRE"), None, does_not_raise()),
            (
                data_filtered_correct("IDQ_PRE"),
                ["IDQ_PRE{:02d}".format(i) for i in range(1, 6)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("IDQ_PRE"),
                ["IDQ_PRE_{}".format(i) for i in range(1, 4)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("IDQ_PRE"),
                ["IDQ_PRE_{}".format(i) for i in range(1, 6)],
                does_not_raise(),
            ),
        ],
    )
    def test_idq_pre_scan_raises(self, data, columns, expected):
        with expected:
            idq_pre_scan(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("IDQ_PRE"), None, result_filtered("IDQ_PRE")),
            (
                data_filtered_correct("IDQ_PRE"),
                ["IDQ_PRE_{}".format(i) for i in range(1, 6)],
                result_filtered("IDQ_PRE"),
            ),
            (
                convert_scale(data_filtered_wrong_range("IDQ_PRE"), 1),
                None,
                result_filtered("IDQ_PRE"),
            ),
        ],
    )
    def test_idq_pre_scan(self, data, columns, result):
        data_out = idq_pre_scan(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)

    @pytest.mark.parametrize(
        "data, columns, expected",
        [
            (data_complete_correct(), None, pytest.raises(ValidationError)),
            (data_filtered_wrong_range("IDQ_POST"), None, pytest.raises(ValueRangeError)),
            (convert_scale(data_filtered_wrong_range("IDQ_POST"), 1), None, does_not_raise()),
            (data_filtered_correct("IDQ_POST"), None, does_not_raise()),
            (
                data_filtered_correct("IDQ_POST"),
                ["IDQ_POST{:02d}".format(i) for i in range(1, 8)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("IDQ_POST"),
                ["IDQ_POST_{}".format(i) for i in range(1, 4)],
                pytest.raises(ValidationError),
            ),
            (
                data_filtered_correct("IDQ_POST"),
                ["IDQ_POST_{}".format(i) for i in range(1, 8)],
                does_not_raise(),
            ),
        ],
    )
    def test_idq_post_scan_raises(self, data, columns, expected):
        with expected:
            idq_post_scan(data, columns)

    @pytest.mark.parametrize(
        "data, columns, result",
        [
            (data_filtered_correct("IDQ_POST"), None, result_filtered("IDQ_POST")),
            (
                data_filtered_correct("IDQ_POST"),
                ["IDQ_POST_{}".format(i) for i in range(1, 8)],
                result_filtered("IDQ_POST"),
            ),
            (
                convert_scale(data_filtered_wrong_range("IDQ_POST"), 1),
                None,
                result_filtered("IDQ_POST"),
            ),
        ],
    )
    def test_idq_post_scan(self, data, columns, result):
        data_out = idq_post_scan(data, columns)
        TestCase().assertListEqual(list(data_out.columns), list(result.columns))
        assert_frame_equal(data_out, result)
