from contextlib import contextmanager

import pytest
from biopsykit.io.eeg import load_eeg_raw_muse
from biopsykit.utils.exceptions import ValidationError

from biopsykit.example_data import _EXAMPLE_DATA_PATH


@contextmanager
def does_not_raise():
    yield


class TestIoEeg:
    @pytest.mark.parametrize(
        "input_data, expected",
        [
            (_EXAMPLE_DATA_PATH.joinpath("eeg_muse_example.csv"), does_not_raise()),
            (_EXAMPLE_DATA_PATH.joinpath("cortisol_sample.csv"), pytest.raises(ValidationError)),
        ],
    )
    def test_load_eeg_raw_muse(self, input_data, expected):
        with expected:
            load_eeg_raw_muse(input_data)
