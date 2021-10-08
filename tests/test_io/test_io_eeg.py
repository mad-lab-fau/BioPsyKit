from contextlib import contextmanager

import pytest

from biopsykit.example_data import _get_data
from biopsykit.io.eeg import load_eeg_raw_muse
from biopsykit.utils.exceptions import ValidationError


@contextmanager
def does_not_raise():
    yield


class TestIoEeg:
    @pytest.mark.parametrize(
        "file_path, expected",
        [
            (_get_data("eeg_muse_example.csv"), does_not_raise()),
            (_get_data("cortisol_sample.csv"), pytest.raises(ValidationError)),
        ],
    )
    def test_load_eeg_raw_muse(self, file_path, expected):
        with expected:
            load_eeg_raw_muse(file_path)
