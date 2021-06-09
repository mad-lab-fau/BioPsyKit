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
        "file_path, expected",
        [
            ("eeg_muse_example.csv", does_not_raise()),
            ("cortisol_sample.csv", pytest.raises(ValidationError)),
        ],
    )
    def test_load_eeg_raw_muse(self, file_path, expected):
        with expected:
            load_eeg_raw_muse(_EXAMPLE_DATA_PATH.joinpath(file_path))
