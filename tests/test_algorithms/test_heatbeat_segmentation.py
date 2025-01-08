import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biopsykit.signals.ecg.segmentation._heartbeat_segmentation import HeartbeatSegmentationNeurokit
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.exceptions import ValidationError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/pep")


@contextmanager
def does_not_raise():
    yield


class TestHeartbeatSegmentationNeurokit:
    def setup(self, variable_length: bool = True, start_factor: float = 0.35):
        # Sample ECG data
        ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0, parse_dates=True)
        ecg_data.index = ecg_data.index.tz_convert("Europe/Berlin")
        self.ecg_data = ecg_data
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit(variable_length=variable_length, start_factor=start_factor)
        self.test_case = unittest.TestCase()

    def test_initialization(self):
        self.setup()
        assert self.segmenter.variable_length
        assert self.segmenter.start_factor == 0.35

    def test_extract_variable_length(self):
        self.setup()

        self.segmenter.extract(ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz)
        assert isinstance(self.segmenter.heartbeat_list_, pd.DataFrame)
        assert "start_sample" in self.segmenter.heartbeat_list_.columns
        assert "end_sample" in self.segmenter.heartbeat_list_.columns
        assert "r_peak_sample" in self.segmenter.heartbeat_list_.columns

    def test_extract_fixed_length(self):
        self.setup()

        self.segmenter.variable_length = False
        self.segmenter.extract(ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz)
        assert isinstance(self.segmenter.heartbeat_list_, pd.DataFrame)
        assert "start_sample" in self.segmenter.heartbeat_list_.columns
        assert "end_sample" in self.segmenter.heartbeat_list_.columns
        assert "r_peak_sample" in self.segmenter.heartbeat_list_.columns

    # add regression test to check if the extracted heartbeats are correct
    def test_regression_extract_variable_length_dataframe(self):
        self.setup()

        reference_heartbeats = self._get_regression_reference("pep_test_heartbeat_reference_variable_length.csv")

        ecg_data = self.ecg_data
        _assert_is_dtype(ecg_data, pd.DataFrame)

        self.segmenter.extract(ecg=ecg_data, sampling_rate_hz=self.sampling_rate_hz)
        # check if the extraction is equal
        self._check_heartbeats_equal(reference_heartbeats, self.segmenter.heartbeat_list_)

    def test_regression_extract_variable_length_series(self):
        self.setup()

        ecg_data = self.ecg_data["ecg"]
        _assert_is_dtype(ecg_data, pd.Series)

        with pytest.raises(ValidationError):
            self.segmenter.extract(ecg=ecg_data, sampling_rate_hz=self.sampling_rate_hz)

    def test_regression_extract_fixed_length_dataframe(self):
        self.setup()

        reference_heartbeats = self._get_regression_reference("pep_test_heartbeat_reference_fixed_length.csv")

        ecg_data = self.ecg_data
        _assert_is_dtype(ecg_data, pd.DataFrame)

        self.segmenter.variable_length = False
        self.segmenter.extract(ecg=ecg_data, sampling_rate_hz=self.sampling_rate_hz)
        # check if the first heartbeat is correct
        self._check_heartbeats_equal(reference_heartbeats, self.segmenter.heartbeat_list_)

    def test_regression_extract_fixed_length_series(self):
        self.setup(variable_length=False)

        ecg_data = self.ecg_data["ecg"]
        _assert_is_dtype(ecg_data, pd.Series)

        with pytest.raises(ValidationError):
            self.segmenter.extract(ecg=ecg_data, sampling_rate_hz=self.sampling_rate_hz)

    def test_regression_extract_fixed_length_numpy(self):
        self.setup(variable_length=False)

        ecg_data = self.ecg_data["ecg"].to_numpy()
        _assert_is_dtype(ecg_data, np.ndarray)

        with pytest.raises(ValidationError):
            self.segmenter.extract(ecg=ecg_data, sampling_rate_hz=self.sampling_rate_hz)

    @staticmethod
    def _get_regression_reference(file_path):
        data = pd.read_csv(TEST_FILE_PATH.joinpath(file_path), index_col=0, parse_dates=True)
        data = data.astype(
            {
                "start_sample": "Int64",
                "end_sample": "Int64",
                "r_peak_sample": "Int64",
                "rr_interval_sample": "Int64",
                "rr_interval_ms": "Float64",
            }
        )
        data["start_time"] = pd.to_datetime(data["start_time"]).dt.tz_convert("Europe/Berlin")
        return data

    @staticmethod
    def _check_heartbeats_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (None, pytest.raises(ValidationError)),
            (pd.Series([], dtype="Float64"), pytest.raises(ValidationError)),
            (pd.DataFrame(), pytest.raises(ValidationError)),
        ],
    )
    def test_invalid_ecg_data(self, data, expected):
        self.setup()

        with expected:
            self.segmenter.extract(ecg=data, sampling_rate_hz=self.sampling_rate_hz)
