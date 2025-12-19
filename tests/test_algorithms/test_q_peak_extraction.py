import unittest
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from biopsykit.signals.ecg.event_extraction import QPeakExtractionMartinez2004Neurokit, QPeakExtractionVanLien2013
from biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit import HeartbeatSegmentationNeurokit
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.exceptions import ValidationError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/pep")


@contextmanager
def does_not_raise():
    yield


class TestQPeakExtractionNeurokitDwt:
    def setup(self):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.extract_algo = QPeakExtractionMartinez2004Neurokit()
        self.test_case = unittest.TestCase()

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(ecg=self.ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        assert isinstance(self.extract_algo.points_, pd.DataFrame)
        assert "q_peak_sample" in self.extract_algo.points_.columns
        assert "nan_reason" in self.extract_algo.points_.columns

    # add regression test to check if the extracted q-peaks match with the saved reference
    def test_regression_extract_dataframe(self):
        self.setup()

        ecg_data = self.ecg_data
        _assert_is_dtype(ecg_data, pd.DataFrame)

        reference_q_peaks = self._get_regression_reference()
        self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_q_peaks_equal(reference_q_peaks, self.extract_algo.points_)

    def test_regression_extract_series(self):
        self.setup()

        ecg_data = self.ecg_data.squeeze()
        _assert_is_dtype(ecg_data, pd.Series)

        with pytest.raises(ValidationError):
            self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

    @staticmethod
    def _get_regression_reference():
        data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_q_peak_reference_neurokit_dwt.csv"), index_col=0)
        data = data.astype({"q_peak_sample": "Int64", "nan_reason": "object"})
        return data

    @staticmethod
    def _check_q_peaks_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)


class TestQPeakExtractionVanLien2013:
    def setup(self, time_interval_ms: int = 40):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.extract_algo = QPeakExtractionVanLien2013(time_interval_ms=time_interval_ms)

    @pytest.mark.parametrize(
        ("time_interval_ms"),
        [34, 36, 38, 40],
    )
    def test_initialization(self, time_interval_ms):
        self.setup(time_interval_ms)
        assert self.extract_algo.time_interval_ms == time_interval_ms

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(ecg=self.ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        assert isinstance(self.extract_algo.points_, pd.DataFrame)
        assert "q_peak_sample" in self.extract_algo.points_.columns

    # add regression test to check if the extracted q-peaks match with the saved reference
    @pytest.mark.parametrize(
        ("time_interval_ms"),
        [34, 36, 38, 40],
    )
    def test_regression_extract_dataframe(self, time_interval_ms):
        self.setup(time_interval_ms)

        ecg_data = self.ecg_data
        _assert_is_dtype(ecg_data, pd.DataFrame)

        reference_q_peaks = self._get_regression_reference(time_interval_ms)
        self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)
        self._check_q_peaks_equal(reference_q_peaks, self.extract_algo.points_)

    @pytest.mark.parametrize(
        ("time_interval_ms"),
        [34, 36, 38, 40],
    )
    def test_regression_extract_series(self, time_interval_ms):
        self.setup(time_interval_ms)

        ecg_data = self.ecg_data.squeeze()
        _assert_is_dtype(ecg_data, pd.Series)

        with pytest.raises(ValidationError):
            self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

    def _get_regression_reference(self, time_interval_ms: int = 40):
        data = pd.read_csv(
            TEST_FILE_PATH.joinpath("pep_test_heartbeat_reference_variable_length.csv"), index_col=0, parse_dates=True
        )
        data = data[["r_peak_sample"]] - int((time_interval_ms / self.sampling_rate_hz) * 1000)
        data = data.assign(nan_reason=pd.NA)
        data.columns = ["q_peak_sample", "nan_reason"]
        data = data.astype({"q_peak_sample": "Int64", "nan_reason": "object"})
        return data

    @staticmethod
    def _check_q_peaks_equal(reference_heartbeats, extracted_heartbeats):
        print("HELLO 2")
        print(reference_heartbeats)
        print(extracted_heartbeats)
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)
