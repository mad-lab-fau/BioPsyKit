from contextlib import contextmanager
from pathlib import Path

import unittest

import numpy as np
import pandas as pd
import pytest

from biopsykit.signals.ecg.event_extraction import QPeakExtractionNeurokitDwt, QWaveOnsetExtractionVanLien2013
from biopsykit.signals.ecg.segmentation._heartbeat_segmentation import HeartbeatSegmentationNeurokit

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
        self.extract_algo = QPeakExtractionNeurokitDwt()
        self.test_case = unittest.TestCase()

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(ecg=self.ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self.test_case.assertIsInstance(self.extract_algo.points_, pd.DataFrame)
        self.test_case.assertIn("q_wave_onset_sample", self.extract_algo.points_.columns)
        self.test_case.assertIn("nan_reason", self.extract_algo.points_.columns)

    # add regression test to check if the extracted q-wave onsets match with the saved reference
    def test_regression_extract_dataframe(self):
        self.setup()

        ecg_data = self.ecg_data
        _assert_is_dtype(ecg_data, pd.DataFrame)

        reference_q_wave_onset = self._get_regression_reference()
        self.extract_algo.extract(ecg=self.ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_q_wave_onset_equal(reference_q_wave_onset, self.extract_algo.points_)

    def test_regression_extract_series(self):
        self.setup()

        ecg_data = self.ecg_data.squeeze()
        _assert_is_dtype(ecg_data, pd.Series)

        reference_q_wave_onset = self._get_regression_reference()
        self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_q_wave_onset_equal(reference_q_wave_onset, self.extract_algo.points_)

    @staticmethod
    def _get_regression_reference():
        reference_q_wave_onset = pd.read_csv(
            TEST_FILE_PATH.joinpath("pep_test_q_wave_onset_reference_neurokit_dwt.csv"), index_col=0
        )
        reference_q_wave_onset = reference_q_wave_onset.convert_dtypes(infer_objects=True)
        return reference_q_wave_onset

    @staticmethod
    def _check_q_wave_onset_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)


class TestQWaveOnsetExtractionVanLien2013:
    def setup(self, time_interval_ms: int = 40):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.extract_algo = QWaveOnsetExtractionVanLien2013(time_interval_ms=time_interval_ms)
        self.test_case = unittest.TestCase()

    @pytest.mark.parametrize(
        ("time_interval_ms"),
        [34, 36, 38, 40],
    )
    def test_initialization(self, time_interval_ms):
        self.setup(time_interval_ms)
        self.test_case.assertEqual(self.extract_algo.time_interval_ms, time_interval_ms)

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(ecg=self.ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self.test_case.assertIsInstance(self.extract_algo.points_, pd.DataFrame)
        self.test_case.assertIn("q_wave_onset_sample", self.extract_algo.points_.columns)

    # add regression test to check if the extracted q-wave onsets match with the saved reference
    @pytest.mark.parametrize(
        ("time_interval_ms"),
        [34, 36, 38, 40],
    )
    def test_regression_extract_dataframe(self, time_interval_ms):
        self.setup(time_interval_ms)

        ecg_data = self.ecg_data
        _assert_is_dtype(ecg_data, pd.DataFrame)

        reference_q_wave_onset = self._get_regression_reference(time_interval_ms)
        self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_q_wave_onset_equal(reference_q_wave_onset, self.extract_algo.points_)

    @pytest.mark.parametrize(
        ("time_interval_ms"),
        [34, 36, 38, 40],
    )
    def test_regression_extract_series(self, time_interval_ms):
        self.setup(time_interval_ms)

        ecg_data = self.ecg_data.squeeze()
        _assert_is_dtype(ecg_data, pd.Series)

        reference_q_wave_onset = self._get_regression_reference(time_interval_ms)
        self.extract_algo.extract(ecg=ecg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_q_wave_onset_equal(reference_q_wave_onset, self.extract_algo.points_)

    def _get_regression_reference(self, time_interval_ms: int = 40):
        data = pd.read_csv(
            TEST_FILE_PATH.joinpath("pep_test_heartbeat_reference_variable_length.csv"), index_col=0, parse_dates=True
        )
        data = data.convert_dtypes(infer_objects=True)
        data = data[["r_peak_sample"]] - int((time_interval_ms / self.sampling_rate_hz) * 1000)
        data.columns = ["q_wave_onset_sample"]

        return data

    @staticmethod
    def _check_q_wave_onset_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)