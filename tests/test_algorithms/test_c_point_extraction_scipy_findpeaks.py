from contextlib import contextmanager
from pathlib import Path

import unittest
from typing import Optional

import pandas as pd
import pytest

from biopsykit.signals.ecg.segmentation._heartbeat_segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.event_extraction import CPointExtractionScipyFindPeaks

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/pep")


@contextmanager
def does_not_raise():
    yield


class TestCPointExtractionSciPyFindpeaks:
    def setup(self):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.icg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_icg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.extract_algo = CPointExtractionScipyFindPeaks()
        self.test_case = unittest.TestCase()

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(icg=self.icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self.test_case.assertIsInstance(self.extract_algo.points_, pd.DataFrame)
        self.test_case.assertIn("c_point_sample", self.extract_algo.points_.columns)
        self.test_case.assertIn("nan_reason", self.extract_algo.points_.columns)

    # add regression test to check if the extracted q-wave onsets match with the saved reference
    def test_regression_extract_dataframe(self):
        self.setup()

        icg_data = self.icg_data
        _assert_is_dtype(icg_data, pd.DataFrame)

        reference_c_points = self._get_regression_reference()
        self.extract_algo.extract(icg=icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_c_point_equal(reference_c_points, self.extract_algo.points_)

    def test_regression_extract_series(self):
        self.setup()

        icg_data = self.icg_data.squeeze()
        _assert_is_dtype(icg_data, pd.Series)

        reference_c_points = self._get_regression_reference()
        self.extract_algo.extract(icg=icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self._check_c_point_equal(reference_c_points, self.extract_algo.points_)

    @staticmethod
    def _get_regression_reference():
        data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_c_point_reference_scipy_findpeaks.csv"), index_col=0)
        data = data.convert_dtypes(infer_objects=True)
        return data

    @staticmethod
    def _check_c_point_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)


class TestCPointExtractionSciPyFindpeaksParameters:
    def setup(
        self,
        window_c_correction: Optional[int] = 3,
        save_candidates: Optional[bool] = False,
    ):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.icg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_icg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.extract_algo = CPointExtractionScipyFindPeaks(
            window_c_correction=window_c_correction, save_candidates=save_candidates
        )
        self.test_case = unittest.TestCase()

    @pytest.mark.parametrize(
        ("window_c_correction"),
        [(1), (2), (3), (5), (7)],
    )
    def test_extract_window_c_correction(self, window_c_correction):
        self.setup(window_c_correction=window_c_correction)

        self.extract_algo.extract(icg=self.icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        print(self.extract_algo.points_)

        self.test_case.assertIsInstance(self.extract_algo.points_, pd.DataFrame)
        self.test_case.assertIn("c_point_sample", self.extract_algo.points_.columns)
        self.test_case.assertIn("nan_reason", self.extract_algo.points_.columns)

    @pytest.mark.parametrize(
        ("save_candidates", "expected_columns"),
        [(True, ["c_point_sample", "nan_reason", "c_point_candidates"]), (False, ["c_point_sample", "nan_reason"])],
    )
    def test_extract_window_(self, save_candidates, expected_columns):
        self.setup(save_candidates=save_candidates)

        self.extract_algo.extract(icg=self.icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        self.test_case.assertIsInstance(self.extract_algo.points_, pd.DataFrame)
        self.test_case.assertListEqual(expected_columns, self.extract_algo.points_.columns.tolist())
