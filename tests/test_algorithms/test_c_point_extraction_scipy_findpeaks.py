import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.event_extraction import CPointExtractionScipyFindPeaks
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.exceptions import ValidationError

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

        assert isinstance(self.extract_algo.points_, pd.DataFrame)
        assert "c_point_sample" in self.extract_algo.points_.columns
        assert "nan_reason" in self.extract_algo.points_.columns

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

        with pytest.raises(ValidationError):
            self.extract_algo.extract(icg=icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

    @staticmethod
    def _get_regression_reference():
        data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_c_point_reference_scipy_findpeaks.csv"), index_col=0)
        data = data.astype({"c_point_sample": "Int64", "nan_reason": "object"})
        return data

    @staticmethod
    def _check_c_point_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)


class TestCPointExtractionSciPyFindpeaksParameters:
    def setup(
        self,
        window_c_correction: Optional[int] = 3,
    ):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.icg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_icg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.extract_algo = CPointExtractionScipyFindPeaks(window_c_correction=window_c_correction)
        self.test_case = unittest.TestCase()

    @pytest.mark.parametrize(
        ("window_c_correction"),
        [(1), (2), (3), (5), (7)],
    )
    def test_extract_window_c_correction(self, window_c_correction):
        self.setup(window_c_correction=window_c_correction)

        self.extract_algo.extract(icg=self.icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz)

        assert isinstance(self.extract_algo.points_, pd.DataFrame)
        assert "c_point_sample" in self.extract_algo.points_.columns
        assert "nan_reason" in self.extract_algo.points_.columns
