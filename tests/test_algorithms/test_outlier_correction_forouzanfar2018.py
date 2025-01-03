import unittest
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from biopsykit.signals.ecg.segmentation._heartbeat_segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.event_extraction import BPointExtractionForouzanfar2018, CPointExtractionScipyFindPeaks
from biopsykit.signals.icg.outlier_correction import OutlierCorrectionForouzanfar2018

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/pep")


@contextmanager
def does_not_raise():
    yield


class TestOutlierCorrectionForouzanfar2018:
    def setup(self):
        # Sample ECG data
        self.ecg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_ecg.csv"), index_col=0)
        self.icg_data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_icg.csv"), index_col=0)
        self.sampling_rate_hz = 1000
        self.segmenter = HeartbeatSegmentationNeurokit()
        self.heartbeats = self.segmenter.extract(
            ecg=self.ecg_data, sampling_rate_hz=self.sampling_rate_hz
        ).heartbeat_list_
        self.c_point_algo = CPointExtractionScipyFindPeaks()
        self.c_points = self.c_point_algo.extract(
            icg=self.icg_data, heartbeats=self.heartbeats, sampling_rate_hz=self.sampling_rate_hz
        ).points_
        self.extract_algo = BPointExtractionForouzanfar2018()
        self.outlier_algo = OutlierCorrectionForouzanfar2018()

        self.b_points = self.extract_algo.extract(
            icg=self.icg_data,
            heartbeats=self.heartbeats,
            c_points=self.c_points,
            sampling_rate_hz=self.sampling_rate_hz,
        ).points_
        self.test_case = unittest.TestCase()

    def test_correct_outlier(self):
        self.setup()

        self.outlier_algo.correct_outlier(
            b_points=self.b_points, c_points=self.c_points, sampling_rate_hz=self.sampling_rate_hz
        )

        assert isinstance(self.outlier_algo.points_, pd.DataFrame)
        assert "b_point_sample" in self.outlier_algo.points_.columns
        assert "nan_reason" in self.outlier_algo.points_.columns

    @pytest.mark.parametrize(
        ("outlier_type"),
        [("middle")],
    )
    def test_regression_correct_outlier(self, outlier_type):
        self.setup()

        if outlier_type == "middle":
            b_points = self._get_b_point_outlier_middle()
        else:
            raise ValueError(f"Unknown outlier type: {outlier_type}")

        self.outlier_algo.correct_outlier(
            b_points=b_points, c_points=self.c_points, sampling_rate_hz=self.sampling_rate_hz
        )

        corrected_beats = (self.b_points - self.outlier_algo.points_)["b_point_sample"] != 0
        corrected_beats = self.b_points.index[corrected_beats]

        self.test_case.assertListEqual(corrected_beats.tolist(), [2, 4, 5, 7])

        reference_corrected_beats = self._get_regression_reference()

        self._check_b_point_equal(reference_corrected_beats, self.outlier_algo.points_)

    def _get_b_point_outlier_middle(self):
        b_points = self.b_points
        # manually set some values to NaN to simulate outliers
        b_points["b_point_sample"].iloc[5] -= 100
        return b_points

    def _get_regression_reference(self):
        data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_icg_outlier_correction_forouzanfar2018.csv"), index_col=0)
        data = data.convert_dtypes(infer_objects=True)
        return data

    @staticmethod
    def _check_b_point_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)
