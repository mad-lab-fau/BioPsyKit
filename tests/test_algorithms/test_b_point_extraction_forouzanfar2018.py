from contextlib import contextmanager
from pathlib import Path

import unittest


import pandas as pd

from biopsykit.signals.ecg.segmentation._heartbeat_segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.event_extraction import (
    CPointExtractionScipyFindPeaks,
    BPointExtractionDebski1993,
    BPointExtractionDrost2022,
    BPointExtractionForouzanfar2018,
)
from biopsykit.signals.icg.event_extraction import BPointExtractionArbol2017

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/pep")


@contextmanager
def does_not_raise():
    yield


class TestBPointExtractionForouzanfar2018:
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
        self.test_case = unittest.TestCase()

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(
            icg=self.icg_data,
            heartbeats=self.heartbeats,
            c_points=self.c_points,
            sampling_rate_hz=self.sampling_rate_hz,
        )

        self.test_case.assertIsInstance(self.extract_algo.points_, pd.DataFrame)
        self.test_case.assertIn("b_point_sample", self.extract_algo.points_.columns)
        self.test_case.assertIn("nan_reason", self.extract_algo.points_.columns)

    # add regression test to check if the extracted q-wave onsets match with the saved reference
    def test_regression_extract_dataframe(self):
        self.setup()

        icg_data = self.icg_data
        _assert_is_dtype(icg_data, pd.DataFrame)

        reference_b_points = self._get_regression_reference()
        self.extract_algo.extract(
            icg=icg_data, heartbeats=self.heartbeats, c_points=self.c_points, sampling_rate_hz=self.sampling_rate_hz
        )

        self._check_b_point_equal(reference_b_points, self.extract_algo.points_)

    def test_regression_extract_series(self):
        self.setup()

        icg_data = self.icg_data.squeeze()
        _assert_is_dtype(icg_data, pd.Series)

        reference_b_points = self._get_regression_reference()
        self.extract_algo.extract(
            icg=icg_data, heartbeats=self.heartbeats, c_points=self.c_points, sampling_rate_hz=self.sampling_rate_hz
        )

        self._check_b_point_equal(reference_b_points, self.extract_algo.points_)

    @staticmethod
    def _get_regression_reference():
        data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_b_point_reference_forouzanfar2018.csv"), index_col=0)
        data = data.convert_dtypes(infer_objects=True)
        return data

    @staticmethod
    def _check_b_point_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)
