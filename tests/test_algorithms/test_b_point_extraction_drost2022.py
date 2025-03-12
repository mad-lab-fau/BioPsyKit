import unittest
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.event_extraction import (
    BPointExtractionDrost2022,
    CPointExtractionScipyFindPeaks,
)
from biopsykit.utils._datatype_validation_helper import _assert_is_dtype
from biopsykit.utils.exceptions import ValidationError

TEST_FILE_PATH = Path(__file__).parent.joinpath("../test_data/pep")


@contextmanager
def does_not_raise():
    yield


class TestBPointExtractionDrost2022:
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
        self.extract_algo = BPointExtractionDrost2022()
        self.test_case = unittest.TestCase()

    def test_extract(self):
        self.setup()

        self.extract_algo.extract(
            icg=self.icg_data,
            heartbeats=self.heartbeats,
            c_points=self.c_points,
            sampling_rate_hz=self.sampling_rate_hz,
        )

        assert isinstance(self.extract_algo.points_, pd.DataFrame)
        assert "b_point_sample" in self.extract_algo.points_.columns
        assert "nan_reason" in self.extract_algo.points_.columns

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

        with pytest.raises(ValidationError):
            self.extract_algo.extract(
                icg=icg_data, heartbeats=self.heartbeats, c_points=self.c_points, sampling_rate_hz=self.sampling_rate_hz
            )

    @staticmethod
    def _get_regression_reference():
        data = pd.read_csv(TEST_FILE_PATH.joinpath("pep_test_b_point_reference_drost2022.csv"), index_col=0)
        data = data.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        return data

    @staticmethod
    def _check_b_point_equal(reference_heartbeats, extracted_heartbeats):
        pd.testing.assert_frame_equal(reference_heartbeats, extracted_heartbeats)
