"""Tests for the tpcp execution contract of ECG outlier algorithms."""

from pathlib import Path

import pandas as pd
import pytest
from tpcp import make_action_safe

from biopsykit.signals.ecg.event_extraction import RPeakExtractionNeurokit
from biopsykit.signals.ecg.outlier_correction import (
    RPeakOutlierCorrection,
    RPeakOutlierDetectionBerntson1990,
    RPeakOutlierDetectionCorrelation,
    RPeakOutlierDetectionPhysiological,
    RPeakOutlierDetectionQuality,
    RPeakOutlierDetectionRRDiffIntervalStatistics,
    RPeakOutlierDetectionRRIntervalStatistics,
)
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit


@pytest.fixture(scope="module")
def ecg_and_rpeaks():
    """Prepare a real recording and detected peaks for every outlier action."""
    path = Path(__file__).parent.parent / "test_data" / "pep" / "pep_test_ecg.csv"
    ecg = pd.read_csv(path, index_col=0, parse_dates=True)
    ecg = EcgPreprocessingNeurokit(method="neurokit").clean(ecg=ecg, sampling_rate_hz=1000).ecg_clean_
    peaks = RPeakExtractionNeurokit().extract(ecg=ecg, sampling_rate_hz=1000).points_
    return ecg, peaks


@pytest.mark.parametrize(
    "algorithm_type",
    [
        RPeakOutlierDetectionBerntson1990,
        RPeakOutlierDetectionCorrelation,
        RPeakOutlierDetectionPhysiological,
        RPeakOutlierDetectionQuality,
        RPeakOutlierDetectionRRIntervalStatistics,
        RPeakOutlierDetectionRRDiffIntervalStatistics,
    ],
)
def test_outlier_detector_is_a_safe_chainable_action(algorithm_type, ecg_and_rpeaks):
    """Each detector returns its instance and exposes results without mutating inputs."""
    ecg, peaks = ecg_and_rpeaks
    ecg_before, peaks_before = ecg.copy(deep=True), peaks.copy(deep=True)
    algorithm = algorithm_type()

    result = make_action_safe(algorithm_type.detect_outlier)(algorithm, ecg=ecg, rpeaks=peaks, sampling_rate_hz=1000)

    assert result is algorithm
    assert isinstance(result.points_, pd.DataFrame)
    assert list(result.points_.columns) == ["is_outlier"]
    assert len(result.points_) == len(peaks)
    pd.testing.assert_frame_equal(ecg, ecg_before)
    pd.testing.assert_frame_equal(peaks, peaks_before)


@pytest.mark.parametrize("imputation_type", ["linear_interpolation", "moving_average"])
def test_outlier_corrector_is_a_safe_chainable_action(imputation_type, ecg_and_rpeaks):
    """Correction returns its instance and leaves caller-owned data unchanged."""
    ecg, peaks = ecg_and_rpeaks
    ecg_before, peaks_before = ecg.copy(deep=True), peaks.copy(deep=True)
    mask = pd.DataFrame({"is_outlier": False}, index=peaks.index)
    mask.iloc[2, 0] = True
    # An explicit dictionary isolates the return contract from default-parameter normalization.
    algorithm = RPeakOutlierCorrection(imputation_type=imputation_type, imputation_params={})

    result = make_action_safe(RPeakOutlierCorrection.correct_outlier)(
        algorithm, ecg=ecg, rpeaks=peaks, outlier_detection_results=mask
    )

    assert result is algorithm
    assert isinstance(result.points_, pd.DataFrame)
    assert isinstance(result.ecg_processed_, pd.DataFrame)
    assert result.points_["r_peak_outlier"].sum() == 1
    pd.testing.assert_frame_equal(ecg, ecg_before)
    pd.testing.assert_frame_equal(peaks, peaks_before)
