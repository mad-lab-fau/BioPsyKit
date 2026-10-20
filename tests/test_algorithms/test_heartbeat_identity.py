"""Regression tests for heartbeat labels independent of row positions."""

from pathlib import Path

import pandas as pd
import pytest

from biopsykit.signals.ecg.outlier_correction import RPeakOutlierDetectionRRDiffIntervalStatistics
from biopsykit.signals.icg.event_extraction import BPointExtractionForouzanfar2018
from biopsykit.utils.exceptions import ValidationError


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([100, 200, 300, 400, 500], name="heartbeat_id"),
        pd.Index(["beat_e", "beat_a", "beat_c", "beat_b", "beat_d"], name="heartbeat_id"),
        pd.MultiIndex.from_product([["phase1"], [10, 20, 30, 40, 50]], names=["phase", "heartbeat_id"]),
    ],
)
def test_rr_difference_outlier_mask_preserves_heartbeat_identity(index):
    """Outlier values and their alignment remain correct for arbitrary beat labels."""
    peaks = pd.DataFrame({"rr_interval_ms": [800.0, 800.0, 800.0, 1800.0, 800.0]}, index=index)
    algorithm = RPeakOutlierDetectionRRDiffIntervalStatistics(rr_diff_statistics_threshold=1.0)

    algorithm.detect_outlier(ecg=pd.DataFrame(), rpeaks=peaks, sampling_rate_hz=1000)

    expected = pd.DataFrame({"is_outlier": [False, False, True, True, False]}, index=index)
    pd.testing.assert_frame_equal(algorithm.points_, expected)
    pd.testing.assert_frame_equal(peaks.loc[algorithm.points_["is_outlier"]], peaks.iloc[[2, 3]])


@pytest.fixture
def forouzanfar_inputs():
    """Load the real ICG recording with its reference heartbeat and C-point tables."""
    data_path = Path(__file__).parent.parent / "test_data" / "pep"
    icg = pd.read_csv(data_path / "pep_test_icg.csv", index_col=0)
    heartbeats = pd.read_csv(data_path / "pep_test_heartbeat_reference_variable_length.csv", index_col=0)
    heartbeats = heartbeats.astype(
        {"start_sample": "Int64", "end_sample": "Int64", "r_peak_sample": "Int64", "rr_interval_sample": "Int64"}
    )
    c_points = pd.read_csv(data_path / "pep_test_c_point_reference_scipy_findpeaks.csv", index_col=0)
    c_points = c_points.astype({"c_point_sample": "Int64", "nan_reason": "object"})
    return icg, heartbeats, c_points


@pytest.mark.parametrize("label_type", ["nonconsecutive_integer", "string"])
def test_forouzanfar_extraction_is_invariant_to_heartbeat_relabeling(label_type, forouzanfar_inputs):
    """Relabeling the same chronological beats leaves their extracted B-points unchanged."""
    icg, heartbeats, c_points = forouzanfar_inputs
    algorithm = BPointExtractionForouzanfar2018(handle_missing_events="ignore")
    baseline = algorithm.extract(icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=1000).points_
    if label_type == "nonconsecutive_integer":
        index = pd.Index([10 * (i + 1) for i in range(len(heartbeats))], name="heartbeat_id")
    else:
        index = pd.Index([f"beat_{i}" for i in range(len(heartbeats))], name="heartbeat_id")
    relabeled_heartbeats = heartbeats.set_axis(index)
    relabeled_c_points = c_points.set_axis(index)

    result = (
        algorithm.clone()
        .extract(icg=icg, heartbeats=relabeled_heartbeats, c_points=relabeled_c_points, sampling_rate_hz=1000)
        .points_
    )

    pd.testing.assert_frame_equal(result, baseline.set_axis(index))


def test_rr_difference_outlier_detection_rejects_duplicate_ids():
    """Duplicate beat IDs cannot identify an outlier unambiguously."""
    peaks = pd.DataFrame({"rr_interval_ms": [800.0, 900.0]}, index=pd.Index([10, 10], name="heartbeat_id"))

    with pytest.raises(ValidationError, match="heartbeat IDs must be unique"):
        RPeakOutlierDetectionRRDiffIntervalStatistics().detect_outlier(
            ecg=pd.DataFrame(), rpeaks=peaks, sampling_rate_hz=1000
        )


def test_forouzanfar_aligns_c_points_to_heartbeat_order(forouzanfar_inputs):
    """The C-point table's storage order must not change which heartbeat precedes another."""
    icg, heartbeats, c_points = forouzanfar_inputs
    algorithm = BPointExtractionForouzanfar2018(handle_missing_events="ignore")
    expected = algorithm.extract(icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=1000).points_

    result = (
        algorithm.clone()
        .extract(icg=icg, heartbeats=heartbeats, c_points=c_points.iloc[::-1], sampling_rate_hz=1000)
        .points_
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "invalid_input", ["duplicate_heartbeats", "duplicate_c_points", "missing_c_point", "extra_c_point"]
)
def test_forouzanfar_rejects_ambiguous_or_mismatched_heartbeat_ids(invalid_input, forouzanfar_inputs):
    """Both event tables must contain the same set of unique heartbeat IDs."""
    icg, heartbeats, c_points = forouzanfar_inputs
    if invalid_input == "duplicate_heartbeats":
        heartbeats = heartbeats.set_axis(pd.Index([0] * len(heartbeats), name="heartbeat_id"))
    elif invalid_input == "duplicate_c_points":
        c_points = c_points.set_axis(pd.Index([0] * len(c_points), name="heartbeat_id"))
    elif invalid_input == "missing_c_point":
        c_points = c_points.iloc[:-1]
    else:
        extra_c_point = c_points.iloc[[-1]].set_axis(pd.Index([len(c_points)], name="heartbeat_id"))
        c_points = pd.concat([c_points, extra_c_point])

    with pytest.raises(ValidationError, match="heartbeat IDs"):
        BPointExtractionForouzanfar2018().extract(
            icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=1000
        )
