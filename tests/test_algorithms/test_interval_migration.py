"""Contract tests for sample-position intervals and their consumers."""

import numpy as np
import pandas as pd
import pytest

from biopsykit.signals.ecg.event_extraction import (
    QPeakExtractionMartinez2004Neurokit,
    QPeakExtractionSciPyFindPeaksNeurokit,
)
from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
from biopsykit.signals.icg.event_extraction import (
    BPointExtractionArbol2017IsoelectricCrossings,
    CPointExtractionScipyFindPeaks,
)
from biopsykit.signals.imu.feature_extraction.static_moments import (
    compute_features,
    mean_orientation,
    static_moment_duration,
)
from biopsykit.signals.imu.static_moment_detection import find_first_static_window_multi_sensor, find_static_moments
from biopsykit.utils.array_handling import bool_array_to_start_end_array


def _data(length, sampling_rate_hz=100):
    return pd.DataFrame(
        np.zeros((length, 3)),
        columns=["acc_x", "acc_y", "acc_z"],
        index=pd.date_range("2026-10-20", periods=length, freq=pd.Timedelta(seconds=1 / sampling_rate_hz), tz="UTC"),
    )


def _intervals(rows):
    return pd.DataFrame(rows, columns=["start", "end"], dtype=int)


def test_detector_to_duration_and_orientation_includes_the_final_sample():
    data = _data(100)
    data.iloc[-1, 0] = 100.0
    moments = find_static_moments(data, threshold=1000, window_samples=10, metric="mean")

    pd.testing.assert_frame_equal(moments, _intervals([[0, 100]]))
    assert len(data.iloc[moments.iloc[0, 0] : moments.iloc[0, 1]]) == 100
    assert static_moment_duration(data, moments.iloc[0].to_numpy(), sampling_rate_hz=100) == pytest.approx(1.0)
    assert mean_orientation(data, moments).iloc[0]["acc_x"] == pytest.approx(1.0)
    result = compute_features(data, moments, sampling_rate_hz=100).iloc[0]
    assert result["sm_number"] == 1
    assert result["sm_max"] == pytest.approx(1.0)
    assert result["sm_number_60"] == 0
    assert np.isnan(result["sm_max_60"])


@pytest.mark.parametrize("length", [0, 100])
def test_no_static_moments_produce_zero_counts_and_nan_statistics(length):
    data = _data(length)
    moments = find_static_moments(data, threshold=-1, window_samples=10)
    result = compute_features(data, moments, sampling_rate_hz=100).iloc[0]

    assert result["sm_number"] == result["sm_number_60"] == 0
    assert result.drop(["sm_number", "sm_number_60"]).isna().all()
    assert mean_orientation(data, moments).empty


@pytest.mark.parametrize("interval", [[0, 1], [99, 100], [0, 100], [20, 35]])
def test_duration_measures_sample_coverage_even_at_recording_boundaries(interval):
    assert static_moment_duration(_data(100), np.array(interval), sampling_rate_hz=100) == pytest.approx(
        (interval[1] - interval[0]) / 100
    )


@pytest.mark.parametrize("length, expected_count", [(5999, 0), (6000, 1), (6001, 1)])
def test_sixty_second_threshold_uses_covered_samples(length, expected_count):
    result = compute_features(_data(length), _intervals([[0, length]]), sampling_rate_hz=100).iloc[0]

    assert result["sm_max"] == pytest.approx(length / 100)
    assert result["sm_number_60"] == expected_count
    if expected_count:
        assert result["sm_max_60"] == pytest.approx(length / 100)


def test_features_have_hand_calculated_duration_statistics_and_position():
    data = _data(100, sampling_rate_hz=1)
    result = compute_features(data, _intervals([[10, 12], [20, 24], [30, 36]]), sampling_rate_hz=1).iloc[0]

    assert result["sm_number"] == 3
    assert result["sm_max"] == 6
    assert result["sm_median"] == result["sm_mean"] == 4
    assert result["sm_std"] == 2
    assert result["sm_skewness"] == pytest.approx(0)
    assert result["sm_max_position"] == pytest.approx(0.3)


def test_requested_time_window_clips_intervals_and_excludes_its_end():
    data = _data(10, sampling_rate_hz=1)
    result = compute_features(
        data,
        _intervals([[0, 4], [5, 10]]),
        start=data.index[2],
        end=data.index[8],
        sampling_rate_hz=1,
    ).iloc[0]

    assert result["sm_number"] == 2
    assert result["sm_max"] == 3  # [5,8), not [5,9)
    assert result["sm_mean"] == 2.5  # [2,4) and [5,8)
    assert result["sm_max_position"] == pytest.approx(0.5)


def test_time_window_without_moments_has_stable_empty_features():
    data = _data(10, sampling_rate_hz=1)
    result = compute_features(
        data, _intervals([[0, 2]]), start=data.index[5], end=data.index[8], sampling_rate_hz=1
    ).iloc[0]

    assert result["sm_number"] == 0
    assert np.isnan(result["sm_max_position"])


def test_equal_length_adjacent_and_overlapping_intervals_preserve_identity():
    data = _data(4)
    data["acc_x"] = [1.0, 3.0, 5.0, 7.0]
    moments = _intervals([[0, 2], [2, 4], [1, 3]])
    moments.index = pd.Index([10, 20, 30], name="interval_id")
    before = moments.copy(deep=True)

    result = mean_orientation(data, moments)

    pd.testing.assert_index_equal(result.index, moments.index)
    np.testing.assert_allclose(result["acc_x"], [2.0, 6.0, 4.0])
    pd.testing.assert_frame_equal(moments, before)


def test_nullable_integer_intervals_and_arbitrary_signal_labels_are_positional():
    data = _data(4)
    data.index = [10, 20, 30, 40]
    data["acc_x"] = [1.0, 3.0, 5.0, 7.0]
    moments = _intervals([[2, 4]]).astype("Int64")

    assert mean_orientation(data, moments).iloc[0]["acc_x"] == 6
    assert static_moment_duration(data, moments.iloc[0].to_numpy(), sampling_rate_hz=2) == 1


@pytest.mark.parametrize("interval", [[-1, 2], [1, 1], [2, 1], [0, 101], [0.0, 2.5], [True, False]])
def test_invalid_intervals_are_rejected_instead_of_silently_sliced(interval):
    data = _data(100)
    with pytest.raises(ValueError, match="Interval"):
        static_moment_duration(data, np.asarray(interval), sampling_rate_hz=100)
    with pytest.raises(ValueError, match="Interval"):
        mean_orientation(data, np.asarray([interval]))


@pytest.mark.parametrize("sampling_rate_hz", [0, -1, np.nan, np.inf, True])
def test_invalid_sampling_rates_are_rejected(sampling_rate_hz):
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        static_moment_duration(_data(100), np.array([0, 1]), sampling_rate_hz=sampling_rate_hz)


def test_supplied_sampling_rate_must_agree_with_datetime_spacing():
    with pytest.raises(ValueError, match="regularly sampled"):
        compute_features(_data(100), _intervals([[0, 100]]), sampling_rate_hz=50)


def test_datetime_rounding_to_nanoseconds_is_accepted():
    sampling_rate_hz = 128
    data = _data(100, sampling_rate_hz)
    data.index = data.index[0] + pd.to_timedelta(np.arange(100) / sampling_rate_hz, unit="s")

    assert static_moment_duration(data, np.array([0, 100]), sampling_rate_hz=sampling_rate_hz) == pytest.approx(
        100 / sampling_rate_hz
    )


def test_reversed_feature_time_window_is_rejected():
    data = _data(100)
    with pytest.raises(ValueError, match="end must be after"):
        compute_features(data, _intervals([[0, 100]]), start=data.index[-1], end=data.index[0], sampling_rate_hz=100)


@pytest.mark.parametrize("consumer", [static_moment_duration, compute_features])
def test_irregular_datetime_data_require_resampling_or_an_explicit_other_policy(consumer):
    data = _data(3)
    data.index = data.index[[0, 2, 1]]
    moments = np.array([0, 3]) if consumer is static_moment_duration else _intervals([[0, 3]])
    with pytest.raises(ValueError, match="regularly sampled"):
        consumer(data, moments, sampling_rate_hz=100)


def test_multisensor_window_has_an_exclusive_stop():
    signals = [np.zeros((20, 3)), np.zeros((20, 3))]
    start, end = find_first_static_window_multi_sensor(signals, 5, 1, "mean")

    assert (start, end) == (0, 5)
    assert len(signals[0][start:end]) == 5


def test_all_short_boolean_masks_roundtrip_through_exclusive_intervals():
    for length in range(8):
        for code in range(2**length):
            mask = np.array([bool(code & (1 << position)) for position in range(length)])
            intervals = bool_array_to_start_end_array(mask)
            reconstructed = np.zeros(length, dtype=bool)
            for start, end in intervals:
                assert 0 <= start < end <= length
                reconstructed[start:end] = True
            np.testing.assert_array_equal(reconstructed, mask)


def _ecg(length):
    return pd.DataFrame({"ecg": np.zeros(length)}, index=pd.RangeIndex(1000, 1000 + length))


def _mock_peaks(monkeypatch, peaks):
    monkeypatch.setattr(
        "biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit.nk.ecg_peaks",
        lambda *args, **kwargs: (None, {"ECG_R_Peaks": np.asarray(peaks)}),
    )


def test_variable_length_heartbeat_ending_exactly_at_recording_end_is_complete(monkeypatch):
    _mock_peaks(monkeypatch, [3, 7])
    result = HeartbeatSegmentationNeurokit(start_factor=0.5).extract(ecg=_ecg(9), sampling_rate_hz=10).heartbeat_list_

    np.testing.assert_array_equal(result[["start_sample", "end_sample"]], [[1, 5], [5, 9]])
    assert result.iloc[-1]["end_sample"] == 9
    assert result.iloc[0]["start_time"] == 1001


def test_variable_length_incomplete_final_heartbeat_is_still_discarded(monkeypatch):
    _mock_peaks(monkeypatch, [3, 7])
    result = HeartbeatSegmentationNeurokit(start_factor=0.5).extract(ecg=_ecg(8), sampling_rate_hz=10).heartbeat_list_

    np.testing.assert_array_equal(result[["start_sample", "end_sample"]], [[1, 5]])


def test_fixed_length_heartbeat_slices_match_neurokit_included_samples(monkeypatch):
    _mock_peaks(monkeypatch, [3, 7])
    included = {"1": pd.DataFrame({"Index": [1, 2, 3, 4]}), "2": pd.DataFrame({"Index": [5, 6, 7, 8]})}
    monkeypatch.setattr(
        "biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit.nk.ecg_segment",
        lambda *args, **kwargs: included,
    )
    ecg = _ecg(9)
    result = HeartbeatSegmentationNeurokit(variable_length=False).extract(ecg=ecg, sampling_rate_hz=10).heartbeat_list_

    np.testing.assert_array_equal(result[["start_sample", "end_sample"]], [[1, 5], [5, 9]])
    for (_, beat), samples in zip(result.iterrows(), included.values()):
        actual_labels = ecg.iloc[beat["start_sample"] : beat["end_sample"]].index
        np.testing.assert_array_equal(actual_labels, 1000 + samples["Index"].to_numpy())


def test_fixed_length_heartbeat_output_feeds_icg_consumers_at_terminal_boundary(monkeypatch):
    _mock_peaks(monkeypatch, [2, 8])
    monkeypatch.setattr(
        "biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit.nk.ecg_segment",
        lambda *args, **kwargs: {
            "1": pd.DataFrame({"Index": np.arange(6)}),
            "2": pd.DataFrame({"Index": np.arange(6, 12)}),
        },
    )
    ecg = _ecg(12)
    heartbeats = (
        HeartbeatSegmentationNeurokit(variable_length=False).extract(ecg=ecg, sampling_rate_hz=10).heartbeat_list_
    )
    icg = pd.DataFrame({"icg_der": [0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0]}, index=ecg.index)

    c_points = (
        CPointExtractionScipyFindPeaks(handle_missing_events="raise")
        .extract(icg=icg, heartbeats=heartbeats, sampling_rate_hz=10)
        .points_
    )
    b_points = (
        BPointExtractionArbol2017IsoelectricCrossings(handle_missing_events="raise")
        .extract(icg=icg, heartbeats=heartbeats, c_points=c_points, sampling_rate_hz=10)
        .points_
    )

    np.testing.assert_array_equal(c_points["c_point_sample"], [3, 9])
    np.testing.assert_array_equal(b_points["b_point_sample"], [2, 8])
    assert heartbeats.iloc[-1]["end_sample"] == len(icg)


@pytest.mark.parametrize("positions", [[-1, 0, 1, 2], [5, 6, 7, 8]])
def test_fixed_length_padded_heartbeats_are_discarded(monkeypatch, positions):
    _mock_peaks(monkeypatch, [1, 5])
    monkeypatch.setattr(
        "biopsykit.signals.ecg.segmentation._heartbeat_segmentation_neurokit.nk.ecg_segment",
        lambda *args, **kwargs: {"1": pd.DataFrame({"Index": positions})},
    )
    result = (
        HeartbeatSegmentationNeurokit(variable_length=False).extract(ecg=_ecg(8), sampling_rate_hz=10).heartbeat_list_
    )

    assert result.empty
    assert list(result.columns) == [
        "start_time",
        "start_sample",
        "end_sample",
        "r_peak_sample",
        "rr_interval_sample",
        "rr_interval_ms",
    ]


@pytest.mark.parametrize("algorithm", [QPeakExtractionMartinez2004Neurokit, QPeakExtractionSciPyFindPeaksNeurokit])
@pytest.mark.parametrize("last_q", [10, 20])
def test_q_peak_assignment_includes_start_and_excludes_end(monkeypatch, algorithm, last_q):
    monkeypatch.setattr("neurokit2.ecg_delineate", lambda *args, **kwargs: (None, {"ECG_Q_Peaks": [0, last_q]}))
    heartbeats = pd.DataFrame(
        {"start_sample": [0, 10], "end_sample": [10, 20], "r_peak_sample": [5, 15]},
        index=pd.Index([0, 1], name="heartbeat_id"),
    ).astype("Int64")

    result = (
        algorithm(handle_missing_events="ignore")
        .extract(ecg=_ecg(20), heartbeats=heartbeats, sampling_rate_hz=10)
        .points_
    )

    assert result.loc[0, "q_peak_sample"] == 0
    if last_q == 10:
        assert result.loc[1, "q_peak_sample"] == 10
    else:
        assert pd.isna(result.loc[1, "q_peak_sample"])
