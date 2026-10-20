"""Known-duration examples for variable sleep epochs and pipeline propagation."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from biopsykit.sleep.sleep_endpoints import compute_sleep_endpoints, endpoints_as_df
from biopsykit.sleep.sleep_processing_pipeline import predict_pipeline_actigraph
from biopsykit.sleep.sleep_wake_detection import SleepWakeDetection
from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke import ColeKripke
from biopsykit.sleep.sleep_wake_detection.algorithms.sadeh import Sadeh
from biopsykit.sleep.sleep_wake_detection.algorithms.sazonov import Sazonov
from biopsykit.sleep.sleep_wake_detection.algorithms.scripps_clinic import ScrippsClinic
from biopsykit.sleep.sleep_wake_detection.algorithms.webster import Webster


def _predictions(values, seconds=30, name=None, *, start="2026-10-20 22:00:00", tz="Europe/Berlin"):
    index = pd.date_range(start, periods=len(values), freq=pd.Timedelta(seconds=seconds), tz=tz, name=name)
    return pd.DataFrame({"sleep_wake": values}, index=index)


def _bed(data, seconds):
    return [data.index[0], data.index[-1] + pd.Timedelta(seconds=seconds)]


@pytest.mark.parametrize("seconds", [0.5, 10, 15, 30, 60, 90, 120])
@pytest.mark.parametrize("index_name", [None, "time", "date", "sleep_wake"])
def test_all_sleep_uses_epoch_coverage_and_has_zero_boundary_latencies(seconds, index_name):
    data = _predictions([1, 1, 1, 1], seconds, index_name)
    before = data.copy(deep=True)
    result = compute_sleep_endpoints(data, _bed(data, seconds), epoch_length=seconds)

    assert result["net_sleep_duration"] == pytest.approx(4 * seconds / 60)
    assert result["total_sleep_duration"] == pytest.approx(4 * seconds / 60)
    assert result["time_in_bed"] == pytest.approx(4 * seconds / 60)
    assert result["sleep_onset_latency"] == result["getup_latency"] == 0
    assert result["sleep_efficiency"] == pytest.approx(100)
    assert result["wake_after_sleep_onset"] == pytest.approx(0)
    assert result["number_wake_bouts"] == 0
    assert result["wake_bouts"].empty
    assert result["sleep_bouts"].iloc[0]["end"] == _bed(data, seconds)[1]
    assert pd.Timestamp(result["wake_onset"]) == _bed(data, seconds)[1]
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("seconds", [10, 30, 60, 90])
def test_mixed_night_has_hand_calculated_endpoints(seconds):
    # Wake, sleep x2, wake, sleep x2, wake x2: 8 epochs in bed, 5-epoch sleep span, 4 asleep.
    data = _predictions([0, 1, 1, 0, 1, 1, 0, 0], seconds)
    result = compute_sleep_endpoints(data, _bed(data, seconds))

    assert result["epoch_length_s"] == seconds
    assert result["time_in_bed"] == pytest.approx(8 * seconds / 60)
    assert result["total_sleep_duration"] == pytest.approx(5 * seconds / 60)
    assert result["net_sleep_duration"] == pytest.approx(4 * seconds / 60)
    assert result["sleep_onset_latency"] == pytest.approx(seconds / 60)
    assert result["getup_latency"] == pytest.approx(2 * seconds / 60)
    assert result["wake_after_sleep_onset"] == pytest.approx(seconds / 60)
    assert result["sleep_efficiency"] == pytest.approx(50)
    assert result["number_wake_bouts"] == 1
    assert result["wake_bouts"].iloc[0].tolist() == [data.index[3], data.index[4]]
    assert result["sleep_bouts"].to_numpy().tolist() == [[data.index[1], data.index[3]], [data.index[4], data.index[6]]]


def test_bed_interval_is_exclusive_and_clips_partial_boundary_epochs():
    data = _predictions([1, 0, 1, 0], 30)
    start = data.index[0] + pd.Timedelta(seconds=15)
    end = data.index[2] + pd.Timedelta(seconds=15)
    result = compute_sleep_endpoints(data, [start, end])

    assert result["time_in_bed"] == result["total_sleep_duration"] == 1
    assert result["net_sleep_duration"] == 0.5
    assert result["wake_after_sleep_onset"] == 0.5
    assert result["sleep_efficiency"] == 50
    assert result["sleep_bouts"].iloc[-1]["end"] == end
    assert result["getup_latency"] == 0
    # The sleep epoch beginning at the bed end must not enter the interval.
    assert compute_sleep_endpoints(data, [data.index[1], data.index[2]]) == {}


@pytest.mark.parametrize("seconds", [10, 30, 60, 90])
def test_integer_indices_count_epochs_but_durations_are_minutes(seconds):
    data = pd.DataFrame({"sleep_wake": [0, 1, 1, 0, 1, 0]}, index=pd.RangeIndex(10, 16, name="epoch"))
    result = compute_sleep_endpoints(data, [10, 16], epoch_length=seconds)

    assert result["date"] == 0
    assert result["sleep_onset"] == 11
    assert result["wake_onset"] == 15
    assert result["time_in_bed"] == pytest.approx(6 * seconds / 60)
    assert result["total_sleep_duration"] == pytest.approx(4 * seconds / 60)
    assert result["net_sleep_duration"] == pytest.approx(3 * seconds / 60)
    assert result["getup_latency"] == pytest.approx(seconds / 60)
    assert result["sleep_efficiency"] == 50


def test_integer_partial_epoch_boundaries_are_supported():
    result = compute_sleep_endpoints(pd.DataFrame({"sleep_wake": [1, 0, 1]}), [0.5, 2.5], epoch_length=30)
    assert result["net_sleep_duration"] == 0.5
    assert result["time_in_bed"] == 1
    assert result["wake_onset"] == 2.5


@pytest.mark.parametrize("values", [[], [0, 0, 0]])
def test_no_sleep_returns_empty_endpoints_and_converts_safely(values):
    data = _predictions(values)
    result = (
        compute_sleep_endpoints(data, [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-21")])
        if not values
        else compute_sleep_endpoints(data, _bed(data, 30))
    )
    assert result == {}
    assert endpoints_as_df(result) is None


@pytest.mark.parametrize("source", ["frequency", "explicit", "metadata"])
def test_single_epoch_requires_known_duration(source):
    data = _predictions([1], 30)
    kwargs = {}
    if source != "frequency":
        data.index = pd.DatetimeIndex(data.index.values, tz="UTC")
    if source == "explicit":
        kwargs["epoch_length"] = 30
    elif source == "metadata":
        data.attrs["epoch_length_s"] = 30
    assert compute_sleep_endpoints(data, _bed(data, 30), **kwargs)["net_sleep_duration"] == 0.5


def test_ambiguous_coverage_requires_an_epoch_length():
    for data, bounds in [
        (pd.DataFrame({"sleep_wake": [1]}), [0, 1]),
        (
            pd.DataFrame({"sleep_wake": [1]}, index=pd.DatetimeIndex([pd.Timestamp("2026-10-20")])),
            [pd.Timestamp("2026-10-20"), pd.Timestamp("2026-10-20 00:00:30")],
        ),
    ]:
        with pytest.raises(ValueError, match="epoch_length"):
            compute_sleep_endpoints(data, bounds)


@pytest.mark.parametrize("epoch_length", [0, -1, np.nan, np.inf, True])
def test_invalid_epoch_durations_are_rejected(epoch_length):
    data = _predictions([1, 1])
    with pytest.raises(ValueError, match="epoch_length"):
        compute_sleep_endpoints(data, _bed(data, 30), epoch_length=epoch_length)


@pytest.mark.parametrize(
    "invalid", ["gap", "duplicate", "reversed", "metadata", "explicit", "missing", "nonbinary", "coverage"]
)
def test_invalid_or_conflicting_coverage_and_predictions_are_rejected(invalid):
    data = _predictions([1, 0, 1], 30)
    bounds = _bed(data, 30)
    kwargs = {"epoch_length": 30}
    if invalid == "gap":
        data = data.iloc[[0, 2]]
    elif invalid == "duplicate":
        data.index = data.index[[0, 0, 2]]
    elif invalid == "reversed":
        data = data.iloc[::-1]
    elif invalid == "metadata":
        data.attrs["epoch_length_s"] = 60
    elif invalid == "explicit":
        kwargs["epoch_length"] = 60
    elif invalid == "missing":
        data.iloc[1, 0] = np.nan
    elif invalid == "nonbinary":
        data.iloc[1, 0] = 2
    else:
        bounds[1] += pd.Timedelta(seconds=30)
    with pytest.raises(ValueError):
        compute_sleep_endpoints(data, bounds, **kwargs)


def test_timezone_and_recording_date_are_preserved_across_dst():
    data = _predictions([1] * 8, 30 * 60, start="2026-10-25 00:00:00")
    result = compute_sleep_endpoints(data, _bed(data, 1800))
    assert result["net_sleep_duration"] == 240
    assert result["sleep_onset_latency"] == result["getup_latency"] == 0
    assert pd.Timestamp(result["date"]).date() == pd.Timestamp("2026-10-24").date()


def test_endpoint_dataframe_preserves_computed_units_and_bouts():
    data = _predictions([0, 1, 1, 0], 30, "date")
    endpoints = compute_sleep_endpoints(data, _bed(data, 30))
    result = endpoints_as_df(endpoints)
    assert result.iloc[0]["net_sleep_duration"] == result.iloc[0]["total_sleep_duration"] == 1
    assert result.iloc[0]["time_in_bed"] == 2
    assert result.iloc[0]["epoch_length_s"] == 30
    assert result.iloc[0]["sleep_bouts"] == [(data.index[1], data.index[3])]
    assert "sleep_bouts" in endpoints


@pytest.mark.parametrize(
    "algorithm", ["cole_kripke", "cole_kripke_old", "sadeh", "sazonov", "scripps_clinic", "webster"]
)
@pytest.mark.parametrize("seconds", [30, 60])
def test_classifier_configuration_and_metadata_reach_endpoint_extraction(algorithm, seconds):
    data = _predictions([0] * 80, seconds).rename(columns={"sleep_wake": "activity"})
    model = SleepWakeDetection(algorithm_type=algorithm, epoch_length=seconds, rescore_data=False)
    predictions = model.predict(data)
    assert model.epoch_length == seconds
    assert predictions.attrs["epoch_length_s"] == seconds
    assert predictions.columns.tolist() == ["sleep_wake"]
    result = compute_sleep_endpoints(predictions, _bed(predictions, seconds))
    if result:
        assert result["epoch_length_s"] == seconds
        assert result["time_in_bed"] == pytest.approx(80 * seconds / 60)


@pytest.mark.parametrize("algorithm", [ColeKripke, Sadeh, Sazonov, ScrippsClinic, Webster])
def test_direct_classifier_predict_override_does_not_mutate_configuration(algorithm):
    data = _predictions([0] * 80, 30).rename(columns={"sleep_wake": "activity"})
    model = algorithm(epoch_length=60)
    result = model.predict(data, epoch_length=30, rescore_data=False)
    assert model.epoch_length == 60
    assert result.attrs["epoch_length_s"] == 30


@pytest.mark.parametrize(
    ("algorithm", "seconds"),
    [
        ("cole_kripke", 60),
        ("cole_kripke_old", 60),
        ("sadeh", 60),
        ("sazonov", 30),
        ("scripps_clinic", 30),
        ("webster", 60),
    ],
)
def test_default_classifier_epochs_are_attached_to_array_predictions(algorithm, seconds):
    model = SleepWakeDetection(algorithm_type=algorithm)
    predictions = model.predict(np.zeros(80), rescore_data=False)
    assert model.epoch_length == seconds
    assert predictions.attrs["epoch_length_s"] == seconds
    assert predictions.columns.tolist() == ["sleep_wake"]
    result = compute_sleep_endpoints(predictions, [0, 80])
    assert result["time_in_bed"] == pytest.approx(80 * seconds / 60)


def test_unsupported_rescoring_epoch_is_rejected_explicitly():
    model = SleepWakeDetection(epoch_length=10)
    with pytest.raises(ValueError, match="rescore_data=False"):
        model.predict(np.zeros(80))
    assert model.predict(np.zeros(80), rescore_data=False).attrs["epoch_length_s"] == 10


@pytest.mark.parametrize("seconds", [10, 30, 60])
def test_actigraph_pipeline_propagates_actual_input_epoch_length(seconds):
    data = _predictions([0] * 80, seconds, "date").rename(columns={"sleep_wake": "activity"})
    result = predict_pipeline_actigraph(data, "cole_kripke", _bed(data, seconds), rescore_data=False)
    assert result["sleep_wake_prediction"].attrs["epoch_length_s"] == seconds
    assert result["sleep_endpoints"]["time_in_bed"] == pytest.approx(80 * seconds / 60)
    assert result["sleep_endpoints"]["net_sleep_duration"] == pytest.approx(80 * seconds / 60)


@pytest.mark.parametrize("seconds", [30, 60])
@pytest.mark.parametrize("extra_seconds", [0, 5])
def test_acceleration_pipeline_aggregates_complete_epochs_without_padding(monkeypatch, seconds, extra_seconds):
    module = import_module("biopsykit.sleep.sleep_processing_pipeline.sleep_processing_pipeline")
    data = _predictions([0] * (10 * seconds + extra_seconds), 1, start="2026-10-20 22:00:00.250").rename(
        columns={"sleep_wake": "acc_z"}
    )

    class FakeWear:
        def __init__(self, **kwargs):
            pass

        def predict(self, data):
            return data

        def get_major_wear_block(self, data):
            return [data.index[0], data.index[-1]]

        def cut_to_wear_block(self, data, block):
            return data

    class FakeCounts:
        def __init__(self, *args):
            pass

        def calculate(self, data):
            padded_index = pd.date_range(data.index[0].floor("s"), periods=len(data) + 1, freq="s")
            return pd.DataFrame({"activity": np.zeros(len(data) + 1)}, index=padded_index)

    class FakeRest:
        def __init__(self, **kwargs):
            pass

        def predict(self, data):
            return pd.DataFrame({"start": [data.index[0]], "end": [data.index[-1] + pd.Timedelta(seconds=1)]})

    monkeypatch.setattr(module, "WearDetection", FakeWear)
    monkeypatch.setattr(module, "ActivityCounts", FakeCounts)
    monkeypatch.setattr(module, "RestPeriods", FakeRest)
    result = module.predict_pipeline_acceleration(
        data, sampling_rate=1, convert_to_g=False, epoch_length=seconds, rescore_data=False
    )

    assert len(result["activity_counts"]) == 10
    assert result["activity_counts"].index[0] == data.index[0]
    assert result["sleep_endpoints"]["net_sleep_duration"] == pytest.approx(10 * seconds / 60)
    assert result["sleep_endpoints"]["epoch_length_s"] == seconds
