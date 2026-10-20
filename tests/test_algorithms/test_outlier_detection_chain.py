"""Contract and integration tests for explicit, configurable outlier chains."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid
from tpcp import get_results, make_action_safe

from biopsykit.signals.ecg import EcgProcessingPipeline
from biopsykit.signals.ecg.event_extraction import RPeakExtractionNeurokit
from biopsykit.signals.ecg.outlier_correction import (
    BaseRPeakOutlierDetection,
    RPeakOutlierCorrection,
    RPeakOutlierDetectionChain,
    RPeakOutlierDetectionPhysiological,
    RPeakOutlierDetectionQuality,
)
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit


class FlagDetector(BaseRPeakOutlierDetection):
    calls: ClassVar[list[str]] = []

    def __init__(self, flagged=(), tag="detector", invalid=None, mutate=False):
        self.flagged = flagged
        self.tag = tag
        self.invalid = invalid
        self.mutate = mutate
        super().__init__()

    def detect_outlier(self, *, ecg, rpeaks, sampling_rate_hz):
        self.calls.append(self.tag)
        self.observed_ = rpeaks.copy(deep=True)
        self.sampling_rate_hz_ = sampling_rate_hz
        result = pd.DataFrame({"is_outlier": rpeaks.index.isin(self.flagged)}, index=rpeaks.index)
        if self.mutate:
            rpeaks.iloc[0, 0] = -999
            ecg.iloc[0, 0] = -999
        if self.invalid == "order":
            result = result.iloc[::-1]
        elif self.invalid == "ids":
            result.index = pd.RangeIndex(len(result))
        elif self.invalid == "integer":
            result = result.astype(int)
        elif self.invalid == "nan":
            result = result.astype("boolean")
            result.iloc[0, 0] = pd.NA
        elif self.invalid == "column":
            result.columns = ["flag"]
        elif self.invalid == "duplicate_column":
            result = pd.concat([result, result], axis=1)
        elif self.invalid == "series":
            result = result.iloc[:, 0]
        self.points_ = result
        return self


@pytest.fixture
def inputs():
    ecg = pd.DataFrame({"ecg": np.arange(30, dtype=float)})
    peaks = pd.DataFrame(
        {"rr_interval_ms": [500.0, 1000.0, 2000.0], "r_peak_quality": [0.9, 0.1, 0.8]},
        index=pd.Index([10, 20, 40], name="heartbeat_id"),
    )
    return {"ecg": ecg, "rpeaks": peaks, "sampling_rate_hz": 1000}


def test_named_children_support_native_nested_discovery_and_assignment():
    child = RPeakOutlierDetectionPhysiological()
    steps = [("physiological", child)]
    chain = RPeakOutlierDetectionChain(steps)
    assert chain.detectors is steps
    assert chain.get_params()["detectors__physiological"] is child
    assert chain.get_params()["detectors__physiological__hr_thresholds"] == (45, 200)

    chain.set_params(detectors__physiological__hr_thresholds=(40, 180))
    assert child.hr_thresholds == (40, 180)
    replacement = RPeakOutlierDetectionQuality(0.2)
    chain.set_params(detectors__physiological=replacement)
    assert chain.get_params()["detectors__physiological"] is replacement


@pytest.mark.parametrize("container", [list, tuple])
def test_cloning_preserves_configuration_without_sharing_child_instances(container):
    chain = RPeakOutlierDetectionChain(container([("physiological", RPeakOutlierDetectionPhysiological())]))
    clone = chain.clone()
    clone.set_params(detectors__physiological__hr_thresholds=(40, 180))
    assert chain.get_params()["detectors__physiological__hr_thresholds"] == (45, 200)
    assert clone.detectors[0][1] is not chain.detectors[0][1]


def test_real_detectors_combine_known_masks_with_or_and_preserve_ids(inputs):
    chain = RPeakOutlierDetectionChain(
        [("physiological", RPeakOutlierDetectionPhysiological()), ("quality", RPeakOutlierDetectionQuality())]
    )
    result = make_action_safe(RPeakOutlierDetectionChain.detect_outlier)(chain, **inputs)
    assert result is chain
    expected = pd.DataFrame(
        {"physiological": [False, False, True], "quality": [False, True, False]}, index=inputs["rpeaks"].index
    )
    pd.testing.assert_frame_equal(chain.outlier_masks_, expected)
    pd.testing.assert_frame_equal(
        chain.points_, pd.DataFrame({"is_outlier": [False, True, True]}, index=expected.index)
    )
    for (name, configured), (executed_name, executed) in zip(chain.detectors, chain.detectors_):
        assert name == executed_name
        assert configured is not executed
        assert not get_results(configured)
        pd.testing.assert_series_equal(executed.points_["is_outlier"], expected[name], check_names=False)


def test_detectors_run_in_order_on_independent_original_inputs(inputs):
    first = FlagDetector((10,), tag="first", mutate=True)
    second = FlagDetector((40,), tag="second")
    chain = RPeakOutlierDetectionChain([("first", first), ("second", second)])
    before = {key: value.copy(deep=True) for key, value in inputs.items() if isinstance(value, pd.DataFrame)}
    FlagDetector.calls.clear()

    chain.detect_outlier(**inputs)

    assert FlagDetector.calls == ["first", "second"]
    for _, executed in chain.detectors_:
        pd.testing.assert_frame_equal(executed.observed_, before["rpeaks"])
        assert executed.sampling_rate_hz_ == 1000
    for name, original in before.items():
        pd.testing.assert_frame_equal(inputs[name], original)
    assert not get_results(first)
    assert not get_results(second)


def test_empty_chain_returns_aligned_false_mask(inputs):
    chain = RPeakOutlierDetectionChain().detect_outlier(**inputs)
    pd.testing.assert_frame_equal(
        chain.points_, pd.DataFrame(False, index=inputs["rpeaks"].index, columns=["is_outlier"])
    )
    assert chain.outlier_masks_.shape == (3, 0)
    assert chain.detectors_ == []


@pytest.mark.parametrize("multi_index", [False, True])
def test_input_index_order_and_levels_are_preserved(inputs, multi_index):
    index = pd.Index([40, 10, 20], name="heartbeat_id")
    if multi_index:
        index = pd.MultiIndex.from_arrays([["A"] * 3, index], names=["subject", "heartbeat_id"])
    inputs["rpeaks"].index = index
    chain = RPeakOutlierDetectionChain([("physiological", RPeakOutlierDetectionPhysiological())])
    chain.detect_outlier(**inputs)
    pd.testing.assert_index_equal(chain.points_.index, index)
    pd.testing.assert_index_equal(chain.outlier_masks_.index, index)
    assert chain.points_["is_outlier"].tolist() == [False, False, True]


@pytest.mark.parametrize("detectors", [(), [("physiological", RPeakOutlierDetectionPhysiological())]])
def test_empty_peak_input_keeps_a_boolean_empty_schema(inputs, detectors):
    inputs["rpeaks"] = inputs["rpeaks"].iloc[:0]
    chain = RPeakOutlierDetectionChain(detectors).detect_outlier(**inputs)
    assert chain.points_.empty
    assert chain.points_.columns.tolist() == ["is_outlier"]
    assert chain.points_["is_outlier"].dtype == bool
    pd.testing.assert_index_equal(chain.points_.index, inputs["rpeaks"].index)


def test_nested_chain_exposes_parameters_and_combines_results(inputs):
    chain = RPeakOutlierDetectionChain(
        [("inner", RPeakOutlierDetectionChain([("flag", FlagDetector((10,)))])), ("last", FlagDetector((40,)))]
    )
    assert chain.get_params()["detectors__inner__detectors__flag__flagged"] == (10,)
    chain.set_params(detectors__inner__detectors__flag__flagged=(20,))
    chain.detect_outlier(**inputs)
    assert chain.points_["is_outlier"].tolist() == [False, True, True]


def test_repeated_execution_replaces_results_and_failed_execution_clears_them(inputs):
    chain = RPeakOutlierDetectionChain([("flag", FlagDetector((10,)))]).detect_outlier(**inputs)
    old_executed = chain.detectors_[0][1]
    chain.set_params(detectors__flag__flagged=(40,))
    chain.detect_outlier(**inputs)
    assert chain.points_["is_outlier"].tolist() == [False, False, True]
    assert chain.detectors_[0][1] is not old_executed
    assert not get_results(chain.detectors[0][1])
    chain.set_params(detectors__flag__invalid="ids")
    with pytest.raises(ValueError, match="index"):
        chain.detect_outlier(**inputs)
    assert not get_results(chain)


@pytest.mark.parametrize("invalid", ["order", "ids", "integer", "nan", "column", "duplicate_column", "series"])
def test_invalid_child_masks_are_rejected(inputs, invalid):
    chain = RPeakOutlierDetectionChain([("invalid", FlagDetector(invalid=invalid))])
    with pytest.raises(ValueError, match="invalid"):
        chain.detect_outlier(**inputs)


@pytest.mark.parametrize(
    "detectors",
    [
        None,
        "invalid",
        [FlagDetector()],
        [("", FlagDetector())],
        [("a__b", FlagDetector())],
        [("a", FlagDetector()), ("a", FlagDetector())],
        [("a", object())],
        [(1, FlagDetector())],
    ],
)
def test_invalid_chain_configuration_is_rejected(inputs, detectors):
    chain = RPeakOutlierDetectionChain(detectors)
    with pytest.raises((TypeError, ValueError)):
        chain.detect_outlier(**inputs)


def test_duplicate_beat_ids_are_rejected(inputs):
    inputs["rpeaks"].index = pd.Index([10, 10, 20], name="heartbeat_id")
    with pytest.raises(ValueError, match="unique"):
        RPeakOutlierDetectionChain().detect_outlier(**inputs)


def test_nullable_boolean_masks_are_supported(inputs):
    class NullableDetector(FlagDetector):
        def detect_outlier(self, **kwargs):
            super().detect_outlier(**kwargs)
            self.points_ = self.points_.astype("boolean")
            return self

    chain = RPeakOutlierDetectionChain([("nullable", NullableDetector((20,)))]).detect_outlier(**inputs)
    assert chain.points_["is_outlier"].tolist() == [False, True, False]
    assert chain.points_["is_outlier"].dtype == bool


def test_parameter_grid_can_tune_named_detector_thresholds(inputs):
    chain = RPeakOutlierDetectionChain([("physiological", RPeakOutlierDetectionPhysiological())])
    results = []
    for params in ParameterGrid({"detectors__physiological__hr_thresholds": [(45, 200), (20, 200)]}):
        results.append(chain.clone().set_params(**params).detect_outlier(**inputs).points_["is_outlier"].sum())
    assert results == [1, 0]


@pytest.fixture
def datapoint():
    path = Path(__file__).parent / "../test_data/pep/pep_test_ecg.csv"
    return SimpleNamespace(ecg=pd.read_csv(path, index_col=0, parse_dates=True), sampling_rate_hz=1000)


def test_default_factory_exposes_native_named_detector_parameters():
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline(compute_hrv=False)
    assert isinstance(pipeline.outlier_detection_algo, RPeakOutlierDetectionChain)
    path = "outlier_detection_algo__detectors__physiological__hr_thresholds"
    assert pipeline.get_params()[path] == (45, 200)
    pipeline.set_params(**{path: (40, 180)})
    assert pipeline.clone().get_params()[path] == (40, 180)
    assert [name for name, _ in pipeline.outlier_detection_algo.detectors] == [
        "berntson",
        "physiological",
        "correlation",
        "quality",
        "rr_statistics",
        "rr_diff_statistics",
    ]


@pytest.mark.parametrize("detector", [RPeakOutlierDetectionPhysiological(), RPeakOutlierDetectionChain(), None])
def test_normal_pipeline_constructor_accepts_leaf_chain_or_none(detector):
    parameters = EcgProcessingPipeline.get_default_biopsykit_pipeline(compute_hrv=False).get_params(deep=False)
    parameters["outlier_detection_algo"] = detector
    pipeline = EcgProcessingPipeline(**parameters)
    assert pipeline.outlier_detection_algo is detector
    assert pipeline.get_params(deep=False)["outlier_detection_algo"] is detector
    pipeline.clone()


def test_single_detector_and_singleton_chain_produce_identical_pipeline_results(datapoint):
    single = EcgProcessingPipeline.get_default_biopsykit_pipeline(compute_hrv=False)
    single.set_params(outlier_detection_algo=RPeakOutlierDetectionPhysiological())
    chain = single.clone().set_params(
        outlier_detection_algo=RPeakOutlierDetectionChain([("physiological", RPeakOutlierDetectionPhysiological())])
    )

    single.safe_run(datapoint)
    chain.safe_run(datapoint)

    pd.testing.assert_frame_equal(single.rpeaks_, chain.rpeaks_)
    pd.testing.assert_frame_equal(single.outlier_mask_, chain.outlier_mask_)
    assert single.outlier_detection_algo_ is not single.outlier_detection_algo
    assert not get_results(single.outlier_detection_algo)
    single.set_params(outlier_detection_algo__hr_thresholds=(40, 180))
    assert single.get_params()["outlier_detection_algo__hr_thresholds"] == (40, 180)


@pytest.mark.parametrize("detector", [None, RPeakOutlierDetectionChain()])
def test_pipeline_without_detectors_uses_a_false_mask(datapoint, detector):
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline(compute_hrv=False)
    pipeline.set_params(outlier_detection_algo=detector).safe_run(datapoint)
    assert not pipeline.outlier_mask_["is_outlier"].any()
    assert len(pipeline.rpeaks_) == len(pipeline.rpeaks_raw_)


def test_pipeline_rejects_implicit_lists_and_invalid_single_masks(datapoint):
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline(compute_hrv=False)
    pipeline.set_params(outlier_detection_algo=[RPeakOutlierDetectionPhysiological()])
    with pytest.raises(TypeError, match="explicit RPeakOutlierDetectionChain"):
        pipeline.run(datapoint)
    pipeline.set_params(outlier_detection_algo=FlagDetector(invalid="order"))
    with pytest.raises(ValueError, match="index"):
        pipeline.run(datapoint)


def test_default_chain_preserves_previous_detection_and_correction_results(datapoint):
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline(compute_hrv=False)
    ecg = EcgPreprocessingNeurokit(method="neurokit").clean(ecg=datapoint.ecg, sampling_rate_hz=1000).ecg_clean_
    peaks = RPeakExtractionNeurokit().extract(ecg=ecg, sampling_rate_hz=1000).points_
    masks = []
    for _, configured in pipeline.outlier_detection_algo.detectors:
        detector = configured.clone()
        detector.detect_outlier(ecg=ecg, rpeaks=peaks, sampling_rate_hz=1000)
        masks.append(detector.points_)
    expected_mask = pd.concat(masks, axis=1).any(axis=1).to_frame("is_outlier")
    expected_mask.index = peaks.index.copy()
    correction = RPeakOutlierCorrection()
    correction.correct_outlier(ecg=ecg, rpeaks=peaks, outlier_detection_results=masks)

    before = {
        name: deepcopy(detector.get_params(deep=False)) for name, detector in pipeline.outlier_detection_algo.detectors
    }
    pipeline.safe_run(datapoint)

    pd.testing.assert_frame_equal(pipeline.outlier_mask_, expected_mask)
    pd.testing.assert_frame_equal(pipeline.rpeaks_, correction.points_)
    assert {
        name: detector.get_params(deep=False) for name, detector in pipeline.outlier_detection_algo.detectors
    } == before
    assert not get_results(pipeline.outlier_detection_algo)
    assert len(pipeline.outlier_detection_algo_.detectors_) == 6
