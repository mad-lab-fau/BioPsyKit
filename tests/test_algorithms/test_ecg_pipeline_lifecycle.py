"""Stage boundaries, child execution, and rerun contracts for ECG pipelines."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from tpcp import Algorithm, get_results

from biopsykit.signals.ecg import EcgProcessingPipeline
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction
from biopsykit.signals.ecg.hrv_extraction import HrvExtraction
from biopsykit.signals.ecg.outlier_correction import BaseRPeakOutlierDetection, RPeakOutlierDetectionChain
from biopsykit.signals.ecg.preprocessing import BaseEcgPreprocessing
from biopsykit.utils.exceptions import EcgProcessingError


class Preprocessor(BaseEcgPreprocessing):
    def __init__(self, handle_missing_events="warn"):
        self.handle_missing_events = handle_missing_events

    def clean(self, *, ecg, sampling_rate_hz):
        ecg["ecg"] += 1
        self.ecg_clean_ = ecg
        return self


class Extractor(BaseEcgExtraction):
    def __init__(self, handle_missing_events="warn", rich_output=True, fail=False):
        self.handle_missing_events = handle_missing_events
        self.rich_output = rich_output
        self.fail = fail

    def extract(self, *, ecg, sampling_rate_hz):
        if self.fail:
            raise RuntimeError("extraction failed")
        self.points_ = pd.DataFrame({"r_peak_sample": [2, 5, 8]}, index=pd.Index([10, 20, 30], name="heartbeat_id"))
        if self.rich_output:
            self.ecg_processed_ = ecg.assign(ecg_quality=0.9, r_peak_indicator=ecg.index.isin([2, 5, 8]))
        # A downstream child must not be able to mutate preprocessing output.
        ecg.iloc[0, 0] = -999
        return self


class Detector(BaseRPeakOutlierDetection):
    def __init__(self, handle_missing_events="warn"):
        self.handle_missing_events = handle_missing_events

    def detect_outlier(self, *, ecg, rpeaks, sampling_rate_hz):
        self.observed_ecg_ = ecg.copy(deep=True)
        self.points_ = pd.DataFrame({"is_outlier": [False, True, False]}, index=rpeaks.index)
        ecg.iloc[0, 0] = -999
        rpeaks.iloc[0, 0] = -999
        return self


class Correction(Algorithm):
    _action_methods = "correct_outlier"

    def __init__(self, handle_missing_events="warn", fail=False):
        self.handle_missing_events = handle_missing_events
        self.fail = fail

    def correct_outlier(self, *, ecg, rpeaks, outlier_detection_results):
        if self.fail:
            raise RuntimeError("correction failed")
        rpeaks.loc[outlier_detection_results["is_outlier"], "r_peak_sample"] = 6
        self.points_ = rpeaks
        ecg["heart_rate_bpm"] = 60.0
        self.ecg_processed_ = ecg
        return self


class HrvCorrection(Algorithm):
    _action_methods = "correct_outlier"

    def __init__(self, handle_missing_events="warn", fail=False):
        self.handle_missing_events = handle_missing_events
        self.fail = fail

    def correct_outlier(self, *, rpeaks, sampling_rate_hz):
        if self.fail:
            raise RuntimeError("HRV correction failed")
        rpeaks["r_peak_sample"] += 1
        self.points_ = rpeaks
        return self


class HrvExtractor(Algorithm):
    _action_methods = "extract"

    def __init__(self, handle_missing_events="warn", fail=False):
        self.handle_missing_events = handle_missing_events
        self.fail = fail

    def extract(self, *, rpeaks, sampling_rate_hz):
        if self.fail:
            raise RuntimeError("HRV extraction failed")
        self.observed_ = rpeaks.copy(deep=True)
        self.hrv_extracted_ = pd.DataFrame({"mean_sample": [rpeaks["r_peak_sample"].mean()]})
        rpeaks.iloc[0, 0] = -999
        return self


@pytest.fixture
def datapoint():
    return SimpleNamespace(ecg=pd.DataFrame({"ecg": list(range(10))}), sampling_rate_hz=1000)


@pytest.fixture
def pipeline():
    return EcgProcessingPipeline(
        preprocessing_algo=Preprocessor(),
        r_peak_algo=Extractor(),
        outlier_detection_algo=Detector(),
        outlier_correction_algo=Correction(),
        hrv_r_peak_correction_algo=HrvCorrection(),
        hrv_extraction_algo=HrvExtractor(),
    )


def test_stage_outputs_preserve_inputs_annotations_and_hrv_source(pipeline, datapoint):
    original = datapoint.ecg.copy(deep=True)
    assert pipeline.safe_run(datapoint) is pipeline
    pd.testing.assert_frame_equal(datapoint.ecg, original)
    pd.testing.assert_frame_equal(pipeline.ecg_raw_, original)
    pd.testing.assert_frame_equal(pipeline.ecg_clean_, original + 1)
    assert pipeline.ecg_processed_.columns.tolist() == ["ecg", "ecg_quality", "r_peak_indicator", "heart_rate_bpm"]
    pd.testing.assert_frame_equal(pipeline.outlier_detection_algo_.observed_ecg_, pipeline.r_peak_algo_.ecg_processed_)
    assert pipeline.ecg_processed_["ecg"].tolist() == list(range(1, 11))
    assert pipeline.rpeaks_raw_["r_peak_sample"].tolist() == [2, 5, 8]
    assert pipeline.rpeaks_["r_peak_sample"].tolist() == [2, 6, 8]
    assert pipeline.rpeaks_hrv_["r_peak_sample"].tolist() == [3, 7, 9]
    pd.testing.assert_frame_equal(pipeline.rpeaks_hrv_, pipeline.hrv_extraction_algo_.observed_)
    assert pipeline.hrv_extracted_.iloc[0, 0] == pytest.approx(19 / 3)


def test_executed_children_are_independent_and_configured_children_remain_unexecuted(pipeline, datapoint):
    pipeline.safe_run(datapoint)
    for name in (
        "preprocessing_algo",
        "r_peak_algo",
        "outlier_detection_algo",
        "outlier_correction_algo",
        "hrv_r_peak_correction_algo",
        "hrv_extraction_algo",
    ):
        configured = getattr(pipeline, name)
        executed = getattr(pipeline, name + "_")
        assert executed is not configured
        assert get_results(executed)
        assert not get_results(configured)
    pipeline.r_peak_algo_.points_.iloc[0, 0] = -100
    pipeline.preprocessing_algo_.ecg_clean_.iloc[0, 0] = -100
    assert pipeline.rpeaks_raw_.iloc[0, 0] == 2
    assert pipeline.ecg_clean_.iloc[0, 0] == 1


@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_factory_preserves_missing_event_policy(policy):
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline(handle_missing_events=policy)
    assert pipeline.handle_missing_events == policy
    assert pipeline.r_peak_algo.handle_missing_events == policy


@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_policy_is_forwarded_to_clones_and_nested_detectors(pipeline, datapoint, policy):
    pipeline.set_params(
        handle_missing_events=policy,
        outlier_detection_algo=RPeakOutlierDetectionChain(
            [("nested", RPeakOutlierDetectionChain([("leaf", Detector())]))]
        ),
    )
    pipeline.safe_run(datapoint)
    for name in (
        "preprocessing_algo",
        "r_peak_algo",
        "outlier_correction_algo",
        "hrv_r_peak_correction_algo",
        "hrv_extraction_algo",
    ):
        assert getattr(pipeline, name).handle_missing_events == "warn"
        assert getattr(pipeline, name + "_").handle_missing_events == policy
    configured_leaf = pipeline.outlier_detection_algo.detectors[0][1].detectors[0][1]
    executed_leaf = pipeline.outlier_detection_algo_.detectors_[0][1].detectors_[0][1]
    assert configured_leaf.handle_missing_events == "warn"
    assert not get_results(configured_leaf)
    assert executed_leaf.handle_missing_events == policy


@pytest.mark.parametrize("correct_hrv", [False, True])
@pytest.mark.parametrize("extract_hrv", [False, True])
def test_optional_hrv_steps_replace_previous_outputs(pipeline, datapoint, correct_hrv, extract_hrv):
    pipeline.run(datapoint)
    previous_child = pipeline.r_peak_algo_
    pipeline.set_params(
        hrv_r_peak_correction_algo=HrvCorrection() if correct_hrv else None,
        hrv_extraction_algo=HrvExtractor() if extract_hrv else None,
        outlier_detection_algo=None,
    )
    pipeline.safe_run(datapoint)
    assert pipeline.r_peak_algo_ is not previous_child
    assert pipeline.outlier_detection_algo_ is None
    assert not pipeline.outlier_mask_["is_outlier"].any()
    assert (pipeline.hrv_r_peak_correction_algo_ is not None) == correct_hrv
    assert (pipeline.hrv_extraction_algo_ is not None) == extract_hrv
    assert (pipeline.hrv_extracted_ is not None) == extract_hrv
    if correct_hrv or extract_hrv:
        assert pipeline.rpeaks_hrv_["r_peak_sample"].tolist() == ([3, 6, 9] if correct_hrv else [2, 5, 8])
    else:
        assert pipeline.rpeaks_hrv_ is None


@pytest.mark.parametrize(
    "step", ["r_peak_algo", "outlier_correction_algo", "hrv_r_peak_correction_algo", "hrv_extraction_algo"]
)
def test_failed_stage_clears_all_results_and_recovery_uses_fresh_children(pipeline, datapoint, step):
    pipeline.run(datapoint)
    old_child = pipeline.r_peak_algo_
    pipeline.set_params(**{step + "__fail": True})
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.run(datapoint)
    assert get_results(pipeline) == {}
    pipeline.set_params(**{step + "__fail": False})
    pipeline.safe_run(datapoint)
    assert pipeline.r_peak_algo_ is not old_child
    assert pipeline.rpeaks_raw_["r_peak_sample"].tolist() == [2, 5, 8]


@pytest.mark.parametrize("invalid", ["policy", "ecg", "empty", "rate"])
def test_input_validation_failure_also_clears_previous_results(pipeline, datapoint, invalid):
    pipeline.run(datapoint)
    if invalid == "policy":
        pipeline.set_params(handle_missing_events="invalid")
    elif invalid == "ecg":
        del datapoint.ecg
    elif invalid == "empty":
        datapoint.ecg = datapoint.ecg.iloc[:0]
    else:
        del datapoint.sampling_rate_hz
    with pytest.raises((EcgProcessingError, ValueError)):
        pipeline.run(datapoint)
    assert get_results(pipeline) == {}


def test_points_only_extractor_falls_back_to_preprocessed_signal(pipeline, datapoint):
    pipeline.set_params(r_peak_algo__rich_output=False)
    pipeline.safe_run(datapoint)
    pd.testing.assert_frame_equal(pipeline.outlier_detection_algo_.observed_ecg_, pipeline.ecg_clean_)
    assert pipeline.ecg_processed_.columns.tolist() == ["ecg", "heart_rate_bpm"]


def test_real_pipeline_keeps_neurokit_annotations_and_exposes_hrv_input():
    path = Path(__file__).parent.parent / "test_data" / "pep" / "pep_test_ecg.csv"
    datapoint = SimpleNamespace(ecg=pd.read_csv(path, index_col=0, parse_dates=True), sampling_rate_hz=1000)
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline()
    pipeline.set_params(hrv_extraction_algo=HrvExtraction("time"))
    pipeline.safe_run(datapoint)
    assert pipeline.ecg_clean_.columns.tolist() == ["ecg"]
    assert {"ecg_quality", "r_peak_indicator", "heart_rate_bpm", "r_peak_outlier"} <= set(pipeline.ecg_processed_)
    pd.testing.assert_frame_equal(pipeline.ecg_clean_, pipeline.preprocessing_algo_.ecg_clean_)
    pd.testing.assert_frame_equal(pipeline.ecg_processed_, pipeline.outlier_correction_algo_.ecg_processed_)
    pd.testing.assert_frame_equal(pipeline.rpeaks_raw_, pipeline.r_peak_algo_.points_)
    pd.testing.assert_frame_equal(pipeline.rpeaks_hrv_, pipeline.hrv_r_peak_correction_algo_.points_)
    expected = HrvExtraction("time").extract(rpeaks=pipeline.rpeaks_hrv_, sampling_rate_hz=1000).hrv_extracted_
    pd.testing.assert_frame_equal(pipeline.hrv_extracted_, expected)
