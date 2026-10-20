"""Regression tests for immutable ECG algorithm configuration during execution."""

from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from tpcp import get_results, make_action_safe

from biopsykit.signals.ecg import EcgProcessingPipeline
from biopsykit.signals.ecg.event_extraction import RPeakExtractionNeurokit
from biopsykit.signals.ecg.hrv_extraction import HrvExtraction
from biopsykit.signals.ecg.outlier_correction import RPeakOutlierCorrection
from biopsykit.signals.ecg.preprocessing import EcgPreprocessingNeurokit


@pytest.fixture(scope="module")
def ecg_datapoint():
    """Load the same real recording used by the event-extraction regression tests."""
    path = Path(__file__).parent.parent / "test_data" / "pep" / "pep_test_ecg.csv"
    return SimpleNamespace(ecg=pd.read_csv(path, index_col=0, parse_dates=True), sampling_rate_hz=1000)


@pytest.fixture(scope="module")
def ecg_and_rpeaks(ecg_datapoint):
    """Prepare signal and peak inputs independently of the processing pipeline."""
    ecg = (
        EcgPreprocessingNeurokit(method="neurokit")
        .clean(ecg=ecg_datapoint.ecg, sampling_rate_hz=ecg_datapoint.sampling_rate_hz)
        .ecg_clean_
    )
    peaks = RPeakExtractionNeurokit().extract(ecg=ecg, sampling_rate_hz=1000).points_
    return ecg, peaks


def _hrv_backend_result(peaks, sampling_rate, *, hrv_type):
    """Keep alias/parameter tests independent of individual HRV backend requirements."""
    return pd.DataFrame({hrv_type: [1.0]})


@pytest.mark.parametrize(
    "hrv_types",
    ["default", "all", "time", "hrv_time", ["time", "frequency"], ("time", "frequency")],
)
def test_hrv_extraction_preserves_constructor_parameters(hrv_types, monkeypatch, ecg_and_rpeaks):
    """Resolving aliases must not overwrite the constructor's scalar or sequence."""
    monkeypatch.setattr(
        HrvExtraction,
        "HRV_METHODS",
        {name: partial(_hrv_backend_result, hrv_type=name) for name in HrvExtraction.HRV_TYPES},
    )
    _, peaks = ecg_and_rpeaks
    algorithm = HrvExtraction(hrv_types)
    parameters_before = deepcopy(algorithm.get_params(deep=False))

    result = make_action_safe(HrvExtraction.extract)(algorithm, rpeaks=peaks, sampling_rate_hz=1000)

    assert result is algorithm
    assert algorithm.get_params(deep=False) == parameters_before
    assert algorithm.hrv_types is hrv_types


@pytest.mark.parametrize("hrv_types", [None, 42, "invalid", ["time", "invalid"]])
def test_invalid_hrv_types_leave_configuration_unchanged(hrv_types, ecg_and_rpeaks):
    """Failed validation must not partially normalize a constructor parameter."""
    _, peaks = ecg_and_rpeaks
    algorithm = HrvExtraction(hrv_types)
    parameters_before = deepcopy(algorithm.get_params(deep=False))

    with pytest.raises(TypeError):
        algorithm.extract(rpeaks=peaks, sampling_rate_hz=1000)

    assert algorithm.get_params(deep=False) == parameters_before


@pytest.mark.parametrize(
    ("imputation_type", "imputation_params"),
    [("linear_interpolation", None), ("moving_average", None), ("moving_average", {"window_size": 3})],
)
def test_outlier_correction_preserves_constructor_parameters(imputation_type, imputation_params, ecg_and_rpeaks):
    """Defaults and caller-owned dictionaries remain unchanged during correction."""
    ecg, peaks = ecg_and_rpeaks
    mask = pd.DataFrame({"is_outlier": False}, index=peaks.index)
    mask.iloc[2, 0] = True
    algorithm = RPeakOutlierCorrection(imputation_type=imputation_type, imputation_params=imputation_params)
    parameters_before = deepcopy(algorithm.get_params(deep=False))

    algorithm.correct_outlier(ecg=ecg, rpeaks=peaks, outlier_detection_results=mask)

    assert algorithm.get_params(deep=False) == parameters_before
    assert algorithm.imputation_params is imputation_params
    assert algorithm.points_["r_peak_outlier"].sum() == 1


@pytest.mark.parametrize("hrv_types", ["time", "default"])
def test_pipeline_safe_run_keeps_configured_hrv_algorithms_unexecuted(hrv_types, ecg_datapoint):
    """The real pipeline runs clones and remains safe across repeated executions."""
    pipeline = EcgProcessingPipeline.get_default_biopsykit_pipeline()
    pipeline.set_params(hrv_extraction_algo=HrvExtraction(hrv_types))
    configured_correction = pipeline.hrv_r_peak_correction_algo
    configured_extraction = pipeline.hrv_extraction_algo

    result = pipeline.safe_run(ecg_datapoint)

    assert result is pipeline
    assert not get_results(configured_correction)
    assert not get_results(configured_extraction)
    assert configured_extraction.hrv_types == hrv_types
    first_hrv = result.hrv_extracted_.copy(deep=True)
    first_peaks = result.rpeaks_.copy(deep=True)

    pipeline.safe_run(ecg_datapoint)

    pd.testing.assert_frame_equal(pipeline.hrv_extracted_, first_hrv)
    pd.testing.assert_frame_equal(pipeline.rpeaks_, first_peaks)
    assert pipeline.hrv_r_peak_correction_algo is configured_correction
    assert pipeline.hrv_extraction_algo is configured_extraction
    assert not get_results(configured_correction)
    assert not get_results(configured_extraction)
