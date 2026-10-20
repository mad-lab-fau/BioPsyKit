"""Small, hand-calculated examples for CFT baseline and feature computations."""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from tpcp import make_action_safe

from biopsykit.protocols import CftFeatureExtraction
from biopsykit.utils.exceptions import FeatureComputationError


def _heart_rates(rates, *, times=None, origin=pd.Timedelta(0), subject="A", ids=None):
    """Create heartbeat-level data without signal-processing or external fixtures."""
    if times is None:
        times = range(len(rates))
    if ids is None:
        ids = range(len(rates))
    index = pd.MultiIndex.from_product([[subject], ids], names=["subject", "heartbeat_id"])
    return pd.DataFrame(
        {"r_peak_time": origin + pd.to_timedelta(times, unit="s"), "heart_rate_bpm": rates}, index=index
    )


def _algorithm(**kwargs):
    """Use short explicit phases, raising on missing baselines unless a test overrides it."""
    parameters = {
        "group_level": "subject",
        "duration_baseline_sec": 2,
        "duration_cft_sec": 5,
        "error_handling": "raise",
    }
    return CftFeatureExtraction(**(parameters | kwargs))


def test_baseline_regression_from_the_api_audit():
    """The original example must return a 100 bpm baseline, rather than 73.33 bpm from exposure."""
    data = _heart_rates([100.0, 100.0, 100.0, 60.0, 60.0, 60.0, 60.0, 60.0])

    result = _algorithm(duration_cft_sec=4).extract(data).output_.iloc[0]

    assert result["baseline_hr"] == pytest.approx(100.0)
    assert result["mean_hr_bpm"] == pytest.approx(70.0)


@pytest.mark.parametrize("origin", [pd.Timedelta(0), pd.Timestamp("2026-10-20T12:00:00Z")])
def test_known_baseline_and_all_cft_features(origin):
    """A quadratic exposure gives exact baseline, bradycardia, latency, and polynomial values."""
    # Baseline: [100, 100]. Exposure: HR(t) = 100 - 6*t + t**2 for t=0,...,4.
    # The large value at t=7 is at the exposure end and must be excluded.
    data = _heart_rates([100.0, 100.0, 100.0, 95.0, 92.0, 91.0, 92.0, 10000.0], origin=origin)
    result = _algorithm().extract(data).output_.iloc[0]

    expected = {
        "baseline_hr": 100.0,
        "cft_start_idx": 2,
        "mean_hr_bpm": 94.0,
        "mean_brady_bpm": -6.0,
        "mean_brady_percent": -6.0,
        "peak_brady_idx": 3,
        "peak_brady_bpm": -9.0,
        "peak_brady_percent": -9.0,
        "peak_brady_latency": 3.0,
        "peak_brady_slope": -3.0,
        "onset_idx": 3,
        "onset_latency": 3.0,
        "onset_hr": 91.0,
        "onset_hr_percent": 9.0,
        "onset_slope": -3.0,
        "poly_fit_a0": 100.0,
        "poly_fit_a1": -6.0,
        "poly_fit_a2": 1.0,
    }
    for name, value in expected.items():
        assert result[name] == pytest.approx(value), name
    assert result["peak_brady_time"] == origin + pd.Timedelta(seconds=5)
    assert result["onset_time"] == origin + pd.Timedelta(seconds=5)


def test_baseline_offset_excludes_leading_data_and_the_first_exposure_beat():
    """The baseline window is [2,4), with the first exposure beat at t=4."""
    data = _heart_rates(
        [999.0, 999.0, 100.0, 100.0, 80.0, 75.0, 70.0, 75.0, 80.0, 999.0],
        ids=[10 * (i + 1) for i in range(10)],
    )
    result = _algorithm(start_baseline_sec=2).extract(data).output_.iloc[0]

    assert result["baseline_hr"] == pytest.approx(100.0)
    assert result["mean_hr_bpm"] == pytest.approx(76.0)
    assert result["mean_brady_percent"] == pytest.approx(-24.0)
    assert result["peak_brady_bpm"] == pytest.approx(-30.0)
    assert result["peak_brady_latency"] == pytest.approx(2.0)
    assert result["cft_start_idx"] == 4


def test_latencies_and_polynomial_use_configured_exposure_start_between_beats():
    """An exposure beginning at t=2 must not restart its clock at the first beat at t=2.5."""
    data = _heart_rates(
        [100.0, 100.0, 100.0, 95.0, 92.0, 91.0, 92.0, 10000.0],
        times=[0, 1, 2.5, 3.5, 4.5, 5.5, 6.5, 7],
    )
    result = _algorithm().extract(data).output_.iloc[0]

    assert result["peak_brady_latency"] == pytest.approx(3.5)
    assert result["onset_latency"] == pytest.approx(3.5)
    assert result["peak_brady_slope"] == pytest.approx(-9.0 / 3.5)
    assert result["onset_slope"] == pytest.approx(-9.0 / 3.5)
    # 100 - 6*(t-0.5) + (t-0.5)**2 = 103.25 - 7*t + t**2.
    assert result["poly_fit_a0"] == pytest.approx(103.25)
    assert result["poly_fit_a1"] == pytest.approx(-7.0)
    assert result["poly_fit_a2"] == pytest.approx(1.0)


@pytest.mark.parametrize("group_level", ["subject", ["subject"], ["subject", "condition"]])
def test_groups_use_their_own_baseline_and_time_origin(group_level):
    """The same relative response at different absolute rates and times stays independent."""
    rates = np.array([100.0, 100.0, 100.0, 95.0, 92.0, 91.0, 92.0, 10000.0])
    first = _heart_rates(rates).assign(condition="C").set_index("condition", append=True)
    second = _heart_rates(0.8 * rates, subject="B", origin=pd.Timedelta(seconds=100))
    second = second.assign(condition="C").set_index("condition", append=True)
    result = _algorithm(group_level=group_level).extract(pd.concat([first, second])).output_

    assert result["baseline_hr"].to_numpy() == pytest.approx([100.0, 80.0])
    assert result["mean_hr_bpm"].to_numpy() == pytest.approx([94.0, 75.2])
    assert result["mean_brady_percent"].to_numpy() == pytest.approx([-6.0, -6.0])
    assert result["peak_brady_latency"].to_numpy() == pytest.approx([3.0, 3.0])


@pytest.mark.parametrize("error_handling", ["warn", "ignore", "raise"])
def test_zero_length_baseline_uses_first_exposure_beat_or_raises(error_handling):
    """Leading data must not become the fallback baseline when exposure starts later."""
    data = _heart_rates([999.0, 999.0, 100.0, 95.0, 92.0, 91.0, 92.0, 999.0])
    algorithm = _algorithm(start_baseline_sec=2, duration_baseline_sec=0, error_handling=error_handling)
    if error_handling == "raise":
        with pytest.raises(FeatureComputationError, match="Baseline is 0"):
            algorithm.extract(data)
    elif error_handling == "warn":
        with pytest.warns(UserWarning, match="Baseline is 0"):
            result = algorithm.extract(data).output_.iloc[0]
        assert result["baseline_hr"] == pytest.approx(100.0)
    else:
        with warnings.catch_warnings(record=True) as caught:
            result = algorithm.extract(data).output_.iloc[0]
        assert not caught
        assert result["baseline_hr"] == pytest.approx(100.0)


def test_recovery_extends_only_the_polynomial_fit():
    """Recovery supplies the third polynomial point but cannot change exposure-only bradycardia."""
    data = _heart_rates([100.0, 100.0, 100.0, 95.0, 92.0, 10000.0])
    result = _algorithm(duration_cft_sec=2, duration_recovery_sec=1).extract(data).output_.iloc[0]

    assert result["mean_hr_bpm"] == pytest.approx(97.5)
    assert result["peak_brady_bpm"] == pytest.approx(-5.0)
    assert result["poly_fit_a0"] == pytest.approx(100.0)
    assert result["poly_fit_a1"] == pytest.approx(-6.0)
    assert result["poly_fit_a2"] == pytest.approx(1.0)


def test_no_three_consecutive_bradycardic_beats_preserves_missing_onset_fields():
    """A missing onset is a result with NaNs rather than a missing set of columns."""
    data = _heart_rates([100.0, 100.0, 100.0, 95.0, 100.0, 95.0, 100.0, 999.0])
    result = _algorithm().extract(data).output_.iloc[0]

    for name in ["onset_time", "onset_latency", "onset_idx", "onset_hr", "onset_hr_percent", "onset_slope"]:
        assert pd.isna(result[name]), name


def test_peak_at_exposure_start_has_undefined_slope():
    """A peak at zero latency must not cause division by zero."""
    result = _algorithm().extract(_heart_rates([100.0, 100.0, 70.0, 80.0, 90.0, 100.0, 110.0, 999.0])).output_.iloc[0]

    assert result["peak_brady_latency"] == 0.0
    assert result["peak_brady_bpm"] == pytest.approx(-30.0)
    assert pd.isna(result["peak_brady_slope"])


def test_selected_features_allow_short_exposure_without_polynomial_fit():
    """Feature selection is a tpcp parameter and does not require unused polynomial inputs."""
    data = _heart_rates([100.0, 100.0, 80.0, 90.0, 999.0])
    algorithm = _algorithm(duration_cft_sec=2, cft_parameter=("mean_bradycardia",))
    before = deepcopy(algorithm.get_params(deep=False))
    data_before = data.copy(deep=True)

    result = make_action_safe(CftFeatureExtraction.extract)(algorithm, data).output_.iloc[0]

    assert result["baseline_hr"] == pytest.approx(100.0)
    assert result["mean_hr_bpm"] == pytest.approx(85.0)
    assert "poly_fit_a0" not in result.index
    assert algorithm.get_params(deep=False) == before
    pd.testing.assert_frame_equal(data, data_before)


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        (_heart_rates([100.0, 100.0, 100.0]), {"duration_baseline_sec": 10}, "Baseline interval"),
        (
            _heart_rates([100.0, 100.0, 90.0, 80.0], times=[0, 1, 4, 5]),
            {"start_baseline_sec": 2, "duration_baseline_sec": 1},
            "Baseline interval",
        ),
        (_heart_rates([100.0, 100.0, 80.0], times=[0, 1, 10]), {"duration_cft_sec": 2}, "CFT interval"),
        (_heart_rates([100.0, 100.0, 80.0, 90.0, 999.0]), {"duration_cft_sec": 2}, "three"),
    ],
)
def test_unusable_intervals_raise_domain_errors(data, kwargs, message):
    """Missing baseline/exposure data and underdetermined fits fail explicitly."""
    with pytest.raises(FeatureComputationError, match=message):
        _algorithm(**kwargs).extract(data)


@pytest.mark.parametrize("baseline", [0.0, np.nan])
def test_nonfinite_or_nonpositive_baseline_is_rejected(baseline):
    """Relative bradycardia must not silently divide by zero or propagate an invalid baseline."""
    data = _heart_rates([baseline, baseline, 100.0, 95.0, 92.0, 91.0, 92.0, 999.0])

    with pytest.raises(FeatureComputationError, match="Baseline heart rate"):
        _algorithm().extract(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_baseline_sec": -1},
        {"duration_baseline_sec": -1},
        {"duration_cft_sec": 0},
        {"duration_recovery_sec": -1},
        {"duration_cft_sec": np.nan},
        {"error_handling": "invalid"},
        {"cft_parameter": ("invalid",)},
    ],
)
def test_invalid_configuration_is_rejected_before_computing_features(kwargs):
    """Invalid timing or feature settings must not silently select a different phase."""
    with pytest.raises(ValueError):
        _algorithm(**kwargs).extract(_heart_rates([100.0] * 8))
