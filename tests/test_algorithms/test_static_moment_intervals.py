"""Regression tests for empty static intervals and existing endpoint conventions."""

import numpy as np
import pandas as pd
import pytest

from biopsykit.signals.imu.static_moment_detection import _find_static_sequences, find_static_moments
from biopsykit.utils.array_handling import bool_array_to_start_end_array


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (np.array([], dtype=bool), []),
        (np.array([False, False, False]), []),
        (np.zeros(3, dtype=int), []),
        (np.zeros(3, dtype=float), []),
        (np.array([True]), [[0, 1]]),
        (np.array([True, True, True]), [[0, 3]]),
        (np.array([True, False, False]), [[0, 1]]),
        (np.array([False, False, True]), [[2, 3]]),
        (np.array([False, True, False]), [[1, 2]]),
        (np.array([False, True, True, False, True]), [[1, 3], [4, 5]]),
        (np.array([0.0, 1.0, 1.0, 0.0]), [[1, 3]]),
    ],
)
def test_boolean_intervals_have_stable_shape_and_exclusive_ends(mask, expected):
    """Empty outputs have no placeholder row; detected runs remain valid positional slices."""
    result = bool_array_to_start_end_array(mask)

    np.testing.assert_array_equal(result, np.asarray(expected, dtype=int).reshape(-1, 2))
    assert result.shape == (len(expected), 2)
    assert np.issubdtype(result.dtype, np.integer)
    reconstructed = np.zeros(len(mask), dtype=bool)
    for start, end in result:
        assert 0 <= start < end <= len(mask)
        reconstructed[start:end] = True
    np.testing.assert_array_equal(reconstructed, mask.astype(bool))


@pytest.mark.parametrize("mask", [np.array([0, 2, 0]), np.array([0.0, np.nan])])
def test_boolean_intervals_still_reject_non_boolean_values(mask):
    """Correcting the empty result must retain the existing input-value validation."""
    with pytest.raises(ValueError, match="Input must be boolean"):
        bool_array_to_start_end_array(mask)


@pytest.mark.parametrize("length", [0, 100])
def test_static_detection_without_static_samples_returns_an_empty_integer_table(length):
    """An empty signal and an active signal both yield zero detected intervals."""
    data = pd.DataFrame(np.ones((length, 3)), columns=["acc_x", "acc_y", "acc_z"])

    result = find_static_moments(data, threshold=0.0, window_samples=10, metric="mean")

    expected = pd.DataFrame({"start": pd.Series(dtype=int), "end": pd.Series(dtype=int)})
    pd.testing.assert_frame_equal(result, expected)


def test_internal_static_detection_without_static_samples_returns_no_intervals():
    """The shared converter must not create a spurious interval for the internal detector."""
    result = _find_static_sequences(np.ones((100, 3)), window_length=10, inactive_signal_th=0.0, metric="mean")

    assert result.shape == (0, 2)
    assert np.issubdtype(result.dtype, np.integer)


def test_nonempty_static_detection_retains_existing_inclusive_ends():
    """The empty-result fix must not migrate the public detector's endpoint convention."""
    result = find_static_moments(pd.DataFrame(np.zeros((100, 3))), threshold=0.0, window_samples=10, metric="mean")

    expected = pd.DataFrame({"start": [0], "end": [99]})
    pd.testing.assert_frame_equal(result, expected)
