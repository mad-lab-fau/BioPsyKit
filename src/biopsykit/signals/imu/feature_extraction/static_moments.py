"""Extract features from static moments of IMU data."""

import numpy as np
import pandas as pd
from scipy.stats import skew

from biopsykit.utils.time import tz


def compute_features(
    data: pd.DataFrame,
    static_moments: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    index: int | str | None = None,
    timezone: str | None = None,
    *,
    sampling_rate_hz: float,
) -> pd.DataFrame:
    """Compute features based on frequency and duration of static moments in given input signal.

    This function computes the following features:

    * ``sm_number``: number of static moments in data
    * ``sm_max``: maximum duration of static moments, i.e., longest duration
    * ``sm_max_position``: location of the beginning of the longest static moment in the input data normalized to
      ``[0, 1]`` where 0 = ``start`` and 1 = ``end``
    * ``sm_median``: median duration of static moments
    * ``sm_mean``: mean duration of static moments
    * ``sm_std``: standard deviation of static moment durations
    * ``sm_skewness``: skewness of static moment durations

    The features are computed on all detected static moments and on static moments lasting at least
    60 seconds (suffix ``_60``). Intervals use integer sample positions [start, end), including their start
    and excluding their end. Duration is sample coverage: ``(end - start) / sampling_rate_hz``.
    Intervals are clipped to the requested time window; intervals outside it are omitted.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data
    static_moments : :class:`~pandas.DataFrame`
        dataframe with integer ``start`` and exclusive ``end`` positions relative to ``data``
    start : :class:`~pandas.Timestamp` or str, optional
        start timestamp in input data for feature extraction or ``None`` to set start index to the first index in
        ``data``. All samples *before* ``start`` will not be used for feature extraction.
    end : :class:`~pandas.Timestamp` or str, optional
        exclusive end timestamp for feature extraction or ``None`` for one sample period after the last
        timestamp in ``data``. Samples at or after ``end`` are excluded.
    index : int or str, optional
        index label of the resulting dataframe or ``None`` to assign a default label (0)
    timezone : str, optional
        timezone of the recorded data or ``None`` to use default timezone ("Europe/Berlin")
    sampling_rate_hz : float
        Positive sampling rate of regularly sampled data. The datetime index must agree with this rate.
        Irregular recordings must be split or resampled before feature extraction.

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with extracted static moment features. Empty intervals/data produce zero counts and
        NaN statistics; the same applies to an empty subset of moments lasting at least 60 seconds.

    """
    _validate_sampling_rate(data, sampling_rate_hz)
    static_moments = _validate_intervals(static_moments, len(data))
    loc_max_moment_relative = np.nan
    if not data.empty:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Feature extraction requires a DatetimeIndex.")
        start, end = _get_start_end(data, start, end, timezone, sampling_rate_hz)
        if end <= start:
            raise ValueError("The feature extraction end must be after its start.")
        lower, upper = data.index.searchsorted([start, end], side="left")
        static_moments = np.column_stack(
            (np.maximum(static_moments[:, 0], lower), np.minimum(static_moments[:, 1], upper))
        )
        static_moments = static_moments[static_moments[:, 0] < static_moments[:, 1]]

    durations = (static_moments[:, 1] - static_moments[:, 0]) / sampling_rate_hz
    durations_60 = durations[durations >= 60]

    if len(durations):
        loc_max_moment = data.index[static_moments[np.argmax(durations), 0]]
        loc_max_moment_relative = (loc_max_moment - start) / (end - start)

    feature_dict = {"sm_max_position": loc_max_moment_relative}

    for dur, suffix in zip([durations, durations_60], ["", "_60"], strict=False):
        feature_dict[f"sm_number{suffix}"] = len(dur)
        feature_dict[f"sm_max{suffix}"] = np.max(dur) if len(dur) else np.nan
        feature_dict[f"sm_median{suffix}"] = np.median(dur) if len(dur) else np.nan
        feature_dict[f"sm_mean{suffix}"] = np.mean(dur) if len(dur) else np.nan
        feature_dict[f"sm_std{suffix}"] = np.std(dur, ddof=1) if len(dur) > 1 else np.nan
        feature_dict[f"sm_skewness{suffix}"] = skew(dur) if len(dur) > 1 and np.ptp(dur) > 0 else np.nan

    if index is None:
        index = 0
    return pd.DataFrame(feature_dict, index=[index])


def _get_start_end(
    data: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    timezone: str | None,
    sampling_rate_hz: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if timezone is None:
        timezone = tz

    if start is None:
        start = data.index[0]
    if end is None:
        end = data.index[-1] + pd.Timedelta(seconds=1 / sampling_rate_hz)

    start = _to_timestamp(start, timezone)
    end = _to_timestamp(end, timezone)
    return start, end


def _to_timestamp(date: str | pd.Timestamp, timezone: str) -> pd.Timestamp:
    if isinstance(date, str):
        date = pd.Timestamp(date, tz=timezone)
    return date


def static_moment_duration(data: pd.DataFrame, start_end: np.ndarray, *, sampling_rate_hz: float) -> float:
    """Compute sample coverage of a static moment in seconds.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        regularly sampled input data
    start_end : array
        inclusive start and exclusive end positions, satisfying ``0 <= start < end <= len(data)``
    sampling_rate_hz : float
        Positive sampling rate. A datetime index must agree with this rate; irregular recordings are unsupported.

    Returns
    -------
    float
        duration in seconds

    """
    _validate_sampling_rate(data, sampling_rate_hz)
    start, end = _validate_intervals(np.asarray([start_end]), len(data))[0]
    return float((end - start) / sampling_rate_hz)


def mean_orientation(data: pd.DataFrame, static_moments: pd.DataFrame) -> pd.DataFrame:
    """Compute mean orientation of acceleration signal within static moment windows.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data
    static_moments : :class:`~pandas.DataFrame`
        dataframe with integer ``start`` and exclusive ``end`` positions of static moments

    Returns
    -------
    :class:`~pandas.DataFrame`
        mean orientation (x, y, z) of acceleration signal for each static moment window, preserving the input
        interval index. Array inputs use an ``interval_id`` index. Equal-length moments remain separate rows.

    """
    intervals = _validate_intervals(static_moments, len(data))
    index = (
        static_moments.index.copy()
        if isinstance(static_moments, pd.DataFrame)
        else pd.RangeIndex(len(intervals), name="interval_id")
    )
    means = [data.iloc[start:end].mean() for start, end in intervals]
    return pd.DataFrame(means, index=index, columns=data.columns, dtype=float)


def _validate_intervals(intervals: pd.DataFrame | np.ndarray, n_samples: int) -> np.ndarray:
    if isinstance(intervals, pd.DataFrame):
        intervals = intervals[["start", "end"]].to_numpy()
    intervals = np.asarray(intervals)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("Intervals must have shape (n, 2), with start and exclusive end positions.")
    # Accept pandas nullable-integer columns, whose numpy representation can have object dtype.
    if not all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in intervals.flat
    ):
        raise ValueError("Interval positions must be integers.")
    intervals = intervals.astype(int)
    if np.any(intervals[:, 0] < 0) or np.any(intervals[:, 0] >= intervals[:, 1]) or np.any(intervals[:, 1] > n_samples):
        raise ValueError("Intervals must satisfy 0 <= start < end <= len(data).")
    return intervals


def _validate_sampling_rate(data: pd.DataFrame, sampling_rate_hz: float) -> None:
    if (
        isinstance(sampling_rate_hz, (bool, np.bool_))
        or not isinstance(sampling_rate_hz, (int, float, np.integer, np.floating))
        or not np.isfinite(sampling_rate_hz)
        or sampling_rate_hz <= 0
    ):
        raise ValueError("sampling_rate_hz must be finite and positive.")
    if isinstance(data.index, pd.DatetimeIndex):
        deltas_ns = np.diff(data.index.as_unit("ns").asi8)
        if data.index.hasnans or not np.allclose(deltas_ns, 1e9 / sampling_rate_hz, rtol=0, atol=1):
            raise ValueError("DatetimeIndex must be regularly sampled at sampling_rate_hz; split or resample the data.")
