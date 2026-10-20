"""Functions for computing endpoints from epoch-level sleep/wake predictions."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from biopsykit.sleep._epoch import resolve_epoch_length
from biopsykit.utils.dtypes import SleepEndpointDataFrame, SleepEndpointDict, _SleepEndpointDataFrame


def compute_sleep_endpoints(
    sleep_wake: pd.DataFrame,
    bed_interval: Sequence[str | int | float | pd.Timestamp | np.datetime64],
    *,
    epoch_length: float | None = None,
) -> SleepEndpointDict:
    """Compute sleep endpoints using explicit epoch coverage.

    Each row covers [index, index + epoch_length), with ``sleep_wake=1`` for sleep and 0 for wake.
    Epoch length is in seconds, while all output durations and latencies are in minutes. Datetime indices
    contain epoch-start timestamps; consecutive integer indices count epochs, independent of epoch duration.
    Index names do not affect computation. Gaps, duplicates, missing/nonbinary predictions, and bed intervals
    extending outside observed coverage are rejected. Epochs overlapping a bed boundary are clipped to it.

    Endpoints:

    * ``date``: recording date, assigned to the previous night when bed start is before 06:00; 0 for integer indices.
    * ``sleep_onset``: start of the first sleep interval, clipped to bed start.
    * ``wake_onset``: exclusive end of the final sleep interval, clipped to bed end.
    * ``total_sleep_duration``: first-to-last sleep span, including intervening wake, in minutes.
    * ``net_sleep_duration``: covered sleep time in minutes.
    * ``time_in_bed``: duration of the bed interval in minutes.
    * ``sleep_efficiency``: 100 * net_sleep_duration / time_in_bed.
    * ``sleep_onset_latency`` / ``getup_latency``: time before first sleep / after final sleep, in minutes.
    * ``wake_after_sleep_onset``: wake time between first and final sleep, in minutes.
    * ``sleep_bouts`` / ``wake_bouts``: DataFrames with inclusive ``start`` and exclusive ``end`` boundaries.
      Wake bouts include only wake between first and final sleep.
    * ``number_wake_bouts``: number of those intervening wake bouts.
    * ``bed_interval_start`` / ``bed_interval_end``: inclusive/exclusive bed boundaries.
    * ``epoch_length_s``: duration of each input epoch in seconds.

    Datetime scalar endpoints are strings; bout boundaries retain their datetime type. Integer-index endpoints
    and bouts remain in epoch-index units, including fractional boundaries when the bed interval clips an epoch.
    No observed sleep (including empty input) returns an empty dictionary.

    Parameters
    ----------
    sleep_wake : pandas.DataFrame
        Predictions in a ``sleep_wake`` column, indexed by epoch starts.
    bed_interval : sequence of two timestamps or numbers
        Half-open bed interval [start, end). It must lie within observed epoch coverage.
        For datetime input, naive boundaries are interpreted in the index timezone. Integer-index boundaries
        use epoch-index units: four epochs starting at 0 have coverage [0, 4).
    epoch_length : float, optional
        Positive epoch duration in seconds. If omitted, use ``sleep_wake.attrs['epoch_length_s']`` or infer it
        from a regular DatetimeIndex. Single timestamps without frequency and integer indices require an
        explicit value or metadata. Explicit values, metadata, and timestamp spacing must agree.

    Returns
    -------
    SleepEndpointDict
        Endpoint values in the units described above, or {} when there is no sleep.

    Notes
    -----
    See :doc:`/guides/sleep_epochs` for coverage rules, endpoint definitions, and migration of existing callers.
    """
    if sleep_wake.empty:
        return {}
    epoch_length = resolve_epoch_length(sleep_wake, epoch_length)
    if "sleep_wake" not in sleep_wake or not sleep_wake["sleep_wake"].isin([0, 1]).all():
        raise ValueError("sleep_wake must contain only binary 0/1 predictions without missing values.")
    if len(bed_interval) != 2:
        raise ValueError("bed_interval must contain a start and exclusive end.")
    datetime_index = isinstance(sleep_wake.index, pd.DatetimeIndex)
    step = pd.Timedelta(seconds=epoch_length) if datetime_index else 1
    bed_start, bed_end = _bed_boundaries(bed_interval, sleep_wake.index)
    if bed_start >= bed_end:
        raise ValueError("bed_interval end must be after its start.")
    if bed_start < sleep_wake.index[0] or bed_end > sleep_wake.index[-1] + step:
        raise ValueError("bed_interval extends outside observed epoch coverage.")

    epochs = pd.DataFrame(
        {"start": sleep_wake.index, "end": sleep_wake.index + step, "sleep_wake": sleep_wake["sleep_wake"].to_numpy()}
    )
    epochs = epochs.loc[(epochs["start"] < bed_end) & (epochs["end"] > bed_start)].copy()
    epochs["start"] = epochs["start"].clip(lower=bed_start)
    epochs["end"] = epochs["end"].clip(upper=bed_end)
    asleep = epochs["sleep_wake"].eq(1)
    if not asleep.any():
        return {}
    sleep_onset = epochs.loc[asleep, "start"].iloc[0]
    wake_onset = epochs.loc[asleep, "end"].iloc[-1]
    sleep_period = epochs.loc[(epochs["start"] >= sleep_onset) & (epochs["end"] <= wake_onset)].copy()
    durations = epochs["end"] - epochs["start"]
    durations = durations.dt.total_seconds() / 60 if datetime_index else durations * epoch_length / 60
    net_sleep_duration = float(durations.loc[asleep].sum())

    sleep_period["block"] = sleep_period["sleep_wake"].diff().ne(0).cumsum()
    bouts = sleep_period.groupby("block").agg(
        start=("start", "first"), end=("end", "last"), sleep_wake=("sleep_wake", "first")
    )
    sleep_bouts = bouts.loc[bouts["sleep_wake"].eq(1), ["start", "end"]].reset_index(drop=True)
    wake_bouts = bouts.loc[bouts["sleep_wake"].eq(0), ["start", "end"]].reset_index(drop=True)
    time_in_bed = _minutes(bed_end - bed_start, epoch_length, datetime_index)
    total_sleep_duration = _minutes(wake_onset - sleep_onset, epoch_length, datetime_index)
    wake_after_sleep_onset = float(durations.loc[sleep_period.index].loc[sleep_period["sleep_wake"].eq(0)].sum())
    date = 0
    if datetime_index:
        date = bed_start.normalize()
        if bed_start.hour < 6:
            date = (bed_start - pd.DateOffset(days=1)).normalize()

    return {
        "date": str(date) if datetime_index else date,
        "sleep_onset": str(sleep_onset) if datetime_index else sleep_onset,
        "wake_onset": str(wake_onset) if datetime_index else wake_onset,
        "total_sleep_duration": total_sleep_duration,
        "net_sleep_duration": net_sleep_duration,
        "time_in_bed": time_in_bed,
        "epoch_length_s": epoch_length,
        "bed_interval_start": str(bed_start) if datetime_index else bed_start,
        "bed_interval_end": str(bed_end) if datetime_index else bed_end,
        "sleep_efficiency": 100 * net_sleep_duration / time_in_bed,
        "sleep_onset_latency": _minutes(sleep_onset - bed_start, epoch_length, datetime_index),
        "getup_latency": _minutes(bed_end - wake_onset, epoch_length, datetime_index),
        "wake_after_sleep_onset": wake_after_sleep_onset,
        "sleep_bouts": sleep_bouts,
        "wake_bouts": wake_bouts,
        "number_wake_bouts": len(wake_bouts),
    }


def _minutes(duration, epoch_length: float, datetime_index: bool) -> float:
    return float(duration / pd.Timedelta(minutes=1) if datetime_index else duration * epoch_length / 60)


def _bed_boundaries(bed_interval, index):
    if not isinstance(index, pd.DatetimeIndex):
        if not all(
            isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) for value in bed_interval
        ):
            raise ValueError("Integer epoch indices require numeric bed boundaries.")
        return tuple(bed_interval)
    boundaries = []
    for boundary in bed_interval:
        value = pd.Timestamp(boundary)
        if pd.isna(value):
            raise ValueError("Bed boundaries must be valid timestamps.")
        if value.tzinfo is None and index.tz is not None:
            value = value.tz_localize(index.tz)
        elif value.tzinfo is not None and index.tz is None:
            raise ValueError("Bed boundaries and epoch timestamps must have compatible timezones.")
        elif value.tzinfo is not None:
            value = value.tz_convert(index.tz)
        boundaries.append(value)
    return tuple(boundaries)


def endpoints_as_df(sleep_endpoints: SleepEndpointDict) -> SleepEndpointDataFrame | None:
    """Convert sleep endpoints into a dataframe.

    Parameters
    ----------
    sleep_endpoints : SleepEndpointDict
        Endpoint dictionary, including sleep/wake bout tables.

    Returns
    -------
    SleepEndpointDataFrame or None
        One-row dataframe preserving duration units and bout boundaries. Empty/no-sleep endpoints return None.
    """
    if not sleep_endpoints:
        return None
    sleep_endpoints = sleep_endpoints.copy()
    sleep_bouts = [tuple(v) for v in sleep_endpoints.pop("sleep_bouts").to_numpy().tolist()]
    wake_bouts = [tuple(v) for v in sleep_endpoints.pop("wake_bouts").to_numpy().tolist()]
    index = pd.to_datetime(pd.Index([sleep_endpoints.pop("date")], name="date"))
    df = pd.DataFrame(sleep_endpoints, index=index).fillna(value=np.nan)
    df["sleep_bouts"] = None
    df["wake_bouts"] = None
    df.at[df.index[0], "sleep_bouts"] = sleep_bouts  # noqa: PD008
    df.at[df.index[0], "wake_bouts"] = wake_bouts  # noqa: PD008
    return _SleepEndpointDataFrame(df)
