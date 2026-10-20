"""Shared epoch-duration and coverage checks for sleep predictions."""

import numpy as np
import pandas as pd

EPOCH_LENGTH_ATTR = "epoch_length_s"


def validate_epoch_length(epoch_length: float) -> float:
    if (
        isinstance(epoch_length, (bool, np.bool_))
        or not isinstance(epoch_length, (int, float, np.integer, np.floating))
        or not np.isfinite(epoch_length)
        or epoch_length <= 0
        or pd.Timedelta(seconds=float(epoch_length)).value == 0
    ):
        raise ValueError("epoch_length must be finite and positive, in seconds.")
    return float(epoch_length)


def resolve_epoch_length(data: pd.DataFrame, epoch_length: float | None = None) -> float:
    """Resolve seconds from an explicit value, prediction metadata, or regular datetime spacing."""
    metadata = data.attrs.get(EPOCH_LENGTH_ATTR)
    if metadata is not None:
        metadata = validate_epoch_length(metadata)
    if epoch_length is not None:
        epoch_length = validate_epoch_length(epoch_length)
        if metadata is not None and not np.isclose(epoch_length, metadata, rtol=0, atol=1e-9):
            raise ValueError("epoch_length conflicts with prediction metadata.")
    else:
        epoch_length = metadata
    index = data.index
    if isinstance(index, pd.DatetimeIndex):
        epoch_length = _datetime_epoch_length(index, epoch_length)
    elif not pd.api.types.is_integer_dtype(index.dtype) or (len(index) > 1 and not np.all(np.diff(index) == 1)):
        raise ValueError("Use a DatetimeIndex or consecutive integer epoch indices.")
    if epoch_length is None:
        raise ValueError("Specify epoch_length in seconds for integer indices or ambiguous timestamp coverage.")
    return epoch_length


def _datetime_epoch_length(index: pd.DatetimeIndex, epoch_length: float | None) -> float | None:
    if index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Epoch timestamps must be valid, unique, and increasing.")
    deltas = np.diff(index.as_unit("ns").asi8)
    if epoch_length is None:
        if len(deltas):
            epoch_length = validate_epoch_length(deltas[0] / 1e9)
        elif index.freq is not None:
            epoch_length = validate_epoch_length(pd.Timedelta(index.freq).total_seconds())
    if epoch_length is not None and not np.allclose(deltas, epoch_length * 1e9, rtol=0, atol=1):
        raise ValueError("Epoch timestamps must be contiguous and agree with epoch_length; split recordings at gaps.")
    return epoch_length
