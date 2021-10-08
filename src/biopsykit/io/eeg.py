"""I/O functions for files related to EEG processing."""
from typing import Tuple

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_has_columns
from biopsykit.utils._types import path_t
from biopsykit.utils.time import tz, utc

__all__ = ["load_eeg_raw_muse"]

MUSE_EEG_SAMPLING_RATE = 250.0


def load_eeg_raw_muse(file_path: path_t) -> Tuple[pd.DataFrame, float]:
    """Load a csv file with raw EEG data recorded by the Muse headband.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file

    Returns
    -------
    data : :class:`~pandas.DataFrame`
        dataframe with raw EEG data
    fs : float
        sampling rate

    Raises
    ------
    ValidationError
        if file specified by ``file_path`` does not contain the required `timestamp` column as well as the
        EEG channel columns

    """
    fs = MUSE_EEG_SAMPLING_RATE
    data = pd.read_csv(file_path)

    _assert_has_columns(data, [["timestamps", "TP9", "AF7", "AF8", "TP10"]])
    # convert timestamps to datetime object, set as dataframe index
    data["timestamps"] = pd.to_datetime(data["timestamps"], unit="s")
    data = data.set_index("timestamps")
    # convert timestamp from UTC into the correct time zone
    data = data.tz_localize(utc).tz_convert(tz)
    # drop the AUX column, if present
    data = data.drop(columns="Right AUX", errors="ignore")
    return data, fs
