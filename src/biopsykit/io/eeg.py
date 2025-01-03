"""I/O functions for files related to EEG processing."""
import warnings
from typing import Optional

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_has_columns
from biopsykit.utils._types import path_t

__all__ = ["MuseDataset", "load_eeg_raw_muse"]


class MuseDataset:
    """Class for loading and processing EEG data from the Muse EEG wearable headband."""

    SAMPLING_RATE_HZ = 250.0
    _start_time_unix: pd.Timestamp
    _tz: str
    _data: pd.DataFrame
    _sampling_rate: int

    def __init__(self, data: pd.DataFrame, tz: Optional[str] = None):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_csv_file` constructor to handle loading recorded Muse data from a CSV file.

        Parameters
        ----------
        data : :class:`pandas.DataFrame`
            DataFrame containing the raw EEG data.
        tz : str, optional
            Timezone of the recording, if present or ``None`` if no timezone is available.

        """
        self._data = data
        self._tz = tz

    @classmethod
    def from_csv_file(cls, file_path: path_t, tz: Optional[str] = None):
        """Load Muse data from a CSV file.

        Parameters
        ----------
        file_path : str or :class:`pathlib.Path`
            Path to the CSV file.
        tz : str, optional
            Timezone of the recording, if present or ``None`` if no timezone is available.

        Returns
        -------
        :class:`~biopsykit.io.eeg.MuseDataset`
            Dataset instance containing the loaded data.

        """
        data = pd.read_csv(file_path)
        _assert_has_columns(data, [["timestamps", "TP9", "AF7", "AF8", "TP10"]])
        # drop the AUX column, if present
        data = data.drop(columns="Right AUX", errors="ignore")
        return cls(data, tz=tz)

    @property
    def sampling_rate_hz(self):
        """Return the sampling rate of the EEG data in Hz."""
        return self.SAMPLING_RATE_HZ

    @property
    def timezone(self) -> str:
        """Return the timezone of the recording."""
        return self._tz

    def data_as_df(self, index: Optional[str] = None) -> pd.DataFrame:
        """Return all data as one combined :class:`pandas.DataFrame`.

        Parameters
        ----------
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame containing the raw EEG data.

        """
        data = self._add_index(self._data, index)
        return data

    def _add_index(self, data: pd.DataFrame, index: str) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")
        data = data.set_index("timestamps")

        index_name = index_names[index]
        data.index.name = index_name

        if index == "utc":
            return data
        if index == "time":
            data.index -= data.index[0]
            return data
        if index is None:
            data = data.reset_index(drop=True)
            data.index.name = index_name
            return data

        # convert counter to pandas datetime index
        data.index = pd.to_datetime(data.index, unit="s").tz_localize("UTC")

        if index == "local_datetime":
            data.index = data.index.tz_convert(self.timezone)

        return data


def load_eeg_raw_muse(file_path: path_t, tz: Optional[str] = "Europe/Berlin") -> tuple[pd.DataFrame, float]:
    """Load a csv file with raw EEG data recorded by the Muse headband.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    tz : str, optional
        timezone of the recording, if present or ``None`` if no timezone is available. Default: "Europe/Berlin"

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
    # raise deprecation warning
    warnings.warn(
        "The function `load_eeg_raw_muse` is deprecated and will be removed in a future version. "
        "Use the `MuseDataset` class instead.",
        category=DeprecationWarning,
    )
    fs = 250.0
    data = pd.read_csv(file_path)

    _assert_has_columns(data, [["timestamps", "TP9", "AF7", "AF8", "TP10"]])
    # convert timestamps to datetime object, set as dataframe index
    data["timestamps"] = pd.to_datetime(data["timestamps"], unit="s")
    data = data.set_index("timestamps")
    # convert timestamp from UTC into the correct time zone
    data = data.tz_localize("UTC").tz_convert(tz)
    # drop the AUX column, if present
    data = data.drop(columns="Right AUX", errors="ignore")
    return data, fs
