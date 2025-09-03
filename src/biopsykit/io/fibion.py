"""Module for importing data recorded by the Fibion sensor system."""

import datetime
from collections.abc import Sequence
from typing import ClassVar

try:
    from mne.io import read_raw_edf
except ImportError as e:
    raise ImportError(
        "The 'mne' package is required to read Fibion EDF data files. "
        "Please install it using 'pip install bioread' or 'uv add bioread' or by installing biopsykit "
        "with the mne extra using 'pip install biopsykit -E mne'."
    ) from e

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types_internal import path_t, str_t

__all__ = ["FibionDataset"]


class FibionDataset:
    """Class for loading and processing Fibion data."""

    _CHANNEL_NAME_MAPPING: ClassVar[dict[str, str]] = {
        "ECG": "ecg",
        "Accelerometer_X": "acc_x",
        "Accelerometer_Y": "acc_y",
        "Accelerometer_Z": "acc_z",
    }

    _start_time_unix: pd.Timestamp
    _tz: str
    _data: ClassVar[dict[str, pd.DataFrame]] = {}
    _sampling_rate: ClassVar[dict[str, float]] = {}

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        sampling_rate_dict: dict[str, float],
        start_time: pd.Timestamp | None = None,
        tz: str | None = None,
    ):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_edf_file`, `from_csv_file`, or `from_folder` constructors to handle loading
            recorded Fibion Sessions.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing data of the channels as :class:`pandas.DataFrame`.
            The keys of the dictionary are the channel names.
        sampling_rate_dict : dict
            Dictionary containing the sampling rate of the channels.
            The keys of the dictionary are the channel names.
        start_time : :class:`pandas.Timestamp`, optional
            Start time of the recording, if present, or ``None`` if no start time is available.
        tz : str, optional
            Timezone of the recording, if present or ``None`` if no timezone is available.

        """
        self._data = data_dict
        for name, data in data_dict.items():
            setattr(self, name, data)
        for name, sampling_rate in sampling_rate_dict.items():
            setattr(self, f"sampling_rate_hz_{name}", sampling_rate)
        setattr(self, "channels", list(self._data.keys()))
        self._sampling_rate = sampling_rate_dict
        self._start_time_unix = start_time
        self._tz = tz

    @classmethod
    def from_edf_file(cls, path: path_t, tz: str | None = "Europe/Berlin"):
        """Create a new Dataset from a valid .edf file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        """
        # assert that file is an edf file
        _assert_file_extension(path, ".edf")
        fibion_data = read_raw_edf(path, verbose=False)
        channel_type = path.stem.split("-")[0].lower()

        start_time = fibion_data.info["meas_date"]

        data = fibion_data.to_data_frame()
        data = data.set_index("time")
        data = data.rename(columns=cls._CHANNEL_NAME_MAPPING)

        return cls(
            data_dict={channel_type: data},
            sampling_rate_dict={channel_type: fibion_data.info["sfreq"]},
            start_time=start_time,
            tz=tz,
        )

    # @classmethod
    # def from_csv_file(cls, path: path_t, tz: str | None = "Europe/Berlin"):
    #     """Create a new Dataset from a valid .csv file.
    #
    #     Parameters
    #     ----------
    #     path : :class:`pathlib.Path` or str
    #         Path to the file
    #     tz : str, optional
    #         Timezone str of the recording. This can be used to localize the start and end time.
    #         Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
    #         recording.
    #
    #     """
    #     # assert that file is an edf file
    #     _assert_file_extension(path, ".csv")
    #     channel_type = path.stem.split("-")[0].lower()
    #     data = pd.read_csv(path)
    #
    #     print(data)
    #
    #     data = data.set_index("time")
    #     data = data.rename(columns=cls._CHANNEL_NAME_MAPPING)
    #
    #     return cls(
    #         data_dict={channel_type: data},
    #         sampling_rate_dict={channel_type: fibion_data.info["sfreq"]},
    #         start_time=start_time,
    #         tz=tz,
    #     )

    @classmethod
    def from_folder(cls, path: path_t, tz: str | None = "Europe/Berlin"):
        """Create a new Dataset from a valid .edf file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        """
        files = sorted(path.glob("*.edf"))
        if len(files) == 0:
            raise ValueError(f"No .edf files found in folder {path}!")

        data_dict = {}
        sampling_rate_dict = {}
        start_time = None
        for file in files:
            dataset = cls.from_edf_file(file, tz=tz)
            data_dict.update(dataset._data)
            sampling_rate_dict.update(dataset._sampling_rate)
            start_time = dataset.start_time_unix

        return cls(data_dict=data_dict, sampling_rate_dict=sampling_rate_dict, start_time=start_time, tz=tz)

    @property
    def start_time_unix(self) -> pd.Timestamp | None:
        """Start time of the recording in UTC time."""
        return self._start_time_unix

    @property
    def timezone(self) -> str:
        """Timezone the dataset was recorded in."""
        return self._tz

    def data_as_df(
        self,
        datastreams: str_t | None = None,
        index: str | None = None,
        start_time: str | datetime.datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return all data as one combined :class:`pandas.DataFrame`.

        Parameters
        ----------
        datastreams : str, optional
            name(s) of datastream to return in dataframe. If ``None``, all datastreams are returned.
        index : str, optional
            Specify which index should be used for the dataset. The options are:
            * "time": For the time in seconds since the first sample
            * "utc": For the utc time stamp of each sample
            * "utc_datetime": for a pandas DateTime index in UTC time
            * "local_datetime": for a pandas DateTime index in the timezone set for the session
            * None: For a simple index (0...N)
        start_time : str, :class:`datetime.datetime`, :class:`pandas.Timestamp`, optional
            Start time of the recording. Can be used to provide a custom start time if no start time can be inferred
            from the recording or to overwrite the start time extracted from the recording.

        """
        # sanitize datastreams input
        datastreams = self._sanitize_datastreams_input(datastreams)

        # assert that all datastreams have the same sampling rate
        sampling_rates = {self._sampling_rate[datastream] for datastream in datastreams}
        if len(sampling_rates) > 1:
            raise ValueError("All datastreams must have the same sampling rate for combining it into one DataFrame!")

        # get datastreams from dict
        data = [self._data[datastream] for datastream in datastreams]
        data = pd.concat(data, axis=1)

        data = self._add_index(data, index, start_time=start_time)
        return data

    def _add_index(self, data: pd.DataFrame, index: str, start_time: pd.Timestamp | None = None) -> pd.DataFrame:
        index_names = {
            None: "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": f"date ({self.timezone})",
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")
        index_name = index_names[index]
        data.index.name = index_name

        if index == "time":
            return data
        if index is None:
            data = data.reset_index(drop=True)
            data.index.name = index_name
            return data

        if start_time is None:
            start_time = self.start_time_unix

        if start_time is None:
            raise ValueError(
                "No start time available - can't convert to datetime index! "
                "Use a different index representation or provide a custom start time using the 'start_time' parameter."
            )

        if index == "utc":
            print(start_time.timestamp())
            # convert counter to utc timestamps
            data.index += start_time.timestamp()
            return data

        # convert counter to pandas datetime index
        data.index = pd.to_timedelta(data.index, unit="s")
        data.index += start_time

        if index == "local_datetime":
            data.index = data.index.tz_convert(self.timezone)

        return data

    def _sanitize_datastreams_input(self, datastreams) -> Sequence[str]:
        if datastreams is None:
            datastreams = list(self._data.keys())
        if isinstance(datastreams, str):
            # ensure list
            datastreams = [datastreams]
        # assert that all datastreams are available
        for datastream in datastreams:
            if datastream not in self._data:
                raise ValueError(f"Datastream '{datastream}' is not available in Dataset!")

        return datastreams
