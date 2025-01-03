"""Module for importing data recorded by the Biopac system."""

import datetime
from collections.abc import Sequence
from typing import ClassVar, Optional, Union

try:
    import bioread
except ImportError as e:
    raise ImportError(
        "The 'bioread' package is required to read Biopac data files. "
        "Please install it using 'pip install bioread' or 'poetry add bioread'."
    ) from e

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t, str_t

__all__ = ["BiopacDataset"]


class BiopacDataset:
    """Class for loading and processing Biopac data."""

    _CHANNEL_NAME_MAPPING: ClassVar[dict[str, str]] = {
        "ECG": "ecg",
        "RSP": "rsp",
        "EDA": "eda",
        "EMG": "emg",
        "ICG - Magnitude": "icg_mag",
        "ICG - Derivative": "icg_der",
        "SYNC": "sync",
    }

    _start_time_unix: pd.Timestamp
    _tz: str
    _event_markers: Optional[Sequence[bioread.reader.EventMarker]] = None
    _data: ClassVar[dict[str, pd.DataFrame]] = {}
    _sampling_rate: ClassVar[dict[str, int]] = {}

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        sampling_rate_dict: dict[str, int],
        start_time: Optional[pd.Timestamp] = None,
        event_markers: Optional[Sequence[bioread.reader.EventMarker]] = None,
        tz: Optional[str] = None,
    ):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_acq_file` constructor to handle loading recorded Biopac Sessions.

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
        event_markers : list of :class:`bioread.reader.EventMarker`, optional
            List of event markers set during the recording if present or ``None`` if no event markers are available.
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
        self._event_markers = event_markers
        self._tz = tz

    @classmethod
    def from_acq_file(
        cls, path: path_t, channel_mapping: Optional[dict[str, str]] = None, tz: Optional[str] = "Europe/Berlin"
    ):
        """Create a new Dataset from a valid .acq file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file
        channel_mapping : dict, optional
            Dictionary containing the mapping of the channel names in the .acq to the channel names used in the Dataset.
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        """
        # assert that file is an acq file
        _assert_file_extension(path, ".acq")
        biopac_data: bioread.reader.Datafile = bioread.read(str(path))

        start_time = None
        # if no event markers are available we can't compute a start time of the recording
        if biopac_data.event_markers is not None and len(biopac_data.event_markers) > 0:
            marker_time = pd.Timestamp(biopac_data.event_markers[0].date_created_utc)
            marker_sample_idx = biopac_data.event_markers[0].sample_index
            # start time is the marker time minus the time at the position of the marker sample
            start_time = marker_time - pd.Timedelta(seconds=biopac_data.time_index[marker_sample_idx])

        if channel_mapping is None:
            channel_mapping = cls._CHANNEL_NAME_MAPPING

        dict_channel_data, dict_sampling_rate = cls._extract_channel_information(biopac_data, channel_mapping)

        return cls(
            data_dict=dict_channel_data,
            sampling_rate_dict=dict_sampling_rate,
            start_time=start_time,
            event_markers=biopac_data.event_markers,
            tz=tz,
        )

    @property
    def start_time_unix(self) -> Optional[pd.Timestamp]:
        """Start time of the recording in UTC time."""
        return self._start_time_unix

    @property
    def timezone(self) -> str:
        """Timezone the dataset was recorded in."""
        return self._tz

    @property
    def event_markers(self):
        """Event markers set in the AcqKnowledge software during the recording."""
        return self._event_markers

    def data_as_df(
        self,
        datastreams: Optional[str_t] = None,
        index: Optional[str] = None,
        start_time: Optional[Union[str, datetime.datetime, pd.Timestamp]] = None,
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

    @classmethod
    def _extract_channel_information(
        cls, biopac_data: bioread.reader.Datafile, channel_mapping: dict[str, str]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
        # TODO raise warning or error when there are more channels than were extracted
        #  (might be an indication that mapping does not contain all channels)
        dict_channel_data = {}
        dict_sampling_rate = {}
        for channel in biopac_data.channels:
            # check if channel name is in mapping
            for key, value in channel_mapping.items():
                if channel.name.startswith(key):
                    ch_name = value
                    channel_df = pd.DataFrame(
                        channel.data, index=pd.Index(biopac_data.time_index, name="t"), columns=[ch_name]
                    )
                    if ch_name in dict_channel_data:
                        if dict_sampling_rate[ch_name] != channel.samples_per_second:
                            raise ValueError(f"Sampling rates for '{ch_name}' must be the same for all channels!")
                        dict_channel_data[ch_name] = pd.concat([dict_channel_data[ch_name], channel_df], axis=1)
                    else:
                        dict_channel_data[ch_name] = channel_df

                    dict_sampling_rate[ch_name] = channel.samples_per_second
                    break

        return dict_channel_data, dict_sampling_rate

    def _add_index(self, data: pd.DataFrame, index: str, start_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
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
        if index == "utc":
            # convert counter to utc timestamps
            data.index += self.start_time_unix.timestamp()
            return data

        if start_time is None:
            start_time = self.start_time_unix

        if start_time is None:
            raise ValueError(
                "No start time available - can't convert to datetime index! "
                "Use a different index representation or provide a custom start time using the 'start_time' parameter."
            )

        # convert counter to pandas datetime index
        data.index = pd.to_timedelta(data.index, unit="s")
        data.index += self.start_time_unix

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
