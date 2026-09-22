"""Module for importing data recorded by the Fibion sensor system."""

import datetime
import re
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
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
        "Sample": "ecg",
        "AccX": "acc_x",
        "AccY": "acc_y",
        "AccZ": "acc_z",
    }

    _start_time_unix: pd.Timestamp | None
    _tz: str | None
    _data: dict[str, pd.DataFrame]
    _sampling_rate: dict[str, float]

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

    @classmethod
    def from_csv_file(cls, path: path_t, tz: str | None = "Europe/Berlin"):
        """Create a new Dataset from a valid .csv file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the file
        tz : str, optional
            Timezone str of the recording. This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        """
        _assert_file_extension(path, ".csv")
        path = Path(path)

        channel_type = path.stem.split("-")[0].lower()
        data, start_time, metadata_lines = cls._load_csv_data(path)
        data, start_time = cls._add_time_index(data, start_time)

        data = data.set_index("time")
        data = data.rename(columns=cls._CHANNEL_NAME_MAPPING)
        sampling_rate_hz = cls._infer_sampling_rate(data, metadata_lines)

        return cls(
            data_dict={channel_type: data},
            sampling_rate_dict={channel_type: float(sampling_rate_hz)},
            start_time=start_time,
            tz=tz,
        )

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
        csv_files = sorted(path.glob("*.csv"))
        edf_files = sorted(path.glob("*.edf"))

        if edf_files:
            files = edf_files
            load_fn = cls.from_edf_file
        elif csv_files:
            files = csv_files
            load_fn = cls.from_csv_file
        else:
            raise ValueError(f"No .csv or .edf files found in folder {path}!")

        data_dict = {}
        sampling_rate_dict = {}
        start_time = None
        for file in files:
            dataset = load_fn(file, tz=tz)
            data_dict.update(dataset._data)
            sampling_rate_dict.update(dataset._sampling_rate)
            if start_time is None:
                start_time = dataset.start_time_unix

        if edf_files and csv_files:
            csv_start_time = cls._get_start_time_from_csv_footer(csv_files[0])
            if csv_start_time is not None:
                start_time = csv_start_time

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

    @staticmethod
    def _read_csv_footer(path: path_t, n_lines: int = 4, chunk_size: int = 1024) -> list[str]:
        path = Path(path)
        with path.open("rb") as file:
            file.seek(0, 2)
            file_size = file.tell()
            buffer = b""
            position = file_size

            while position > 0 and buffer.count(b"\n") < n_lines + 1:
                read_size = min(chunk_size, position)
                position -= read_size
                file.seek(position)
                buffer = file.read(read_size) + buffer

        return [line.strip() for line in buffer.decode("utf-8", errors="replace").splitlines() if line.strip()][
            -n_lines:
        ]

    @classmethod
    def _get_start_time_from_csv_footer(cls, path: path_t) -> pd.Timestamp | None:
        for line in cls._read_csv_footer(path):
            match = re.search(r"UTC Timestamp at start\s*:\s*(\d+)\s*ms", line)
            if match:
                return pd.to_datetime(int(match.group(1)), unit="ms", utc=True)
        return None

    @staticmethod
    def _read_csv_lines(path: Path) -> list[str]:
        with path.open(encoding="utf-8", errors="replace") as file:
            return [line.strip() for line in file if line.strip()]

    @staticmethod
    def _split_csv_sections(lines: list[str], path: Path) -> tuple[str, list[str], list[str]]:
        body_lines = lines[:-4]
        header_idx = next((i for i, line in enumerate(body_lines) if "," in line), None)
        if header_idx is None:
            raise ValueError(f"Could not find CSV header in file {path}.")

        data_lines = body_lines[header_idx + 1 :]
        if not data_lines:
            raise ValueError(f"No data rows found in CSV file {path}.")
        return body_lines[header_idx], data_lines, body_lines[:header_idx]

    @classmethod
    def _load_csv_data(cls, path: Path) -> tuple[pd.DataFrame, pd.Timestamp | None, list[str]]:
        lines = cls._read_csv_lines(path)
        header, data_lines, metadata_lines = cls._split_csv_sections(lines, path)
        csv_body = "\n".join([header, *data_lines])
        data = pd.read_csv(StringIO(csv_body))
        start_time = cls._get_start_time_from_csv_footer(path)
        return data, start_time, metadata_lines

    @staticmethod
    def _add_time_index(
        data: pd.DataFrame, start_time: pd.Timestamp | None
    ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        if start_time is None and "Timestamp" in data.columns:
            start_time = pd.to_datetime(data["Timestamp"].iloc[0], unit="s", utc=True)

        if "Timestamp" in data.columns and start_time is not None:
            data["time"] = data["Timestamp"].astype(float) - start_time.timestamp()
            data = data.drop(columns=["Timestamp"])
        elif "Timestamp" in data.columns:
            first_timestamp = float(data["Timestamp"].iloc[0])
            data["time"] = data["Timestamp"].astype(float) - first_timestamp
            data = data.drop(columns=["Timestamp"])
        else:
            # Fallback if there is no timestamp column (e.g., HR exports).
            data["time"] = data.index.astype(float)
        return data, start_time

    @staticmethod
    def _infer_sampling_rate(data: pd.DataFrame, metadata_lines: list[str]) -> float:
        for line in reversed(metadata_lines):
            match = re.search(r"\d{2}\.\d{2}\.\d{2}\d{2}\.\d{2}\.\d{2}(\d+)\s*$", line)
            if match:
                return float(match.group(1))

            match = re.search(r"(\d+)\s*$", line)
            if match:
                candidate = float(match.group(1))
                if 0 < candidate < 5000:
                    return candidate

        if len(data) > 1:
            diffs = pd.Series(data.index).diff()
            positive_diffs = diffs[diffs > 0]
            if not positive_diffs.empty:
                sample_interval = positive_diffs.median()
                if sample_interval > 0:
                    return 1 / sample_interval
        return float("nan")
