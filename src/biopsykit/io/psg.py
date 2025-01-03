"""Module for importing data recorded by a PSG system (expects .edf files)."""

import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

try:
    import mne
except ImportError as e:
    raise ImportError(
        "The 'mne' package is required to read edf data files. "
        "Please install it using 'pip install mne' or 'poetry add mne'."
    ) from e

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_is_dir
from biopsykit.utils._types import path_t

__all__ = ["PSGDataset"]


class PSGDataset:
    """Class for loading and processing PSG data."""

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        sampling_rate_dict: dict[str, int],
        start_time: Optional[pd.Timestamp] = None,
        tz: Optional[str] = "Europe/Berlin",
    ):

        self._data = data_dict
        for name, data in data_dict.items():
            setattr(self, name, data)
        for _, sampling_rate in sampling_rate_dict.items():
            setattr(self, "sampling_rate", sampling_rate)
        setattr(self, "channels", list(self._data.keys()))
        self._sampling_rate = sampling_rate_dict
        self._start_time_datetime = pd.Timestamp(start_time)
        self._start_time_unix = int(time.mktime(start_time.timetuple()))
        self._tz = tz

    @classmethod
    def from_edf_file(
        cls,
        path: path_t,
        datastreams: Optional[Sequence] = None,
        tz: Optional[str] = "Europe/Berlin",
    ):
        """Create a new Dataset from a valid .edf file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            Path to the .edf file
        datastreams : Optional[Sequence], optional
            List of datastreams to load, by default None. If None, all datastreams are loaded.
        tz : Optional[str], optional
            Timezone of the recording, by default "Europe/Berlin".

        Returns
        -------
        PSGDataset : New Dataset instance

        """
        if path.is_dir():
            data_dict, fs, start_time = cls.load_data_folder(path, datastreams)
        else:
            data_dict, fs, start_time = cls.load_data(path, datastreams)

        return cls(data_dict=data_dict, sampling_rate_dict={"sampling_rate": fs}, start_time=start_time, tz=tz)

    @property
    def start_time_unix(self) -> Optional[pd.Timestamp]:
        """Start time of the recording in UTC time."""
        return self._start_time_unix

    @property
    def start_time_datetime(self) -> Optional[pd.Timestamp]:
        """Start time of the recording in UTC time."""
        return self._start_time_datetime

    @property
    def timezone(self) -> str:
        """Timezone the dataset was recorded in."""
        return self._tz

    def data_as_df(self, index: Optional[str] = None) -> pd.DataFrame:
        """Return data as one combined pandas.DataFrame.

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
        pd.DataFrame: Combined data as pandas DataFrame with respective index

        """
        # get datastreams from dict
        datastreams = self._data.keys()
        data = [self._data[datastream] for datastream in datastreams]
        data = pd.concat(data, axis=1)

        data = self._add_index(data, index)

        return data

    def load_ground_truth(
        self,
        path: path_t,
    ):
        """Load ground truth data from a .xlsx file which can be exported from the Software Somnomedics.

        .. note::
            Other formats are not supported yet and raise a FileNotFoundError.

        path : :class:`pathlib.Path` or str
            path to the .xlsx file
        return: pd.DataFrame
            ground truth data as pandas DataFrame

        """
        file_path = path.parents[1].joinpath("labels/PSG_analyse.xlsx")
        try:
            sleep_phases = pd.read_excel(file_path, sheet_name="Schlafprofil", header=7, index_col=0)
        except FileNotFoundError:
            warnings.warn("No ground truth found")
            return pd.DataFrame()

        return sleep_phases
        # TODO: Read in excel or txt files with sleep labels

    @classmethod
    def load_data_folder(
        cls,
        folder_path: path_t,
        datastreams: Optional[Sequence] = None,
    ):
        """Load data from a folder containing a single .edf file.

        Parameters
        ----------
        folder_path : :class:`pathlib.Path` or str
            path to the folder containing the .edf file
        datastreams: lst, optional
            list of datastreams to load. If None, all datastreams are loaded
        timezone: str, optional
            timezone of the recording. If None, the timezone is set to UTC

        Returns
        -------
        data_dict: dict of datastreams
        fs: sampling rate
        start_time: start time of the recording

        Raises
        ------
        FileNotFoundError: if no .edf file is found in the folder
        ValueError: if more than one .edf file is found in the folder

        """
        _assert_is_dir(folder_path)

        # look for all PSG .edf files in the folder
        dataset_list = sorted(folder_path.glob("*.edf"))
        if len(dataset_list) == 0:
            raise FileNotFoundError(f"No PSG files found in folder {folder_path}!")
        if len(dataset_list) > 1:
            raise ValueError(
                f"More than one PSG files found in folder {folder_path}!"
                f"This function only supports one recording per folder!"
            )
        result_dict, fs, start_time = cls.load_data(folder_path.joinpath(dataset_list[0]), datastreams)

        return result_dict, fs, start_time

    @classmethod
    def load_data(
        cls,
        path: path_t,
        datastreams: Optional[Sequence] = None,
    ):
        """Load PSG data from a valid .edf file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            path to the .edf file
        datastreams: lst, optional
            list of datastreams to load. If None, all datastreams are loaded
        timezone: str, optional
            timezone of the recording. If None, the timezone is set to UTC

        Returns
        -------
        data_dict: dict of datastreams
        fs: sampling rate
        start_time: start time of the recording

        Raises
        ------
        Value Error: Not all datastreams are found in the .edf file

        """
        # load raw data
        data_psg, fs = cls.load_data_raw(path)

        # select datastreams to extract
        if datastreams is None:
            datastreams = data_psg.ch_names
        if isinstance(datastreams, str):
            datastreams = [datastreams]

        # save extracted datastreams in dict
        result_dict = {}
        for datastream in datastreams:
            try:
                time_idx, _, start_time = cls._create_dt_index(data_psg.info["meas_date"], times_array=data_psg.times)
                psg_datastream = data_psg.copy().pick([datastream]).get_data()[0, :]
                result_dict[datastream] = pd.DataFrame(psg_datastream, index=time_idx, columns=[datastream])
            except ValueError:
                print(
                    "Not all channels match the selected datastreams - Following Datastreams are available: "
                    + str(data_psg.ch_names)
                )

        return result_dict, fs, start_time

    @classmethod
    def load_data_raw(
        cls,
        path: path_t,
    ):
        """Load PSG data from .edf file.

        Parameters
        ----------
        path : :class:`pathlib.Path` or str
            path to the .edf file
        timezone: str, optional
            timezone of the recording. If None, the timezone is set to UTC

        Returns
        -------
        data: mne.io.Raw object
        fs: sampling rate

        """
        # ensure pathlib
        path = Path(path)
        _assert_file_extension(path, ".edf")

        # load data from edf file
        edf = mne.io.read_raw_edf(path)

        # get sampling rate
        fs = edf.info["sfreq"]

        return edf, fs

    @classmethod
    def _create_dt_index(cls, starttime, times_array):
        """Create a datetime index from the start time and the times array."""
        starttime_s = starttime.timestamp()
        # add start time to array of timestamps
        times_array = times_array + starttime_s
        # convert to datetime
        datetime_index = pd.to_datetime(times_array, unit="s")
        # generate epochs from datetime index
        epochs, start_time = cls._generate_epochs(datetime_index)
        return datetime_index, epochs, start_time

    @classmethod
    def _generate_epochs(cls, datetime_index):
        """Generate epochs from a datetime index."""
        start_time = datetime_index[0]
        # round to 30 second epochs
        epochs_30s = datetime_index.round("30s")

        epochs_clear = (epochs_30s - start_time).total_seconds()

        epochs_clear = epochs_clear / 30
        epochs = epochs_clear.astype(int)
        return epochs, start_time

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
        index_name = index_names[index]
        data.index.name = index_name
        if index is None:
            data = data.reset_index(drop=True)
            data.index.name = index_name
            return data
        if index == "utc_datetime":
            data.index = data.index.tz_localize(self.timezone).tz_convert("UTC")
            return data
        if index == "time":
            data.index = data.index - self.start_time_datetime
            data.index = data.index.total_seconds()
            return data
        if index == "utc":
            # convert counter to utc timestamps i seconds
            data = data.reset_index(drop=True)
            data.index.astype("int64")
            data.index += self.start_time_unix
            return data
        if index == "local_datetime":
            data.index = data.index.tz_localize(self.timezone)
            return data
        return data
