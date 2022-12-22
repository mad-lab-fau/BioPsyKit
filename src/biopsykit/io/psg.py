"""Module for importing data recorded by a PSG system (expects .edf files)."""

from biopsykit.utils._types import path_t
from typing import Dict, Optional, Sequence, Union

import pandas as pd

import datetime
import warnings
from pathlib import Path
from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_is_dir
import mne

__all__ = ["PSGDataset"]


class PSGDataset:
    """Class for loading and processing PSG data."""

    def __init__(
            self, data_dict: Dict[str, pd.DataFrame], sampling_rate_dict: Dict[str, int], start_time: Optional[pd.Timestamp] = None,
            tz: Optional[str] = None,
    ):

        self._data = data_dict
        for name, data in data_dict.items():
            setattr(self, name, data)
        for name, sampling_rate in sampling_rate_dict.items():
            setattr(self, "sampling_rate", sampling_rate)
        setattr(self, "channels", list(self._data.keys()))
        self._sampling_rate = sampling_rate_dict
        self._start_time_unix = start_time
        self._tz = tz

    @classmethod
    def from_edf_file(
            cls,
            path: path_t,
            datastreams: Optional[Sequence] = None,
            tz: Optional[str] = "Europe/Berlin",
    ):
        """Create a new Dataset from a valid .edf file.
        ----------
        Parameters
        ----------
        path : path_t: Path to the .edf file
        datastreams : Optional[Sequence], optional : List of datastreams to load, by default None. If None, all datastreams are loaded.
        tz : Optional[str], optional : Timezone of the recording, by default "Europe/Berlin".
        ----------
        Returns
        ----------
        PSGDataset : New Dataset instance
        """

        if path.is_dir():
            data_dict, fs, start_time = cls.load_data_folder(path, datastreams, tz)
        else:
            data_dict, fs, start_time = cls.load_data(path, datastreams, tz)

        return cls(
            data_dict=data_dict,
            sampling_rate_dict={"sampling_rate": fs},
            start_time=start_time,
            tz=tz
        )

    @property
    def start_time_unix(self) -> Optional[pd.Timestamp]:
        """Start time of the recording in UTC time."""
        return self._start_time_unix

    @property
    def timezone(self) -> str:
        """Timezone the dataset was recorded in."""
        return self._tz

    def data_as_df(self):
        """Return data as pandas DataFrame."""
        # get datastreams from dict
        datastreams = self._data.keys()
        data = [self._data[datastream] for datastream in datastreams]
        data = pd.concat(data, axis=1)

        return data

    def load_ground_truth(self, path: path_t, ):
        """Load ground truth data from a .xlsx file which can be exported from the Analyse Sofeware Somnomedics.
        Other formats are not supported yet and raise a FileNotFoundError.
        path: path to the .xlsx file
        return: ground truth data as pandas DataFrame
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
    def load_data_folder(cls,
                         folder_path: path_t,
                         datastreams: Optional[Sequence] = None,
                         timezone: Optional[Union[datetime.tzinfo, str]] = "Europe/Berlin",
                         ):
        """Load data from a folder containing a single .edf file.
        ----------
        Parameters
        ----------
        folder_path: path to the folder containing the .edf file
        datastreams: list of datastreams to load. If None, all datastreams are loaded
        timezone: timezone of the recording. If None, the timezone is set to UTC
        -------
        Returns
        -------
        data_dict: dict of datastreams
        fs: sampling rate
        start_time: start time of the recording
        -------
        Raises
        -------
        FileNotFoundError: if no .edf file is found in the folder
        ValueError: if more than one .edf file is found in the folder
        """
        _assert_is_dir(folder_path)

        # look for all PSG .edf files in the folder
        dataset_list = list(sorted(folder_path.glob("*.edf")))
        if len(dataset_list) == 0:
            raise FileNotFoundError(f"No PSG files found in folder {folder_path}!")
        if len(dataset_list) > 1:
            raise ValueError(
                f"More than one PSG files found in folder {folder_path}! This function only supports one recording per folder!"
            )

        result_dict, fs, start_time = cls.load_data(path_t.joinpath(dataset_list[0]), datastreams, timezone)

        return result_dict, fs, start_time

    @classmethod
    def load_data(cls,
                  path: path_t,
                  datastreams: Optional[Sequence] = None,
                  timezone: Optional[Union[datetime.tzinfo, str]] = "Europe/Berlin",
                  ):
        """Load PSG data from a valid .edf file.
        ----------
        Parameters
        ----------
        path: path to the .edf file
        datastreams: list of datastreams to load. If None, all datastreams are loaded
        timezone: timezone of the recording. If None, the timezone is set to UTC
        -------
        Returns
        -------
        data_dict: dict of datastreams
        fs: sampling rate
        start_time: start time of the recording
        -------
        Raises
        -------
        Value Error: Not all datastreams are found in the .edf file
        """

        # load raw data
        data_psg, fs = cls.load_data_raw(path, timezone)

        # select datastreams to extract
        if datastreams is None:
            datastreams = data_psg.ch_names
        if isinstance(datastreams, str):
            datastreams = [datastreams]

        # save extracted datastreams in dict
        result_dict = {}
        for datastream in datastreams:
            try:
                time, epochs, start_time = cls._create_datetime_index(data_psg.info["meas_date"], times_array=data_psg.times)
                psg_datastream = data_psg.copy().pick_channels([datastream]).get_data()[0, :]
                result_dict[datastream] = pd.DataFrame(psg_datastream, index=time, columns=[datastream])
            except ValueError:
                raise ValueError(
                    "Not all channels match the selected datastreams - Following Datastreams are available: "
                    + str(data_psg.ch_names)
                )

        return result_dict, fs, start_time

    @classmethod
    def load_data_raw(
            cls, path: path_t, timezone: Optional[Union[datetime.tzinfo, str]] = "Europe/Berlin",
    ):
        """load PSG data from .edf file.
        ----------
        Parameters
        ----------
        path: path to the .edf file
        timezone: timezone of the recording. If None, the timezone is set to UTC
        -------
        Returns
        -------
        data: mne.io.Raw object
        fs: sampling rate
        """

        # ensure pathlib
        path = Path(path)
        _assert_file_extension(path, ".edf")

        if timezone is None:
            timezone = "Europe/Berlin"

        # load data from edf file
        edf = mne.io.read_raw_edf(path)

        # get sampling rate
        fs = edf.info["sfreq"]

        return edf, fs

    @classmethod
    def _create_datetime_index(cls, starttime, times_array):
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
