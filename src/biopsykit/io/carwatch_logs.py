"""Module providing functions to load and save logs from the *CARWatch* app."""
import json
import re
import warnings
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import pandas as pd
from tqdm.auto import tqdm

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t
from biopsykit.utils.time import tz, utc

if TYPE_CHECKING:
    from biopsykit.carwatch_logs import LogData

LOG_FILENAME_PATTERN = "logs_(.*?)"


def load_logs_all_subjects(
    base_folder: path_t,
    has_subject_folders: Optional[bool] = True,
    log_filename_pattern: Optional[str] = None,
    return_df: Optional[bool] = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load log files from all subjects in a folder.

    This function iterates through the base folder and looks for subfolders
    (if ``has_subject_folders`` is ``True``), or for .csv files or .zip files matching the log file name pattern.

    Files from all subjects are then loaded and returned as one :class:`~pandas.DataFrame`
    (if ``return_df`` is ``True``) or a dictionary (if ``return_df`` is ``False``).


    Parameters
    ----------
    base_folder : str or :class:`~pathlib.Path`
        path to base folder containing log files
    has_subject_folders : boolean, optional
        ``True`` if log files are stored in subfolders per subject, ``False`` if they are all stored in one
        top-level folder
    log_filename_pattern : str, optional
        file name pattern of log files as regex string or ``None`` if files have default filename
        pattern: "logs_(.*?)". A custom filename pattern needs to contain a capture group to extract the subject ID
    return_df : bool, optional
        ``True`` to return data from all subjects combined as one dataframe, ``False`` to return a dictionary with
        data per subject. Default: ``True``


    Returns
    -------
    :class:`~pandas.DataFrame` or dict
        dataframe with log data for all subjects (if ``return_df`` is ``True``).
        or dictionary with log data per subject

    """
    # ensure pathlib
    base_folder = Path(base_folder)

    if has_subject_folders:
        folder_list = [p for p in sorted(base_folder.glob("*")) if p.is_dir() and not p.name.startswith(".")]
        dict_log_files = _load_log_file_folder(folder_list)
    else:
        # first, look for available csv files
        file_list = list(sorted(base_folder.glob("*.csv")))
        if len(file_list) > 0:
            dict_log_files = _load_log_file_list_csv(file_list, log_filename_pattern)
        else:
            # fallback: look for zip files
            file_list = list(sorted(base_folder.glob("*.zip")))
            dict_log_files = _load_log_file_zip(file_list, log_filename_pattern)

    if return_df:
        return pd.concat(dict_log_files, names=["subject_id"])
    return dict_log_files


def _load_log_file_folder(folder_list: Sequence[Path]):
    dict_log_files = {}
    for folder in tqdm(folder_list):
        subject_id = folder.name
        dict_log_files[subject_id] = load_log_one_subject(folder)

    return dict_log_files


def _load_log_file_list_csv(file_list: Sequence[Path], log_filename_pattern: str):
    dict_log_files = {}
    if log_filename_pattern is None:
        log_filename_pattern = LOG_FILENAME_PATTERN + ".csv"
    for file in tqdm(file_list):
        subject_id = re.search(log_filename_pattern, file.name).group(1)
        df = pd.read_csv(file, sep=";")
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        dict_log_files[subject_id] = df
    return dict_log_files


def _load_log_file_zip(file_list: Sequence[Path], log_filename_pattern: str) -> Dict[str, pd.DataFrame]:
    dict_log_files = {}
    if log_filename_pattern is None:
        log_filename_pattern = LOG_FILENAME_PATTERN + ".zip"
    for file in tqdm(file_list):
        subject_id = re.search(log_filename_pattern, file.name).group(1)
        dict_log_files[subject_id] = load_log_one_subject(file)

    return dict_log_files


def load_log_one_subject(
    path: path_t,
    log_filename_pattern: Optional[str] = None,
    overwrite_unzipped_logs: Optional[bool] = False,
) -> pd.DataFrame:
    """Load log files from one subject.

    Parameters
    ----------
    path : :class:`~pathlib.Path` or str
        path to folder containing log files from subject or path to log file from subject
    log_filename_pattern : str, optional
        file name pattern of log files as regex string or ``None`` if file has default filename
        pattern: "logs_(.*?)". A custom filename pattern needs to contain a capture group to extract the subject ID
    overwrite_unzipped_logs : bool, optional
        ``True`` to overwrite already unzipped log files, ``False`` to not overwrite.
        Only relevant if log files are provided as zip files. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with log data for one subject

    """
    if log_filename_pattern is None:
        log_filename_pattern = LOG_FILENAME_PATTERN + ".zip"

    path = Path(path)

    if path.is_dir():
        # TODO add error messages if no log files are found (e.g. wrong folder path etc.)
        return log_folder_to_dataframe(path)

    if path.is_file():
        _assert_file_extension(path, [".zip", ".csv"])
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zip_ref:
            export_folder = path.parent.joinpath(re.search(log_filename_pattern, path.name).group(1))
            export_folder.mkdir(exist_ok=True)
            if overwrite_unzipped_logs or len(list(export_folder.glob("*"))) == 0:
                zip_ref.extractall(export_folder)
                # call recursively with newly exported folder
                return log_folder_to_dataframe(export_folder)
            # folder not empty => inform user and load folder
            warnings.warn(
                f"Folder {export_folder.name} already contains log files which will be loaded. "
                f"Set `overwrite_logs_unzip = True` to overwrite log files."
            )
            return log_folder_to_dataframe(export_folder)
    return _load_log_file_csv(path)


def log_folder_to_dataframe(folder_path: path_t) -> pd.DataFrame:
    """Load log data from folder of one subject and return it as dataframe.

    Parameters
    ----------
    folder_path : :class:`~pathlib.Path` or str
        path to folder containing log files from subject


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with log data for one subject

    """
    file_list = list(sorted(folder_path.glob("*.csv")))
    return pd.concat([_load_log_file_csv(file) for file in file_list])


def _load_log_file_csv(file_path: path_t) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=";", header=None, names=["time", "action", "extras"])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    df.index = df.index.tz_localize(utc).tz_convert(tz)
    df = df.sort_index()
    df = df.apply(_parse_date, axis=1)
    return df


def save_log_data(
    log_data: Union[pd.DataFrame, "LogData"],
    path: path_t,
    subject_id: Optional[str] = None,
    overwrite: Optional[bool] = False,
    show_skipped: Optional[bool] = False,
):
    """Save log data for a single subject or for all subjects at once.

    The behavior of this function depends on the input passed to ``log_data``:

    * If ``log_data`` is a :class:`~pandas.DataFrame` with a :class:`~pandas.MultiIndex` it is assumed that
      the dataframe contains data from multiple subjects and will be exported accordingly as one combined csv file.
    * If ``log_data`` is a :class:`~pandas.DataFrame` without :class:`~pandas.MultiIndex` it is assumed that the
      dataframe only contains data from one single subject and will be exported accordingly as csv file.

    Parameters
    ----------
    log_data : :class:`~pandas.DataFrame` or :class:`~biopsykit.carwatch_logs.log_data.LogData`
        log data to save
    path : :class:`~pathlib.Path` or str
        path for export. The expected format of ``path`` depends on ``log_data``:

        * If ``log_data`` is log data from a single subject ``path`` needs to specify a **folder**.
          The log data will then be exported to "path/logs_<subject_id>.csv".
        * If ``log_data`` is log data from multiple subjects ``path`` needs to specify a **file**.
          The combined log data of all subjects will then be exported to "path".

    subject_id : str, optional
        subject ID or ``None`` to get subject ID from ``LogData`` object
    overwrite : bool, optional
        ``True`` to overwrite file if it already exists, ``False`` otherwise. Default: ``False``
    show_skipped : bool, optional
        ``True`` to print message if log data was already exported and will be skipped, ``False`` otherwise.
        Default: ``False``

    """
    from biopsykit.carwatch_logs import LogData  # pylint: disable=import-outside-toplevel

    if isinstance(log_data, pd.DataFrame):
        if isinstance(log_data.index, pd.MultiIndex):
            # dataframe has a multiindex => it's a combined dataframe for all subjects
            log_data.to_csv(path, sep=";")
            return
        log_data = LogData(log_data)

    if subject_id is None:
        subject_id = log_data.subject_id

    export_path = path.joinpath("logs_{}.csv".format(subject_id))
    if not export_path.exists() or overwrite:
        log_data.data.to_csv(export_path, sep=";")
    else:
        if show_skipped:
            print("Skipping subject {}. Already exported.".format(subject_id))


def _parse_date(row: pd.Series) -> pd.Series:
    """Parse date in "timestamp" and "timestamp_hidden" columns of a pandas series.

    Parameters
    ----------
    row : :class:`~pandas.Series`
        one row of a dataframe

    Returns
    -------
    :class:`~pandas.Series`
        series with parsed date

    """
    json_extra = json.loads(row["extras"])
    row_cpy = row.copy()
    keys = ["timestamp", "timestamp_hidden"]
    for key in keys:
        if key in json_extra:
            # convert to datetime and localize
            time = utc.localize(pd.to_datetime(json_extra[key], unit="ms"))
            # convert to correct time zone
            time = time.astimezone(tz)
            json_extra[key] = str(time)

    # fix wrong key for saliva_id in "alarm_ring" action (old app versions)
    if "extra_saliva_id" in json_extra:
        json_extra["saliva_id"] = json_extra["extra_saliva_id"]
        del json_extra["extra_saliva_id"]
    row_cpy["extras"] = json.dumps(json_extra)
    return row_cpy
