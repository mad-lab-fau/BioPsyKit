from pathlib import Path
from typing import Dict, Union

import pandas as pd

from biopsykit.io import write_pandas_dict_excel
from biopsykit.utils import path_t
from biopsykit.utils.time import tz


def load_hr_subject_dict(file_path: path_t) -> Dict[str, pd.DataFrame]:
    """
    Loads excel file containing heart rate data of one subject (as exported by `write_hr_subject`).

    The dictionary will have the following format: { "<Phase>" : hr_dataframe }

    Each hr_dataframe has the following format:
        * 'time' Index: DateTimeIndex with timestamps of the heart rate samples
        * 'ECG_Rate' Column: heart rate samples

    Parameters
    ----------
    file_path : path or str
        path to file

    Returns
    -------
    dict
        Excel sheet dictionary
    """
    # ensure pathlib
    file_path = Path(file_path)

    dict_hr = pd.read_excel(file_path, index_col="time", sheet_name=None)
    # (re-)localize each sheet since Excel does not support timezone-aware dates
    dict_hr = {k: v.tz_localize(tz) for k, v in dict_hr.items()}
    return dict_hr


def load_combine_hr_all_subjects(base_path: path_t, subject_folder_pattern: str,
                                 filename_pattern: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads HR processing results (as exported by `write_hr_subject_dict`) from all subjects and combines them into one
    dictionary ('HR subject dict').

    The dictionary will have the following format:

    { "<Subject_ID>" : {"<Phase>" : hr_dataframe } }


    Parameters
    ----------
    base_path : str or path
        path to top-level folder containing all subject folders
    subject_folder_pattern : str
        subject folder pattern. Folder names are assumed to be Subject IDs
    filename_pattern : str
        filename pattern of HR result files

    Returns
    -------

    Examples
    --------
    >>> from biopsykit.io.ecg import load_combine_hr_all_subjects
    >>> base_path = "../signals/ecg/"
    >>> dict_hr_subjects = load_combine_hr_all_subjects(
    >>>                         base_path, subject_folder_pattern="Vp_*",
    >>>                         filename_pattern="ecg_result*.xlsx")
    >>>
    >>> print(dict_hr_subjects)
    >>> {
    >>>     'Vp_01': {}, # dict as returned by load_hr_excel
    >>>     'Vp_02': {},
    >>>     # ...
    >>> }
    """
    # ensure pathlib
    base_path = Path(base_path)

    subject_dirs = list(sorted(base_path.glob(subject_folder_pattern)))
    dict_hr_subjects = {}
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        # check whether old processing results already exist
        hr_files = sorted(subject_dir.glob(filename_pattern))
        if len(hr_files) == 1:
            dict_hr_subjects[subject_id] = load_hr_subject_dict(hr_files[0])
        else:
            print("No HR data for subject {}".format(subject_id))
    return dict_hr_subjects


def write_hr_subject_dict(ep_or_dict: Union['EcgProcessor', Dict[str, pd.DataFrame]], file_path: path_t) -> None:
    """
    Writes heart rate dictionary of one subject to an Excel file.
    Each of the phases in the dictionary will be a separate sheet in the file.

    The Excel file will have the following columns:
        * date: timestamps of the heart rate samples (string, will be converted to DateTimeIndex)
        * ECG_Rate: heart rate samples (float)

    Parameters
    ----------
    ep_or_dict : EcgProcessor or dict
        EcgProcessor instance or 'HR subject dict'
    file_path : path or str
        path to file
    """
    from biopsykit.signals.ecg import EcgProcessor

    if isinstance(ep_or_dict, EcgProcessor):
        hr_subject_dict = ep_or_dict.heart_rate
    else:
        hr_subject_dict = ep_or_dict
    write_pandas_dict_excel(hr_subject_dict, file_path)
