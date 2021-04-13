"""I/O functions for files related to ECG processing."""
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from biopsykit.io import write_pandas_dict_excel
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import HeartRateSubjectDict, is_hr_subject_dict
from biopsykit.utils.file_handling import is_excel_file, get_subject_dirs
from biopsykit.utils.time import tz

__all__ = ["load_hr_subject_dict", "load_hr_subject_dict_folder", "write_hr_subject_dict"]


def load_hr_subject_dict(file_path: path_t, assert_format: Optional[bool] = True) -> HeartRateSubjectDict:
    """Load Excel file containing time series heart rate data of one subject.

    The returned dictionary will be a
    :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`,
    i.e. a dict with heart rate data from one subject split into phases (as exported by :func:`write_hr_subject_dict`).

    Parameters
    ----------
    file_path : :any:`pathlib.Path` or str
        path to file
    assert_format : bool, optional
        whether to check if the imported dict is in the right format or not

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`
        Dict with heart rate data split into phases

    Raises
    ------
    :class:`~biopsykit.exceptions.ValidationError`
        if file in ``file_path`` is not a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`
        (if ``assert_format`` is ``True``)
    :class:`~biopsykit.exceptions.FileExtensionError`
        if file is no Excel file (.xls or .xlsx)

    See Also
    --------
    ~biopsykit.utils.datatype_helper.HeartRateSubjectDict : Dictionary format
    write_hr_subject_dict : Write `HeartRateSubject` dict to file

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, (".xls", ".xlsx"))

    # assure that the file is an Excel file
    is_excel_file(file_path)

    dict_hr: HeartRateSubjectDict = pd.read_excel(file_path, index_col="time", sheet_name=None)
    if assert_format:
        # assert that the dictionary is in the correct format
        is_hr_subject_dict(dict_hr)

    # (re-)localize each sheet since Excel does not support timezone-aware dates
    dict_hr = {k: v.tz_localize(tz) for k, v in dict_hr.items()}
    return dict_hr


def load_hr_subject_dict_folder(
    base_path: path_t,
    filename_pattern: str,
    subfolder_pattern: Optional[str] = None,
) -> Dict[str, HeartRateSubjectDict]:
    r"""Load a folder with ``HeartRateSubjectDict``s and concatenate them into one dictionary.

    This functions looks for all files that match the ``file_pattern`` in the folder specified by ``base_path``
    and loads the files that are expected to be :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`.

    Subject IDs are extracted from the file name. Hence, ``file_pattern`` needs to be a regex
    including a capture group, e.g. "ecg_results_(\w+).xlsx".

    Alternatively, if the files are stored in subfolders, the name pattern of these subfolders can be specified by
    ``subject_folder_pattern``. Then, it is expected that the subfolder names correspond to the subject IDs.

    The returned dictionary will have the following format:

    { ``subject_id`` : :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict` }


    Parameters
    ----------
    base_path : :any:`pathlib.Path` or str
        path to top-level folder containing all subject folders
    filename_pattern : str
        filename pattern of exported :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`.
        Must be a regex string with capture group to extract subject IDs, or a regular regex string
        (without capture group) if ``subfolder_pattern`` is specified
    subfolder_pattern : str, optional
        subfolder name pattern if files are stored in subfolders.
        Then, ``filename_pattern`` does **not** need to be a regex with a capture group because it is assumed that
        the names of the subfolders correspond to the subject IDs.

    Returns
    -------
    dict
        dictionary with :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict` of all subjects

    Raises
    ------
    :class:`~biopsykit.exceptions.ValidationError`
        if any file that matches ``filename_pattern`` is not a
        :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`
    :exc:`FileNotFoundError`
        if no files match ``filename_pattern`` or no subfolders match ``subfolder_pattern``

    See Also
    --------
    biopsykit.utils.file_handliner.get_subject_dirs : Filter for subject subfolders in a given folder
    load_hr_subject_dict : Load HeartRateSubjectDict for one subject

    Examples
    --------
    >>> from biopsykit.io.ecg import load_hr_subject_dict_folder
    >>> base_path = "./ecg_results/"
    >>> # Option 1: all files are stored in `base_path`, subject IDs are extracted from the file names
    >>> dict_hr_subjects = load_hr_subject_dict_folder(
    >>>                         base_path,
    >>>                         filename_pattern=r"ecg_result_(\w+).xlsx")
    >>> print(dict_hr_subjects)
    {
         'Vp01': {}, # HeartRateSubjectDict
         'Vp02': {},
         # ...
    }
    >>> # Option 2: files are stored in subfolders, the name of the subfolders is the corresponding subject ID
    >>> dict_hr_subjects = load_hr_subject_dict_folder(
    >>>                         base_path,
    >>>                         filename_pattern=r"ecg_result*.xlsx",
    >>>                         subfolder_pattern="Vp*")
    >>> print(dict_hr_subjects)
    {
         'Vp01': {}, # HeartRateSubjectDict
         'Vp02': {},
         # ...
    }

    """
    # ensure pathlib
    base_path = Path(base_path)

    dict_hr_subjects = {}
    if subfolder_pattern is None:
        file_list = list(sorted(base_path.glob("*")))
        file_list = [f for f in file_list if re.search(filename_pattern, f.name)]
        if len(file_list) == 0:
            raise FileNotFoundError(
                'No files matching the pattern "{}" found in {}.'.format(filename_pattern, base_path)
            )
        for file in file_list:
            subject_id = re.findall(filename_pattern, file.name)[0]
            dict_hr_subjects[subject_id] = load_hr_subject_dict(file)
    else:
        subject_dirs = get_subject_dirs(base_path, subfolder_pattern)
        if len(subject_dirs) == 0:
            raise FileNotFoundError(
                'No subfolders matching the pattern "{}" found in {}.'.format(subfolder_pattern, base_path)
            )
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            # first try to search for files with glob (assuming that a regex string without capture group was passed),
            # then try to search via regex search (assuming that a regex string with capture group was passed,
            # which should actually not be done if subfolder_pattern is passed)
            file_list = list(sorted(subject_dir.glob(filename_pattern)))
            if len(file_list) == 0:
                file_list = sorted(subject_dir.glob("*"))
                # then extract the ones that match
                file_list = [f for f in file_list if re.search(filename_pattern, f.name)]
            if len(file_list) == 1:
                dict_hr_subjects[subject_id] = load_hr_subject_dict(file_list[0])
            elif len(file_list) > 1:
                warnings.warn(
                    'More than one file matching file pattern "{}" found in folder {}. '
                    "Trying to merge these files into one HeartRateSubjectDict".format(filename_pattern, subject_dir)
                )
                dict_hr = {}
                for file in file_list:
                    dict_hr.update(load_hr_subject_dict(file))
                dict_hr_subjects[subject_id] = dict_hr
            else:
                print("No Heart Rate data for subject {}".format(subject_id))
    return dict_hr_subjects


def write_hr_subject_dict(hr_subject_dict: HeartRateSubjectDict, file_path: path_t) -> None:
    """Write :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict` to an Excel file.

    The :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict` is a dictionary with heart rate time
    series data split into phases.

    Each of the phases in the dictionary will be a separate sheet in the Excel file.

    Parameters
    ----------
    hr_subject_dict : :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`
        a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict` containing pandas dataframes
        with heart rate data
    file_path : :any:`pathlib.Path` or str
        path to export file

    See Also
    --------
    ~biopsykit.utils.datatype_helper.HeartRateSubjectDict : Dictionary format
    load_hr_subject_dict : Load `HeartRateSubjectDict` written to file
    ~biopsykit.io.write_pandas_dict_excel : Write dictionary with pandas dataframes to Excel file

    """
    # assert that file path is an Excel file
    is_excel_file(file_path)

    # assert that dict is in the right format
    is_hr_subject_dict(hr_subject_dict)
    write_pandas_dict_excel(hr_subject_dict, file_path)
