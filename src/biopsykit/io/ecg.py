"""I/O functions for files related to ECG processing."""
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_is_dir
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import HeartRatePhaseDict, HeartRateSubjectDataDict, is_hr_phase_dict
from biopsykit.utils.file_handling import get_subject_dirs, is_excel_file
from biopsykit.utils.time import tz

__all__ = [
    "load_hr_phase_dict",
    "load_hr_phase_dict_folder",
    "load_hr_phase_dict_csv",
    "write_hr_phase_dict",
    "write_hr_phase_dict_csv",
]


def load_hr_phase_dict(file_path: path_t, assert_format: Optional[bool] = True) -> HeartRatePhaseDict:
    """Load Excel file containing time series heart rate data of one subject.

    The returned dictionary will be a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`,
    i.e., a dict with heart rate data from one subject split into phases (as exported by :func:`write_hr_phase_dict`
    or :func:`write_hr_phase_dict_csv`).

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    assert_format : bool, optional
        whether to check if the imported dict is in the right format or not

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        Dict with heart rate data split into phases

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if file in ``file_path`` is not a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        (if ``assert_format`` is ``True``)
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if file is no Excel file (`.xls` or `.xlsx`)

    See Also
    --------
    ~biopsykit.utils.datatype_helper.HeartRatePhaseDict : Dictionary format
    write_hr_phase_dict : Write :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` to file

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, (".xls", ".xlsx"))

    # assure that the file is an Excel file
    is_excel_file(file_path)

    dict_hr: HeartRatePhaseDict = pd.read_excel(file_path, index_col="time", sheet_name=None)
    if assert_format:
        # assert that the dictionary is in the correct format
        is_hr_phase_dict(dict_hr)

    # (re-)localize each sheet since Excel does not support timezone-aware dates
    dict_hr = {k: v.tz_localize(tz) for k, v in dict_hr.items()}
    return dict_hr


def load_hr_phase_dict_folder(
    base_path: path_t, filename_pattern: str, subfolder_pattern: Optional[str] = None
) -> HeartRateSubjectDataDict:
    """Load a folder with multiple :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` and concatenate them \
    into a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`.

    This functions looks for all files that match the ``file_pattern`` in the folder specified by ``base_path``
    and loads the files that are all expected to be :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`.

    Subject IDs are extracted from the file name. Hence, ``file_pattern`` needs to be a regex
    including a capture group, e.g. "ecg_results_(\\w+).xlsx".

    Alternatively, if the files are stored in subfolders, the name pattern of these subfolders can be specified by
    ``subject_folder_pattern``. Then, it is expected that the subfolder names correspond to the subject IDs.

    The returned dictionary will be a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`
    with the following format:

    { ``subject_id`` : :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` }


    Parameters
    ----------
    base_path : :class: `~pathlib.Path` or str
        path to top-level folder containing all subject folders
    filename_pattern : str
        filename pattern of exported :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`.
        Must be a regex string with capture group to extract subject IDs, or a regular regex string
        (without capture group) if ``subfolder_pattern`` is specified
    subfolder_pattern : str, optional
        subfolder name pattern if files are stored in subfolders.
        Then, ``filename_pattern`` does **not** need to be a regex with a capture group because it is assumed that
        the names of the subfolders correspond to the subject IDs.

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`
        :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`, i.e., a dictionary with
        :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` of multiple subjects

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if any file that matches ``filename_pattern`` is not a
        :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
    :exc:`FileNotFoundError`
        if no files match ``filename_pattern`` or no subfolders match ``subfolder_pattern``

    See Also
    --------
    ~biopsykit.utils.file_handling.get_subject_dirs : Filter for subject subfolders in a given folder
    load_hr_phase_dict : Load :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` for one subject

    Examples
    --------
    >>> from biopsykit.io.ecg import load_hr_phase_dict_folder
    >>> base_path = "./ecg_results/"
    >>> # Option 1: all files are stored in `base_path`, subject IDs are extracted from the file names
    >>> dict_hr_subjects = load_hr_phase_dict_folder(
    >>>                         base_path,
    >>>                         filename_pattern=r"ecg_result_(\\w+).xlsx")
    >>> print(dict_hr_subjects)
    {
         'Vp01': {}, # one single HeartRatePhaseDict
         'Vp02': {},
         # ...
    }
    >>> # Option 2: files are stored in subfolders, the name of the subfolders is the corresponding subject ID
    >>> dict_hr_subjects = load_hr_phase_dict_folder(
    >>>                         base_path,
    >>>                         filename_pattern=r"ecg_result*.xlsx",
    >>>                         subfolder_pattern="Vp*")
    >>> print(dict_hr_subjects)
    {
         'Vp01': {}, # one single HeartRatePhaseDict
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
                "No files matching the pattern '{}' found in {}.".format(filename_pattern, base_path)
            )
        for file in file_list:
            subject_id = re.findall(filename_pattern, file.name)[0]
            dict_hr_subjects[subject_id] = load_hr_phase_dict(file)
    else:
        subject_dirs = get_subject_dirs(base_path, subfolder_pattern)

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            hr_phase_dict = _load_hr_phase_dict_single_subject(subject_dir, filename_pattern)
            if hr_phase_dict is None:
                continue
            dict_hr_subjects[subject_id] = hr_phase_dict
    return dict_hr_subjects


def _load_hr_phase_dict_single_subject(subject_dir: Path, filename_pattern: str) -> Optional[HeartRatePhaseDict]:
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
        return load_hr_phase_dict(file_list[0])
    if len(file_list) > 1:
        warnings.warn(
            'More than one file matching file pattern "{}" found in folder {}. '
            "Trying to merge these files into one HeartRatePhaseDict".format(filename_pattern, subject_dir)
        )
        dict_hr = {}
        for file in file_list:
            dict_hr.update(load_hr_phase_dict(file))
        return dict_hr

    print("No Heart Rate data for subject {}".format(subject_id))
    return None


def write_hr_phase_dict(hr_phase_dict: HeartRatePhaseDict, file_path: path_t):
    """Write :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` to an Excel file.

    The :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` is a dictionary with heart rate time
    series data split into phases.

    Each of the phases in the dictionary will be a separate sheet in the Excel file.

    Parameters
    ----------
    hr_phase_dict : :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` containing pandas dataframes
        with heart rate data
    file_path : :class:`~pathlib.Path` or str
        path to export file

    See Also
    --------
    ~biopsykit.utils.datatype_helper.HeartRatePhaseDict : Dictionary format
    load_hr_phase_dict : Load :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` written to file
    ~biopsykit.io.write_pandas_dict_excel : Write dictionary with pandas dataframes to Excel file

    """
    from biopsykit.io import write_pandas_dict_excel  # pylint:disable=import-outside-toplevel

    # assert that file path is an Excel file
    is_excel_file(file_path)

    # assert that dict is in the right format
    is_hr_phase_dict(hr_phase_dict)
    write_pandas_dict_excel(hr_phase_dict, file_path)


def write_hr_phase_dict_csv(hr_phase_dict: HeartRatePhaseDict, folder_path: path_t, file_pattern: path_t):
    """Write :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` to a series of csv files.

    The :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` is a dictionary with heart rate time
    series data split into phases.

    Each of the phases in the dictionary will be a separate csv file.

    Parameters
    ----------
    hr_phase_dict : :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` containing pandas dataframes
        with heart rate data
    folder_path : :class:`~pathlib.Path` or str
        folder path to export csv files
    file_pattern : :class:`~pathlib.Path` or str
        file pattern to save. The file pattern must include a placeholder "{}" which will be filled with the
        name of the phase.

    Raises
    ------
    ValueError
        if ``file_pattern`` does not include a placeholder "{}" that can be filled with the phase name
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``hr_phase_dict`` is not a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if ``file_pattern`` is no csv file

    See Also
    --------
    ~biopsykit.utils.datatype_helper.HeartRatePhaseDict : Dictionary format
    load_hr_phase_dict_csv : Load :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` written to csv files

    """
    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)

    # check file pattern for sanity
    file_pattern = str(file_pattern)
    if "{}" not in file_pattern:
        raise ValueError(
            f"'file_pattern' must include a placeholder to be filled with the phase name!. Got {file_pattern}."
        )
    is_hr_phase_dict(hr_phase_dict)

    for phase, data in hr_phase_dict.items():
        file_name = Path(file_pattern.format(phase))
        file_path = folder_path.joinpath(file_name)
        _assert_file_extension(file_path, ".csv")
        data.to_csv(file_path)


def load_hr_phase_dict_csv(
    folder_path: path_t,
    file_pattern: path_t,
    phase_order: Optional[Sequence[str]] = None,
    assert_format: Optional[bool] = True,
) -> HeartRatePhaseDict:
    """Load csv file containing time series heart rate data of one subject from folder and combine it into a \
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`.

    The returned dictionary will be a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`,
    i.e., a dict with heart rate data from one subject split into phases (as exported by :func:`write_hr_phase_dict`).

    Parameters
    ----------
    folder_path : :class:`~pathlib.Path` or str
        folder path to export csv files
    file_pattern : :class:`~pathlib.Path` or str
        file pattern of the csv files. `file_pattern` must include a regex capture group (see Examples) which is used
        to extract the phase name from the file name.
    phase_order : list of str, optional
        list of phase names to order resulting :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` or ``None``
        to order phases according to the file name ordering in ``folder_path``.
        Default: ``None``
    assert_format : bool, optional
        whether to check if the imported dict is in the right format or not

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        Dict with heart rate data split into phases

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if file in ``file_path`` is not a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        (if ``assert_format`` is ``True``)
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if file is no Excel file (`.xls` or `.xlsx`)

    See Also
    --------
    ~biopsykit.utils.datatype_helper.HeartRatePhaseDict : Dictionary format
    write_hr_phase_dict_csv : Write :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` to a series of csv files

    """
    # ensure pathlib
    folder_path = Path(folder_path)
    _assert_is_dir(folder_path)

    file_list = sorted(folder_path.glob("*.csv"))
    if len(file_list) == 0:
        raise ValueError(f"No csv files found in {folder_path}!")

    dict_hr = {}
    for file in file_list:
        phase_name = re.findall(file_pattern, file.name)
        if len(phase_name) == 0:
            raise ValueError(f"No string matching the pattern {file_pattern} found in {file}!")
        if len(phase_name) > 1:
            raise ValueError(f"Too many strings matching the pattern {file_pattern} found in {file}!")
        phase_name = phase_name[0]
        df = pd.read_csv(file, index_col="time")
        df.index = pd.to_datetime(df.index, errors="ignore")
        dict_hr[phase_name] = df

    if assert_format:
        # assert that the dictionary is in the correct format
        is_hr_phase_dict(dict_hr)

    if phase_order is not None:
        _assert_phase_names(folder_path, phase_order, dict_hr.keys())
        dict_hr = {phase: dict_hr[phase] for phase in phase_order}

    return dict_hr


def _assert_phase_names(folder_path, phase_order, dict_keys):
    if len(phase_order) != len(dict_keys):
        raise ValueError(
            f"Number of phases provided by 'phase_order' do not match. "
            f"Expected {len(dict_keys)}, got {len(phase_order)}."
        )
    if set(phase_order) != set(dict_keys):
        raise ValueError(
            f"Phases provided by 'phase_order' do not match the phases of the files in {folder_path}. "
            f"Expected {phase_order}, got {list(dict_keys)}."
        )
