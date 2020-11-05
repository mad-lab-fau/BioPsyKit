# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
import warnings
from pathlib import Path
from typing import TypeVar, Sequence, Optional, Dict, Union, List
import pytz
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilspodlib import Dataset

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')

tz = pytz.timezone('Europe/Berlin')
utc = pytz.timezone('UTC')

cmap_fau = sns.color_palette(["#003865", "#c99313", "#8d1429", "#00b1eb", "#009b77", "#98a4ae"])
_keys_fau = ['fau', 'phil', 'wiso', 'med', 'nat', 'tech']


def cmap_fau_blue(cmap_type: Union[str, None]) -> Sequence[str]:
    palette_fau = sns.color_palette(
        ["#001628", "#001F38", "#002747", "#003056", "#003865",
         "#26567C", "#4D7493", "#7392AA", "#99AFC1", "#BFCDD9",
         "#E6EBF0"]
    )
    if cmap_type == '3':
        return palette_fau[1::3]
    elif cmap_type == '2':
        return palette_fau[5::4]
    elif cmap_type == '2_lp':
        return palette_fau[2::5]
    else:
        return palette_fau


def fau_color(key: str) -> str:
    return cmap_fau[_keys_fau.index(key)] or cmap_fau['fau']


def adjust_color(key: str, amount: Optional[float] = 1.5) -> str:
    import colorsys
    import matplotlib.colors as mc
    c = colorsys.rgb_to_hls(*mc.to_rgb(fau_color(key)))
    return mc.to_hex(colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))


def split_data(time_intervals: Union[pd.DataFrame, pd.Series, Dict[str, Sequence[str]]],
               dataset: Optional[Dataset] = None, df: Optional[pd.DataFrame] = None,
               timezone: Optional[Union[str, pytz.timezone]] = tz, include_start: Optional[bool] = False) -> Dict[
    str, pd.DataFrame]:
    """
    Splits the data into parts based on time intervals.

    Parameters
    ----------
    time_intervals : dict or pd.Series or pd.DataFrame
        time intervals indicating where the data should be split.
        Can either be a pandas Series or 1 row of a pandas Dataframe with the `start` times of the single phases
        (the names of the phases are then derived from the index in case of a Series or column names in case of a
        Dataframe) or a dictionary with tuples indicating start and end times of the phases
        (the names of the phases are then derived from the dict keys)
    dataset : Dataset, optional
        NilsPodLib dataset object to be split
    df : pd.DataFrame, optional
        data to be split
    timezone : str or pytz.timezone, optional
        timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')
    include_start: bool, optional
        ``True`` to include the data from the beginning of the recording to the first time interval as the
        first interval, ``False`` otherwise. Default: ``False``

    Returns
    -------
    dict
        Dictionary containing split data

    Examples
    --------
    >>> import biopsykit.utils as utils
    >>>
    >>> # Example 1: define time intervals (start and end) of the different recording phases as dictionary
    >>> time_intervals = {"Part1": ("09:00", "09:30"), "Part2": ("09:30", "09:45"), "Part3": ("09:45", "10:00")}
    >>> # Example 2: define time intervals as pandas Series. Here, only start times of the are required, it is assumed
    >>> # that the phases are back to back
    >>> time_intervals = pd.Series(data=["09:00", "09:30", "09:45", "10:00"], index=["Part1", "Part2", "Part3", "End"])
    >>>
    >>> # read pandas dataframe from csv file and split data based on time interval dictionary
    >>> df = pd.read_csv(path_to_file)
    >>> data_dict = utils.split_data(time_intervals, df=df)
    >>>
    >>> # Example: Get Part 2 of data_dict
    >>> print(data_dict['Part2'])
    """
    data_dict: Dict[str, pd.DataFrame] = {}
    if dataset is None and df is None:
        raise ValueError("Either 'dataset' or 'df' must be specified as parameter!")
    if dataset:
        if isinstance(timezone, str):
            # convert to pytz object
            timezone = pytz.timezone(timezone)
        df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utc).tz_convert(tz=timezone)
    if isinstance(time_intervals, pd.DataFrame):
        if len(time_intervals) > 1:
            raise ValueError("Only dataframes with 1 row allowed!")
        time_intervals = time_intervals.iloc[0]

    if isinstance(time_intervals, pd.Series):
        if include_start:
            time_intervals["Start"] = df.index[0].to_pydatetime().time()
        time_intervals.sort_values(inplace=True)
        for name, start, end in zip(time_intervals.index, np.pad(time_intervals, (0, 1)), time_intervals[1:]):
            data_dict[name] = df.between_time(start, end)
    else:
        if include_start:
            time_intervals["Start"] = (df.index[0].to_pydatetime().time(), list(time_intervals.values())[0][0])
        data_dict = {name: df.between_time(*start_end) for name, start_end in time_intervals.items()}
    return data_dict


def check_input(ecg_processor: 'EcgProcessor', key: str, ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame) -> bool:
    """
    Checks valid input, i.e. if either `ecg_processor` **and** `key` are supplied as arguments *or* `ecg_signal` **and**
    `rpeaks`. Used as helper method for several functions.

    Parameters
    ----------
    ecg_processor : EcgProcessor
        `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
    key : str
        Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
    ecg_signal : str
        dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
    rpeaks : str
        dataframe with R peaks. Output of `EcgProcessor.ecg_process()`

    Returns
    -------
    ``True`` if correct input was supplied, raises ValueError otherwise

    Raises
    ------
    ValueError
        if invalid input supplied
    """

    if all([x is None for x in [ecg_processor, key, ecg_signal, rpeaks]]):
        raise ValueError(
            "Either `ecg_processor` and `key` or `rpeaks` and `ecg_signal` must be passed as arguments!")
    if ecg_processor:
        if key is None:
            raise ValueError("`key` must be passed as argument when `ecg_processor` is passed!")

    return True


def read_time_log(file_path: path_t, index_cols: Optional[Union[str, Sequence[str]]] = None,
                  phase_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Loads time log file.

    Parameters
    ----------
    file_path : str or path
        path to time log file, either Excel or csv file
    index_cols : str list, optional
        column name (or list of column names) that should be used for dataframe index or ``None`` for no index.
        Default: ``None``
    phase_cols : list, optional
        list of column names that contain time log information or ``None`` to use all columns. Default: ``None``

    Returns
    -------
    pd.DataFrame
        pandas dataframe with time log information

    """
    # ensure pathlib
    file_path = Path(file_path)
    if file_path.suffix in ['.xls', '.xlsx']:
        df_time_log = pd.read_excel(file_path)
    elif file_path.suffix in ['.csv']:
        df_time_log = pd.read_csv(file_path)
    else:
        raise ValueError("Unrecognized file format {}!".format(file_path.suffix))

    if isinstance(index_cols, str):
        index_cols = [index_cols]

    if index_cols:
        df_time_log.set_index(index_cols, inplace=True)
    if phase_cols:
        df_time_log = df_time_log.loc[:, phase_cols]
    return df_time_log


def convert_time_log_datetime(time_log: pd.DataFrame, dataset: Optional['Dataset'] = None,
                              date: Optional[Union[str, 'datetime']] = None,
                              timezone: Optional[str] = "Europe/Berlin") -> pd.DataFrame:
    if dataset is None and date is None:
        raise ValueError("Either `dataset` or `date` must be supplied as argument!")

    if dataset is not None:
        date = dataset.info.utc_datetime_start.date()
    if isinstance(date, str):
        # ensure datetime
        date = datetime.datetime(date)
    time_log = time_log.applymap(lambda x: pytz.timezone(timezone).localize(datetime.datetime.combine(date, x)))
    return time_log


def write_hr_to_excel(ecg_processor: 'EcgProcessor', file_path: path_t) -> None:
    """
    Writes heart rate dictionary of one subject to an Excel file.
    Each of the phases in the dictionary will be a separate sheet in the file.

    The Excel file will have the following columns:
        * date: timestamps of the heart rate samples (string, will be converted to DateTimeIndex)
        * ECG_Rate: heart rate samples (float)


    Parameters
    ----------
    ecg_processor : EcgProcessor
        EcgProcessor instance
    file_path : path or str
        path to file
    """

    write_dict_to_excel(ecg_processor.heart_rate, file_path)


def write_dict_to_excel(data_dict: Dict[str, pd.DataFrame], file_path: path_t,
                        index_col: Optional[bool] = True) -> None:
    """
    Writes a dictionary containing pandas dataframes to an Excel file.

    Parameters
    ----------
    data_dict : dict
        dict with pandas dataframes
    file_path : str or path
        filepath
    index_col : bool, optional
        ``True`` to include dataframe index in Excel export, ``False`` otherwise. Default: ``True``
    """

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    for key in data_dict:
        if isinstance(data_dict[key].index, pd.DatetimeIndex):
            # un-localize DateTimeIndex because Excel doesn't support timezone-aware dates
            data_dict[key].tz_localize(None).to_excel(writer, sheet_name=key, index=index_col)
        else:
            data_dict[key].to_excel(writer, sheet_name=key, index=index_col)
    writer.save()


def load_hr_excel(file_path: path_t) -> Dict[str, pd.DataFrame]:
    """
    Loads excel file containing heart rate data of one subject (as exported by `write_hr_to_excel`).

    The dictionary will have the following format: { "<Phase>" : hr_dataframe }

    Each hr_dataframe has the following format:
        * 'date' Index: DateTimeIndex with timestamps of the heart rate samples
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
    dict_hr = pd.read_excel(file_path, index_col="date", sheet_name=None)
    # (re-)localize each sheet since Excel does not support timezone-aware dates
    dict_hr = {k: v.tz_localize(tz) for k, v in dict_hr.items()}
    return dict_hr


def load_hr_excel_all_subjects(base_path: path_t, subject_folder_pattern: str,
                               filename_pattern: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads HR processing results (as exported by `write_hr_to_excel`) from all subjects and combines them into one
    dictionary ('HR subject dict').

    The dictionary will have the following format:

    { "<Subject_ID>" : {"<Phase>" : hr_dataframe } }@


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
    >>> import biopsykit as ep
    >>> base_path = "./"
    >>> dict_hr_subjects = ep.load_hr_excel_all_subjects(
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
    subject_dirs = list(sorted(base_path.glob(subject_folder_pattern)))
    dict_hr_subjects = {}
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        # check whether old processing results already exist
        hr_files = sorted(subject_dir.glob(filename_pattern))
        if len(hr_files) == 1:
            dict_hr_subjects[subject_id] = load_hr_excel(hr_files[0])
        else:
            print("No HR data for subject {}".format(subject_id))
    return dict_hr_subjects


def normalize_heart_rate(dict_hr_subjects: Dict[str, Dict[str, pd.DataFrame]],
                         normalize_to: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Normalizes heart rate per subject to the phase specified by `normalize_to`.

    The result is the relative change of heart rate compared to the mean heart rate in the `normalize_to` phase.

    Parameters
    ----------
    dict_hr_subjects : dict
        dictionary with heart rate data of all subjects as returned by `load_hr_excel_all_subjects`
    normalize_to : str
        phase (i.e., dict key) of data to normalize all other data to

    Returns
    -------
    dict
        dictionary with normalized heart rate data per subject
    """

    dict_hr_subjects_norm = {}
    for subject_id, dict_hr in dict_hr_subjects.items():
        bl_mean = dict_hr[normalize_to].mean()
        dict_hr_norm = {phase: (df_hr - bl_mean) / bl_mean * 100 for phase, df_hr in dict_hr.items()}
        dict_hr_subjects_norm[subject_id] = dict_hr_norm

    return dict_hr_subjects_norm


def write_result_dict_to_csv(result_dict: Dict[str, pd.DataFrame], file_path: path_t,
                             identifier_col: Optional[str] = "Subject_ID",
                             index_cols: Optional[List[str]] = ["Phase", "Subphase"],
                             overwrite_file: Optional[bool] = False) -> None:
    """
    Saves dictionary with processing results (e.g. HR, HRV, RSA) of all subjects as csv file.

    Simply pass a dictionary with processing results. The keys in the dictionary should be the Subject IDs
    (or any other identifier), the values should be pandas dataframes. The resulting index can be specified by the
    `identifier_col` parameter.

    The dictionary will be concatenated to one large dataframe which will then be saved as csv file.

    *Notes*:
        * If a file with same name exists at the specified location, it is assumed that this is a result file from a
          previous run and the current results should be appended to this file.
          (To disable this behavior, set `overwrite_file` to ``False``).
        * Per default, it is assumed that the 'values' dataframes has a multi-index with columns ["Phase", "Subphase"].
          The resulting dataframe would, thus, per default have the columns ["Subject_ID", "Phase", "Subphase"].
          This can be changed by the parameter `index_cols`.

    Parameters
    ----------
    result_dict : dict
        Dictionary containing processing results for all subjects. The keys in the dictionary should be the Subject IDs
        (or any other identifier), the values should be pandas dataframes
    file_path : path, str
        path to file
    identifier_col : str, optional
        Name of the index in the concatenated dataframe. Default: "Subject_ID"
    index_cols : list of str, optional
        List of index columns of the single dataframes in the dictionary. Not needed if `overwrite_file` is ``False``.
        Default: ["Phase", "Subphase"]
    overwrite_file : bool, optional
        ``True`` to overwrite the file if it already exists, ``False`` otherwise. Default: ``True``

    Examples
    --------
    >>>
    >>> import biopsykit as ep
    >>>
    >>> file_path = "./param_results.csv"
    >>>
    >>> dict_param_output = {
    >>> 'S01' : pd.DataFrame(), # dataframe from mist_param_subphases,
    >>> 'S02' : pd.DataFrame(),
    >>> # ...
    >>> }
    >>>
    >>> ep.write_result_dict_to_csv(dict_param_output, file_path=file_path,
    >>>                             identifier_col="Subject_ID", index_cols=["Phase", "Subphase"])
    """

    # TODO check if index_cols is really needed?

    identifier_col = [identifier_col]

    if index_cols is None:
        index_cols = []

    df_result_concat = pd.concat(result_dict, names=identifier_col)
    if file_path.exists() and not overwrite_file:
        df_result_old = pd.read_csv(file_path, index_col=identifier_col + index_cols)
        df_result_concat = df_result_concat.combine_first(df_result_old).sort_index(level=0)
    df_result_concat.reset_index().to_csv(file_path, index=False)


def export_figure(fig: plt.Figure, filename: path_t, base_dir: path_t, formats: Optional[Sequence[str]] = None,
                  use_subfolder: Optional[bool] = True) -> None:
    """
    Exports a figure to a file.

    Parameters
    ----------
    fig : Figure
        matplotlib figure object
    filename : path or str
        name of the output file
    base_dir : path or str
        base directory to save file
    formats: list of str, optional
        list of file formats to export or ``None`` to export as pdf. Default: ``None``
    use_subfolder : bool, optional
        whether to create an own subfolder per file format and export figures into these subfolders. Default: True

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import biopsykit.utils as utils
    >>> fig = plt.Figure()
    >>>
    >>> base_dir = "./img"
    >>> filename = "plot"
    >>> formats = ["pdf", "png"]

    >>> # Export into subfolders (default)
    >>> utils.export_figure(fig, filename=filename, base_dir=base_dir, formats=formats)
    >>> # | img/
    >>> # | - pdf/
    >>> # | - - plot.pdf
    >>> # | - png/
    >>> # | - - plot.png

    >>> # Export into one folder
    >>> utils.export_figure(fig, filename=filename, base_dir=base_dir, formats=formats, use_subfolder=False)
    >>> # | img/
    >>> # | - plot.pdf
    >>> # | - plot.png
    """
    if formats is None:
        formats = ['pdf']

    # ensure pathlib
    base_dir = Path(base_dir)
    filename = Path(filename)
    subfolders = [base_dir] * len(formats)

    if use_subfolder:
        subfolders = [base_dir.joinpath(f) for f in formats]
        for folder in subfolders:
            folder.mkdir(exist_ok=True, parents=True)

    for f, subfolder in zip(formats, subfolders):
        fig.savefig(subfolder.joinpath(filename.name + '.' + f), transparent=(f == 'pdf'), format=f)


def sanitize_input(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """
    Converts 1D array-like data (numpy array, pandas dataframe/series) to a numpy array.

    Parameters
    ----------
    data : array_like
        input data. Needs to be 1D

    Returns
    -------
    array_like
        data as numpy array

    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # only 1D pandas DataFrame allowed
        if isinstance(data, pd.DataFrame) and len(data.columns) != 1:
            raise ValueError("Only 1D DataFrames allowed!")
        data = np.squeeze(data.values)

    return data