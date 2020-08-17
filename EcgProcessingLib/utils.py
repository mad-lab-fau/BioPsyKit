# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
from pathlib import Path
from typing import TypeVar, Sequence, Optional, Dict, Union, List
import pytz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from NilsPodLib import Dataset

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
    if cmap_type is '3':
        return palette_fau[1::3]
    elif cmap_type is '2':
        return palette_fau[5::4]
    elif cmap_type is '2_lp':
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


def split_data(time_intervals: Union[pd.Series, Dict[str, Sequence[str]]],
               dataset: Optional[Dataset] = None, df: Optional[pd.DataFrame] = None,
               timezone: Optional[Union[str, pytz.timezone]] = tz) -> Dict[str, pd.DataFrame]:
    """
    Splits the data into parts based on time intervals.

    Parameters
    ----------
    time_intervals : dict or pd.Series
        time intervals indicating where the data should be split.
        Can either be a pandas Series with the `start` times of the single phases
        (the names of the phases are then derived from the index) or a dictionary of lists of tuples indicating
        start and end times of the phases (the names of the phases are then derived from the dict keys)
    dataset : Dataset
        NilsPodLib dataset object to be split
    df : pd.DataFrame, optional
        data to be split
    timezone : str or pytz.timezone, optional
        timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')

    Returns
    -------
    dict
        Dictionary containing split data

    """
    data_dict: Dict[str, pd.DataFrame] = {}
    if dataset is None and df is None:
        raise ValueError("Either 'dataset' or 'df' must be specified as parameter!")
    if dataset:
        if isinstance(timezone, str):
            # convert to pytz object
            timezone = pytz.timezone(timezone)
        df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utc).tz_convert(tz=timezone)
    if isinstance(time_intervals, pd.Series):
        for name, start, end in zip(time_intervals.index, np.pad(time_intervals, (0, 1)), time_intervals[1:]):
            data_dict[name] = df.between_time(start, end)
    else:
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


def write_hr_to_excel(ecg_processor: 'EcgProcessor', file_path: path_t) -> None:
    """
    Writes heart rate dictionary of EcgProcessor instance to an Excel file.
    Each of the phases in the dictionary will be a separate sheet in the file.


    Parameters
    ----------
    ecg_processor : EcgProcessor
        EcgProcessor instance
    file_path : path or str
        path to file

    """
    # ensure pathlib
    file_path = Path(file_path)

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    for label, df_hr in ecg_processor.heart_rate.items():
        # un-localize because Excel doesn't support timezone-aware dates
        df_hr.tz_localize(None).to_excel(writer, sheet_name=label)
    writer.save()


def load_hr_excel(file_path: path_t) -> Dict[str, pd.DataFrame]:
    """
    Loads excel file containing heart rate data (as exported by `write_hr_to_excel`).

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
        fig.savefig(subfolder.joinpath(filename.name + '.' + f), transparent=(f is 'pdf'), format=f)
