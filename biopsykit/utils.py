# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
from pathlib import Path
from typing import TypeVar, Sequence, Optional, Dict, Union, Tuple
import pytz

import pandas as pd
import numpy as np
from nilspodlib import Dataset
import warnings

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')

tz = pytz.timezone('Europe/Berlin')
utc = pytz.timezone('UTC')


def export_figure(fig: 'plt.Figure', filename: path_t, base_dir: path_t, formats: Optional[Sequence[str]] = None,
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
    >>> import biopsykit as bp
    >>> fig = plt.Figure()
    >>>
    >>> base_dir = "./img"
    >>> filename = "plot"
    >>> formats = ["pdf", "png"]

    >>> # Export into subfolders (default)
    >>> bp.plotting.export_figure(fig, filename=filename, base_dir=base_dir, formats=formats)
    >>> # | img/
    >>> # | - pdf/
    >>> # | - - plot.pdf
    >>> # | - png/
    >>> # | - - plot.png

    >>> # Export into one folder
    >>> bp.plotting.export_figure(fig, filename=filename, base_dir=base_dir, formats=formats, use_subfolder=False)
    >>> # | img/
    >>> # | - plot.pdf
    >>> # | - plot.png
    """
    import matplotlib.pyplot as plt

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


def split_data(time_intervals: Union[pd.DataFrame, pd.Series, Dict[str, Sequence[str]]],
               dataset: Optional[Dataset] = None, df: Optional[pd.DataFrame] = None,
               timezone: Optional[Union[str, pytz.timezone]] = tz,
               include_start: Optional[bool] = False) -> Dict[str, pd.DataFrame]:
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
    >>> import biopsykit.su as utils
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


def sanitize_input_1d(data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
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


def sanitize_input_nd(
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        ncols: Union[int, Tuple[int, ...]]
) -> np.ndarray:
    """
    Converts nD array-like data (numpy array, pandas dataframe/series) to a numpy array.

    Parameters
    ----------
    data : array_like
        input data
    ncols : int or tuple of ints
        number of columns (2nd dimension) the 'data' array should have

    Returns
    -------
    array_like
        data as numpy array
    """

    # ensure tuple
    if isinstance(ncols, int):
        ncols = (ncols,)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = np.squeeze(data.values)

    if data.ndim == 1:
        if 1 in ncols:
            return data
        else:
            raise ValueError("Invalid number of columns! Expected one of {}, got 1.".format(ncols))
    elif data.shape[1] not in ncols:
        raise ValueError("Invalid number of columns! Expected one of {}, got {}.".format(ncols, data.shape[1]))
    return data


def check_tz_aware(data: pd.DataFrame) -> bool:
    return isinstance(data.index, pd.DatetimeIndex) and (data.index.tzinfo is not None)


def exclude_subjects(excluded_subjects: Union[Sequence[str], Sequence[int]],
                     **kwargs) -> Dict[str, pd.DataFrame]:
    cleaned_data: Dict[str, pd.DataFrame] = {}

    for key, data in kwargs.items():
        if 'subject' in data.index.names:
            if (data.index.get_level_values('subject').dtype == np.object and all(
                    [isinstance(s, str) for s in excluded_subjects])) or (
                    data.index.get_level_values('subject').dtype == np.int and
                    all([isinstance(s, int) for s in excluded_subjects])):
                # dataframe index and subjects are both strings or both integers
                try:
                    if isinstance(data.index, pd.MultiIndex):
                        # MultiIndex => specify index level
                        cleaned_data[key] = data.drop(index=excluded_subjects, level='subject')
                    else:
                        # Regular Index
                        cleaned_data[key] = data.drop(index=excluded_subjects)
                except KeyError:
                    warnings.warn("Not all subjects of {} exist in the dataset!".format(excluded_subjects))
            else:
                raise ValueError("{}: dtypes of index and subject ids to be excluded do not match!".format(key))
        else:
            raise ValueError("No 'subject' level in index!")
    return cleaned_data
