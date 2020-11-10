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


def check_tz_aware(data: pd.DataFrame) -> bool:
    return data.index.tzinfo is not None
