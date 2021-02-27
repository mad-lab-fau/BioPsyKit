import warnings
from typing import Sequence, Union, Dict, Optional

import numpy as np
import pandas as pd
import pytz
from nilspodlib import Dataset

from biopsykit.utils.time import tz, utc


def multi_xs(data: pd.DataFrame, keys: Sequence[str], level: str) -> pd.DataFrame:
    levels = data.index.names
    data_xs = pd.concat({key: data.xs(key, level=level) for key in keys}, names=[level])
    return data_xs.reorder_levels(levels).sort_index()


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


def exclude_subjects(excluded_subjects: Union[Sequence[str], Sequence[int]],
                     **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
    if len(cleaned_data) == 1:
        cleaned_data = list(cleaned_data.values())[0]
    return cleaned_data
