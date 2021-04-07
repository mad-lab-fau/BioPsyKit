"""Utility functions for working with saliva dataframes."""
import re

from typing import Optional, Dict, Sequence, Union, Tuple
from datetime import time, datetime

import pandas as pd
import numpy as np
from biopsykit.utils.datatype_helper import SalivaRawDataFrame


def extract_saliva_columns(data: pd.DataFrame, saliva_type: str, col_pattern: Optional[str] = None) -> pd.DataFrame:
    """Extract saliva sample columns from a pandas dataframe.

    Parameters
    ----------
    data: :class:`pandas.DataFrame`
        dataframe to extract columns from
    saliva_type: str
        saliva type variable which should be used to extract columns (e.g. 'cortisol')
    col_pattern: str, optional
        string pattern to identify saliva columns. If ``None``, it is attempted to automatically infer column names
        using :func:`get_saliva_column_suggestions()`

    Returns
    -------
    :class:`pandas.DataFrame`
        dataframe containing saliva data

    """
    if col_pattern is None:
        col_suggs = get_saliva_column_suggestions(data, saliva_type)
        if len(col_suggs) > 1:
            raise KeyError(
                "More than one possible column pattern was found! "
                "Please check manually which pattern is correct: {}".format(col_suggs)
            )
        col_pattern = col_suggs[0]
    return data.filter(regex=col_pattern)


def get_saliva_column_suggestions(data: pd.DataFrame, saliva_type: str) -> Sequence[str]:
    """Automatically extract possible saliva data columns from a pandas dataframe.

    This is for example useful when one large dataframe is used to store demographic information,
    questionnaire data and saliva data.

    Parameters
    ----------
    data: :class:`pandas.DataFrame`
        dataframe which should be extracted
    saliva_type: str
        saliva type variable which should be used to extract columns (e.g. 'cortisol')

    Returns
    -------
    list
        list of suggested columns containing saliva data

    """
    sugg_filt = list(
        filter(
            lambda col: any(k in col for k in _dict_saliva_type_suggs[saliva_type]),
            data.columns,
        )
    )
    sugg_filt = list(filter(lambda s: any(str(i) in s for i in range(0, 20)), sugg_filt))
    sugg_filt = list(
        filter(
            lambda s: all(
                k not in s
                for k in (
                    "AUC",
                    "auc",
                    "TSST",
                    "max",
                    "log",
                    "inc",
                    "lg",
                    "ln",
                    "GenExp",
                    "inv",
                )
            ),
            sugg_filt,
        )
    )
    # replace il{} with il6 since this was removed out by the previous filter operation
    sugg_filt = [re.sub(r"\d", "{}", s).replace("il{}", "il6").replace("IL{}", "IL6") for s in sugg_filt]
    sugg_filt = sorted(list(filter(lambda s: "{}" in s, set(sugg_filt))))

    # build regex for column extraction
    sugg_filt = ["^{}$".format(s.replace("{}", r"(\d)")) for s in sugg_filt]
    return sugg_filt


def wide_to_long(
    data: pd.DataFrame,
    saliva_type: str,
    levels: Union[str, Sequence[str]],
    sep: Optional[str] = "_",
) -> pd.DataFrame:
    """Convert a dataframe containing saliva data from wide-format into long-format.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        pandas DataFrame containing saliva data in wide-format, i.e. one column per saliva sample, one row per subject
    saliva_type : str
        saliva type (e.g. 'cortisol')
    levels : str or list of str
        index levels of the resulting long-format dataframe. In the wide-format dataframe, the level keys are expected
        to be encoded in the column names and separated by ``sep``
    sep : str, optional
        character separating index levels in the column names of the wide-format dataframe

    Returns
    -------
    :class:`pandas.DataFrame`
        pandas DataFrame in long-format

    """
    if isinstance(levels, str):
        levels = [levels]

    data = data.filter(like=saliva_type)
    # reverse level order because nested multi-level index will be constructed from back to front
    levels = levels[::-1]
    # iteratively build up long-format dataframe
    for i, level in enumerate(levels):
        stubnames = list(data.columns)
        # stubnames are everything except the last part separated by underscore
        stubnames = sorted({"_".join(s.split("_")[:-1]) for s in stubnames})
        data = pd.wide_to_long(
            data.reset_index(),
            stubnames=stubnames,
            i=["subject"] + levels[0:i],
            j=level,
            sep=sep,
            suffix=r"\w+",
        )

    # reorder levels and sort
    return data.reorder_levels(["subject"] + levels[::-1]).sort_index()


def sample_times_datetime_to_minute(sample_times: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Convert sample times from datetime or timedelta objects into minutes.

    In order to compute certain saliva features (such as :func:`~biopsykit.saliva.saliva.auc` or
    :func:`~biopsykit.saliva.saliva.slope`) the saliva sampling times are needed.
    This function can be used to convert sampling times into minutes relative to the first saliva sample.

    Parameters
    ----------
    sample_times : :class:`pandas.Series` or :class:`pandas.DataFrame`
        saliva sampling times in a Python datetime- or timedelta-related format.
        If ``sample_times`` is a Series, it is assumed to be in long-format and will be unstacked into wide-format
        along the `sample` level.
        If ``sample_times`` is a DataFrame, it is assumed to be in wide-format already.
        If values in ``sample_times`` are ``str``, they are assumed to be strings with time information only
        (**not** including date), e.g., "09:00", "09:15", ...

    Returns
    -------
    :class:`pandas.DataFrame`
        dataframe in wide-format with saliva sampling times in minutes relative to the first saliva sample

    Raises
    ------
    ValueError
        if sample times are not in a datetime- or timedelta-related format

    """
    if isinstance(sample_times.values.flatten()[0], str):
        sample_times = pd.to_timedelta(sample_times)

    if not isinstance(sample_times.values.flatten()[0], (time, datetime, pd.Timedelta, np.timedelta64)):
        raise ValueError("Sample times must be instance of `datetime.datetime()`, `datetime.time()` or `pd.Timedelta`!")

    if isinstance(sample_times, pd.Series) and "sample" in sample_times.index.names:
        # unstack the multi-index dataframe in the 'samples' level so that time differences can be computes in minutes.
        # Then stack it back together
        if isinstance(sample_times[0], pd.Timedelta):
            sample_times = pd.to_timedelta(sample_times).unstack(level="sample").diff(axis=1)
        else:
            sample_times = pd.to_datetime(sample_times.astype(str)).unstack(level="sample").diff(axis=1)
        sample_times = sample_times.apply(lambda s: (s.dt.total_seconds() / 60))
        sample_times = sample_times.cumsum(axis=1)
        sample_times.iloc[:, 0].fillna(0, inplace=True)
        sample_times = sample_times.stack()
        return sample_times

    # assume sample times are already unstacked at the 'samples' level and provided as dataframe
    sample_times = sample_times.astype(str).apply(pd.to_datetime).diff(axis=1)
    sample_times = sample_times.apply(lambda s: (s.dt.total_seconds() / 60))
    sample_times.iloc[:, 0].fillna(0, inplace=True)
    return sample_times


def _remove_s0(data: SalivaRawDataFrame) -> SalivaRawDataFrame:
    """Helper function to remove first saliva sample.

    Parameters
    ----------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format

    Returns
    -------
    :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format without the first saliva sample

    """
    data = data.drop(0, level="sample", errors="ignore")
    data = data.drop("0", level="sample", errors="ignore")
    data = data.drop("S0", level="sample", errors="ignore")
    return data


def _check_sample_times(sample_times: np.array):
    if np.any(np.diff(sample_times) <= 0):
        raise ValueError("`sample_times` must be increasing!")


def _get_sample_times(
    data: pd.DataFrame, sample_times: Union[np.array, Sequence[int]], remove_s0: Optional[bool] = False
) -> np.array:
    if sample_times is None:
        # check if dataframe has 'time' column
        if "time" in data.columns:
            sample_times = np.array(data.unstack()["time"])
            if np.all((sample_times == sample_times[0])):
                # all subjects have the same saliva times
                sample_times = sample_times[0]
        else:
            raise ValueError("No sample times specified!")

    # ensure numpy
    sample_times = np.array(sample_times)

    if remove_s0:
        # check whether we have the same saliva times for all subjects (1d array) or not (2d array)
        if sample_times.ndim <= 2:
            sample_times = sample_times[..., 1:]
        else:
            raise ValueError("`sample_times` has invalid dimensions: {}".format(sample_times.ndim))

    return sample_times


def _get_saliva_idx_labels(
    columns: pd.Index,
    sample_labels: Optional[Union[Tuple, Sequence]] = None,
    sample_idx: Optional[Union[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[Sequence, Sequence]:

    if sample_labels is not None:
        try:
            sample_idx = [columns.get_loc(label) for label in sample_labels]
        except KeyError as e:
            raise IndexError("Invalid sample_labels `{}`".format(sample_labels)) from e
    else:
        try:
            # ensure list
            sample_idx = list(sample_idx)
            sample_labels = columns[sample_idx]
        except IndexError as e:
            raise IndexError("`sample_idx[1]` is out of bounds!") from e

    if len(sample_idx) != 2:
        raise IndexError("Exactly 2 indices needed for computing slope. Got {} indices.".format(len(sample_idx)))

    # replace idx values like '-1' with the actual index
    if sample_idx[0] < 0:
        sample_idx[0] = len(columns) + sample_idx[0]

    if sample_idx[1] < 0:
        sample_idx[1] = len(columns) + sample_idx[1]

    # check that second index is bigger than first index
    if sample_idx[0] >= sample_idx[1]:
        raise IndexError("`sample_idx[1]` must be bigger than `sample_idx[0]`. Got {}".format(sample_idx))

    return sample_labels, sample_idx


_dict_saliva_type_suggs: Dict[str, Sequence[str]] = {
    "cortisol": ["cortisol", "cort", "Cortisol", "_c_"],
    "amylase": ["amylase", "amy", "Amylase", "sAA"],
    "il6": ["il6", "IL6", "il-6", "IL-6", "il_6", "IL_6"],
}
"""Dictionary containing possible column patterns for different saliva types."""
