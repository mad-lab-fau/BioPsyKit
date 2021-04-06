"""Utility functions for working with saliva dataframes."""
import re

from typing import Optional, Dict, Sequence, Union, Tuple
from datetime import time, datetime

import pandas as pd
import numpy as np

from biopsykit.utils.functions import se


def extract_saliva_columns(data: pd.DataFrame, saliva_type: str, col_pattern: Optional[str] = None) -> pd.DataFrame:
    """Extract columns containing saliva samples from a DataFrame.

    Parameters
    ----------
    data: pd.DataFrame
        dataframe which should be extracted
    saliva_type: str
        saliva type variable which should be used to extract columns (e.g. 'cortisol')
    col_pattern: str, optional
        string pattern to identify saliva columns. If None, it is attempted to automatically infer column names using
        `bp.saliva.utils.get_saliva_column_suggestions()`

    Returns
    -------
    pd.DataFrame
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
    """Extract possible columns containing saliva data from a dataframe.

    This is for example useful when one large dataframe is used to store demographic information,
    questionnaire data and saliva data.

    Parameters
    ----------
    data: pd.DataFrame
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


def saliva_times_datetime_to_minute(saliva_times: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """Convert datetime objects into minute.

    Parameters
    ----------
    saliva_times :

    Returns
    -------
    dataframe

    """
    if isinstance(saliva_times.values.flatten()[0], str):
        saliva_times = pd.to_timedelta(saliva_times)

    if not isinstance(saliva_times.values.flatten()[0], (time, datetime, pd.Timedelta, np.timedelta64)):
        raise ValueError("Saliva times must be instance of `datetime.datetime()`, `datetime.time()` or `pd.Timedelta`!")

    if isinstance(saliva_times, pd.Series) and "sample" in saliva_times.index.names:
        # unstack the multi-index dataframe in the 'samples' level so that time differences can be computes in minutes.
        # Then stack it back together
        if isinstance(saliva_times[0], pd.Timedelta):
            saliva_times = pd.to_timedelta(saliva_times).unstack(level="sample").diff(axis=1)
        else:
            saliva_times = pd.to_datetime(saliva_times.astype(str)).unstack(level="sample").diff(axis=1)
        saliva_times = saliva_times.apply(lambda s: (s.dt.total_seconds() / 60))
        saliva_times = saliva_times.cumsum(axis=1)
        saliva_times.iloc[:, 0].fillna(0, inplace=True)
        saliva_times = saliva_times.stack()
        return saliva_times

    # assume saliva times are already unstacked in the 'samples' level
    saliva_times = saliva_times.astype(str).apply(pd.to_datetime).diff(axis=1)
    saliva_times = saliva_times.apply(lambda s: (s.dt.total_seconds() / 60))
    saliva_times.iloc[:, 0].fillna(0, inplace=True)
    return saliva_times


def mean_se(
    data: pd.DataFrame,
    biomarker_type: Optional[Union[str, Sequence[str]]] = "cortisol",
    remove_s0: Optional[bool] = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compute mean and standard error per saliva sample.

    Parameters
    ----------
    data :
    biomarker_type :
    remove_s0 :

    Returns
    -------
    dataframe

    """
    if isinstance(biomarker_type, list):
        dict_result = {}
        for biomarker in biomarker_type:
            biomarker_cols = [biomarker]
            if "time" in data:
                biomarker_cols = ["time"] + biomarker_cols
            dict_result[biomarker] = mean_se(data[biomarker_cols], biomarker_type=biomarker, remove_s0=remove_s0)
        return dict_result

    if remove_s0:
        data = data.drop(0, level="sample", errors="ignore")
        data = data.drop("0", level="sample", errors="ignore")
        data = data.drop("S0", level="sample", errors="ignore")

    group_cols = list(data.index.names)
    group_cols.remove("subject")

    if "time" in data:
        group_cols = group_cols + ["time"]
    data_grp = data.groupby(group_cols).agg([np.mean, se])[biomarker_type]
    return data_grp


def _check_data_format(data: pd.DataFrame):
    if data is None:
        raise ValueError("`data` must not be None!")
    if any(s not in data.index.names for s in ["subject", "sample"]) or data.index.nlevels <= 2:
        raise ValueError(
            "`data` is expected in long-format with subject IDs ('subject') as 1st and "
            "sample IDs ('sample') as 2nd index level!"
        )


def _check_saliva_times(saliva_times: np.array):
    if np.any(np.diff(saliva_times) <= 0):
        raise ValueError("`saliva_times` must be increasing!")


def _get_saliva_times(data: pd.DataFrame, saliva_times: np.array, remove_s0: bool) -> np.array:
    if saliva_times is None:
        # check if dataframe has 'time' column
        if "time" in data.columns:
            saliva_times = np.array(data.unstack()["time"])
            if np.all((saliva_times == saliva_times[0])):
                # all subjects have the same saliva times
                saliva_times = saliva_times[0]
        else:
            raise ValueError("No saliva times specified!")

    if isinstance(saliva_times, str):
        saliva_times = data[saliva_times]
        saliva_times = saliva_times.unstack(level="sample")

    # ensure numpy
    saliva_times = np.array(saliva_times)

    if remove_s0:
        # check whether we have the same saliva times for all subjects (1d array) or not (2d array)
        if saliva_times.ndim <= 2:
            saliva_times = saliva_times[..., 1:]
        else:
            raise ValueError("`saliva_times` has invalid dimensions: {}".format(saliva_times.ndim))

    return saliva_times


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
