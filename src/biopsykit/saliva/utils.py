"""Utility functions for working with saliva dataframes."""
import re

from typing import Optional, Dict, Sequence, Union, Tuple, List
from datetime import time, datetime

import pandas as pd
import numpy as np

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_has_index_levels
from biopsykit.utils.datatype_helper import SalivaRawDataFrame, SalivaFeatureDataFrame, _SalivaRawDataFrame
from biopsykit.utils._types import arr_t

__all__ = [
    "saliva_feature_wide_to_long",
    "get_saliva_column_suggestions",
    "extract_saliva_columns",
    "sample_times_datetime_to_minute",
]


def saliva_feature_wide_to_long(
    data: SalivaFeatureDataFrame,
    saliva_type: str,
) -> pd.DataFrame:
    """Convert ``SalivaFeatureDataFrame`` from wide-format into long-format.

    Parameters
    ----------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
        dataframe containing saliva features in wide-format, i.e. one column per saliva sample, one row per subject.
    saliva_type : str
        saliva type (e.g. 'cortisol')

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with saliva features in long-format

    """
    data = data.filter(like=saliva_type)
    index_cols = list(data.index.names)
    j = "saliva_feature"
    # iteratively build up long-format dataframe
    data = pd.wide_to_long(
        data.reset_index(),
        stubnames=saliva_type,
        i=index_cols,
        j=j,
        sep="_",
        suffix=r"\w+",
    )

    # reorder levels and sort
    return data.reorder_levels(index_cols + [j]).sort_index()


def get_saliva_column_suggestions(data: pd.DataFrame, saliva_type: Union[str, Sequence[str]]) -> Sequence[str]:
    """Automatically extract possible saliva data columns from a pandas dataframe.

    This is for example useful when one large dataframe is used to store demographic information,
    questionnaire data and saliva data.

    Parameters
    ----------
    data: :class:`~pandas.DataFrame`
        dataframe which should be extracted
    saliva_type: str or list of str
        saliva type variable which or list of saliva types should be used to extract columns (e.g. 'cortisol')

    Returns
    -------
    list or dict
        list of suggested columns containing saliva data or dict of such if ``saliva_type`` is a list

    """
    # check if input is dataframe
    _assert_is_dtype(data, pd.DataFrame)

    if isinstance(saliva_type, list):
        dict_result = {}
        for saliva in saliva_type:
            dict_result[saliva] = get_saliva_column_suggestions(data=data, saliva_type=saliva)
        return dict_result

    if saliva_type not in _dict_saliva_type_suggs:
        raise ValueError(
            "Invalid saliva type '{}'! Must be one of {}.".format(saliva_type, list(_dict_saliva_type_suggs.keys()))
        )

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


def extract_saliva_columns(
    data: pd.DataFrame, saliva_type: Union[str, Sequence[str]], col_pattern: Optional[Union[str, Sequence[str]]] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Extract saliva sample columns from a pandas dataframe.

    Parameters
    ----------
    data: :class:`~pandas.DataFrame`
        dataframe to extract columns from
    saliva_type: str or list of str
        saliva type variable or list of saliva types which should be used to extract columns (e.g. 'cortisol')
    col_pattern: str, optional
        string pattern or list of string patterns to identify saliva columns.
        If ``None``, it is attempted to automatically infer column names using :func:`get_saliva_column_suggestions()`.
        If ``col_pattern`` is a list, it must be the same length like ``saliva_type``.

    Returns
    -------
    :class:`~pandas.DataFrame` or dict
        pandas dataframe with extracted columns or dict of such if ``saliva_type`` is a list

    """
    if isinstance(saliva_type, list):
        if isinstance(col_pattern, list) and len(saliva_type) is not len(col_pattern):
            raise ValueError("'saliva_type' and 'col_pattern' must have same length!")
        dict_result = {}
        if col_pattern is None:
            col_pattern = [None] * len(saliva_type)
        for saliva, col_p in zip(saliva_type, col_pattern):
            dict_result[saliva] = extract_saliva_columns(data=data, saliva_type=saliva, col_pattern=col_p)
        return dict_result

    if col_pattern is None:
        col_suggs = get_saliva_column_suggestions(data, saliva_type)
        if len(col_suggs) > 1:
            raise ValueError(
                "More than one possible column pattern was found! "
                "Please check manually which pattern is correct: {}".format(col_suggs)
            )
        col_pattern = col_suggs[0]
    return data.filter(regex=col_pattern)


def _sample_times_datetime_to_minute_apply(
    sample_times: Union[pd.DataFrame, pd.Series]
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(sample_times.values.flatten()[0], (pd.Timedelta, np.timedelta64)):
        return sample_times.apply(pd.to_timedelta)
    return sample_times.astype(str).apply(pd.to_datetime)


def sample_times_datetime_to_minute(sample_times: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """Convert sample times from datetime or timedelta objects into minutes.

    In order to compute certain saliva features (such as :func:`~biopsykit.saliva.auc` or
    :func:`~biopsykit.saliva.slope`) the saliva sampling times are needed.
    This function can be used to convert sampling times into minutes relative to the first saliva sample.

    Parameters
    ----------
    sample_times : :class:`~pandas.Series` or :class:`~pandas.DataFrame`
        saliva sampling times in a Python datetime- or timedelta-related format.
        If ``sample_times`` is a Series, it is assumed to be in long-format and will be unstacked into wide-format
        along the `sample` level.
        If ``sample_times`` is a DataFrame, it is assumed to be in wide-format already.
        If values in ``sample_times`` are ``str``, they are assumed to be strings with time information only
        (**not** including date), e.g., "09:00", "09:15", ...

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe in wide-format with saliva sampling times in minutes relative to the first saliva sample

    Raises
    ------
    ValueError
        if sample times are not in a datetime- or timedelta-related format

    """
    if isinstance(sample_times.values.flatten()[0], str):
        sample_times = _get_sample_times_str(sample_times)

    if not isinstance(sample_times.values.flatten()[0], (time, datetime, pd.Timedelta, np.timedelta64, np.datetime64)):
        raise ValueError(
            "Sample times must be instance of `datetime.datetime()`, `datetime.time()`,"
            " `np.datetime64`, `np.timedelta64`, or `pd.Timedelta`!"
        )

    is_series = isinstance(sample_times, pd.Series)
    if is_series:
        _assert_has_index_levels(sample_times, index_levels=["sample"], match_atleast=True)
        # unstack the multi-index dataframe in the 'samples' level so that time differences can be computed in minutes.
        # Then stack it back together
        sample_times = sample_times.unstack(level="sample")

    sample_times = _sample_times_datetime_to_minute_apply(sample_times)

    sample_times = sample_times.diff(axis=1).apply(lambda s: (s.dt.total_seconds() / 60))
    sample_times = sample_times.cumsum(axis=1)
    sample_times.iloc[:, 0].fillna(0, inplace=True)
    if is_series:
        sample_times = sample_times.stack()
    return sample_times


def _get_sample_times_str(sample_times: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(sample_times, pd.DataFrame):
        return pd.to_timedelta(sample_times.stack()).unstack("sample")
    return pd.to_timedelta(sample_times)


def _remove_s0(data: SalivaRawDataFrame) -> SalivaRawDataFrame:
    """Remove first saliva sample.

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
    return _SalivaRawDataFrame(data)


def _check_sample_times(sample_times: np.array) -> None:
    """Check that all sample times are monotonously increasing.

    Parameters
    ----------
    sample_times : array-like
        list of sample times

    Raises
    ------
    ValueError
        if values in ``sample_times`` are not monotonously increasing

    """
    if np.any(np.diff(sample_times) <= 0):
        raise ValueError("'sample_times' must be increasing!")


def _get_sample_times(
    data: pd.DataFrame,
    saliva_type: str,
    sample_times: Optional[Union[np.array, Sequence[int]]] = None,
    remove_s0: Optional[bool] = False,
) -> np.array:
    if sample_times is None:
        # check if dataframe has 'time' index
        if "time" in data.index.names:
            data = data.reset_index("time")
        # check if dataframe has 'time' column
        if "time" in data.columns:
            sample_times = np.array(data.unstack(level="sample")["time"])
            if np.all((sample_times == sample_times[0])):
                # all subjects have the same saliva times
                sample_times = sample_times[0]
        else:
            raise ValueError("No sample times specified!")

    # ensure numpy
    sample_times = np.squeeze(sample_times)

    # check whether we have the same saliva times for all subjects (1d array) or not (2d array)
    # and whether the input format is correct
    sample_times = _sample_times_sanitize(data, sample_times, saliva_type)
    _get_sample_times_check_dims(data, sample_times, saliva_type)

    if remove_s0:
        sample_times = sample_times[..., 1:]
    return sample_times


def _sample_times_sanitize(data: pd.DataFrame, sample_times: arr_t, saliva_type: str) -> arr_t:
    if sample_times.ndim == 1:
        exp_shape = data.unstack(level="sample")[saliva_type].shape[1]
        act_shape = sample_times.shape[0]
        if act_shape != exp_shape and (act_shape % exp_shape) == 0:
            # saliva times are in long-format => number of sample times corresponds to 2nd dimension
            sample_times = np.array(sample_times.unstack("sample").squeeze())
    return sample_times


def _get_sample_times_check_dims(data: pd.DataFrame, sample_times: arr_t, saliva_type: str):
    if sample_times.ndim == 1:
        exp_shape = data.unstack(level="sample")[saliva_type].shape[1]
        act_shape = sample_times.shape[0]
        # saliva times equal for all subjects
        # => number of sample times must correspond to 2nd dimension of wide-format data
        if act_shape != exp_shape:
            raise ValueError(
                "'sample_times' does not correspond to the number of saliva samples in 'data'! "
                "Expected {}, got {}.".format(exp_shape, act_shape)
            )
    elif sample_times.ndim == 2:
        # saliva time different for all subjects
        exp_shape = data.unstack(level="sample")[saliva_type].shape
        act_shape = sample_times.shape
        if act_shape != exp_shape:
            raise ValueError(
                "Dimensions of 'sample_times' does not correspond to dimensions of 'data'! "
                "Expected {}, got {}.".format(exp_shape, act_shape)
            )
    else:
        raise ValueError("'sample_times' has invalid dimensions! Expected 1 or 2, got {}".format(sample_times.ndim))


def _get_saliva_idx_labels(
    columns: pd.Index,
    sample_labels: Optional[Union[Tuple, Sequence]] = None,
    sample_idx: Optional[Union[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[Sequence, Sequence]:
    """Get sample labels and indices from data, if only one of both was specified.

    Parameters
    ----------
    columns : :class:`pandas.Index`
        dataframe columns
    sample_labels : list, optional
        list of sample labels
    sample_idx : list, optional
        list of sample indices

    Returns
    -------
    sample_labels:
        list of sample labels
    sample_idx:
        list of sample indices

    """
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

    sample_idx = _get_saliva_idx_labels_sanitize(sample_idx, columns)
    return sample_labels, sample_idx


def _get_saliva_idx_labels_sanitize(sample_idx: List[int], columns: Sequence[str]):
    # replace idx values like '-1' with the actual index
    if sample_idx[0] < 0:
        sample_idx[0] = len(columns) + sample_idx[0]

    if sample_idx[1] < 0:
        sample_idx[1] = len(columns) + sample_idx[1]

    # check that second index is bigger than first index
    if sample_idx[0] >= sample_idx[1]:
        raise IndexError("`sample_idx[1]` must be bigger than `sample_idx[0]`. Got {}".format(sample_idx))
    return sample_idx


def _get_group_cols(
    data: SalivaRawDataFrame, group_cols: Union[str, Sequence[str]], group_type: str, function_name: str
) -> List[str]:
    """Get appropriate columns for grouping.

    Parameters
    ----------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    group_cols : str or list of str
        list of index levels and column names to group by
    group_type : str
        which index level should be grouped by: 'subject' (for computing features per subject, along 'sample' axis)
        or 'sample' (for computing values per sample, along 'subject' axis)
    function_name : str
        function name for error message: 'standard_features' or 'mean_se'

    Returns
    -------
    list of str
        list of group by columns

    """
    if isinstance(group_cols, str):
        # ensure list
        group_cols = [group_cols]
    elif group_cols is None:
        # group by all available index levels
        group_cols = list(data.index.names)
        group_cols.remove(group_type)

    if any(col not in list(data.index.names) + list(data.columns) for col in group_cols):
        # check for valid groupers
        raise ValueError(
            "Computing {} failed: Not all of '{}' are valid index levels or column names!".format(
                function_name, group_cols
            )
        )
    return group_cols


_dict_saliva_type_suggs: Dict[str, Sequence[str]] = {
    "cortisol": ["cortisol", "cort", "Cortisol", "_c_"],
    "amylase": ["amylase", "amy", "Amylase", "sAA"],
    "il6": ["il6", "IL6", "il-6", "IL-6", "il_6", "IL_6"],
}
"""Dictionary containing possible column patterns for different saliva types."""
