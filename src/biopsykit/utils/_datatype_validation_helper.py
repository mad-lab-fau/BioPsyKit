"""Internal helpers for dataset validation."""
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from biopsykit.utils._types_internal import _Hashable, path_t
from biopsykit.utils.exceptions import FileExtensionError, ValidationError, ValueRangeError


def _assert_is_dir(path: path_t, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check if a path is a directory.

    Parameters
    ----------
    path : path or str
        path to check if it's a directory
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``path`` is a directory, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValueError
        if ``raise_exception`` is ``True`` and ``path`` is not a directory

    """
    # ensure pathlib
    file_name = Path(path)
    if not file_name.is_dir():
        if raise_exception:
            raise ValueError(f"The path '{path}' is expected to be a directory, but it's not!")
        return False

    return True


def _assert_file_extension(
    file_name: path_t, expected_extension: Union[str, Sequence[str]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if a file has the correct file extension.

    Parameters
    ----------
    file_name : path or str
        file name to check for correct extension
    expected_extension : str or list of str
        file extension (or a list of file extensions) to check for
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``file_name`` ends with one of the specified file extensions, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.FileExtensionError`
        if ``raise_exception`` is ``True`` and ``file_name`` does not end with any of the specified
        ``expected_extension``

    """
    # ensure pathlib
    file_name = Path(file_name)
    if isinstance(expected_extension, str):
        expected_extension = [expected_extension]
    if file_name.suffix not in expected_extension:
        if raise_exception:
            raise FileExtensionError(
                f"The file name extension is expected to be one of {expected_extension}. "
                f"Instead it has the following extension: {file_name.suffix}"
            )
        return False
    return True


def _assert_is_dtype(
    obj, dtype: Union[type, tuple[type, ...]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if an object has a specific data type.

    Parameters
    ----------
    obj : any object
        object to check
    dtype : type or list of type
        data type of tuple of data types to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``obj`` is one of the expected data types, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``obj`` is none of the expected data types

    """
    if not isinstance(obj, dtype):
        if raise_exception:
            raise ValidationError(f"The data object is expected to be one of ({dtype},). But it is a {type(obj)}")
        return False
    return True


def _assert_has_multiindex(
    df: pd.DataFrame,
    expected: Optional[bool] = True,
    nlevels: Optional[int] = 2,
    nlevels_atleast: Optional[int] = False,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if a :any:`pandas.DataFrame` has a :any:`pandas.MultiIndex` as index.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        The dataframe to check
    expected : bool, optional
        Whether the df is expected to have a MultiIndex index or not
    nlevels : int, optional
        If MultiIndex is expected, how many levels the MultiIndex index should have
    nlevels_atleast : bool, optional
        Whether the MultiIndex has to have at least ``nlevels`` (``True``)
        or exactly match the number of levels (``False``)
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` meets the expected index format, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``df`` does not meet the expected index format

    """
    return _multiindex_check_helper(
        df=df,
        idx_or_col="index",
        expected=expected,
        nlevels=nlevels,
        nlevels_atleast=nlevels_atleast,
        raise_exception=raise_exception,
    )


def _assert_has_index_levels(
    df: pd.DataFrame,
    index_levels: Iterable[_Hashable],
    match_atleast: Optional[bool] = False,
    match_order: Optional[bool] = False,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if the dataframe has all index level names.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        The dataframe to check
    index_levels : list
        Set of index level names to check
    match_atleast : bool, optional
        Whether the MultiIndex columns have to have at least the specified column levels (``True``)
        or exactly match the column levels (``False``)
    match_order : bool, optional
        Whether to also match the level order
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected index level names, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``df`` does not have the expected index level names

    """
    return _multiindex_level_names_helper(
        df,
        level_names=index_levels,
        idx_or_col="index",
        match_atleast=match_atleast,
        match_order=match_order,
        raise_exception=raise_exception,
    )


def _assert_has_columns(
    df: pd.DataFrame,
    column_sets: Sequence[Union[list[_Hashable], list[str], pd.Index]],
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if the dataframe has at least all columns sets.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        The dataframe to check
    column_sets : list
        Column set or list of column sets to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected column names, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``df`` does not have the expected index level names

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> df.columns = ["col1", "col2"]
    >>> _assert_has_columns(df, [["other_col1", "other_col2"], ["col1", "col2"]])
    >>> # This raises no error, as df contains all columns of the second set

    """
    columns = df.columns
    result = False
    for col_set in column_sets:
        result = result or all(v in columns for v in col_set)

    if result is False:
        if len(column_sets) == 1:
            helper_str = f"the following columns: {column_sets[0]}"
        else:
            helper_str = f"one of the following sets of columns: {column_sets}"
        if raise_exception:
            raise ValidationError(
                f"The dataframe is expected to have {helper_str}. "
                f"Instead it has the following columns: {list(df.columns)}"
            )
    return result


def _assert_has_column_multiindex(
    df: pd.DataFrame,
    expected: Optional[bool] = True,
    nlevels: Optional[int] = 2,
    nlevels_atleast: Optional[int] = False,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if a :any:`pandas.DataFrame` has a :any:`pandas.MultiIndex` as columns.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        The dataframe to check
    expected : bool, optional
        Whether the df is expected to have MultiIndex column or not
    nlevels : int, optional
        If MultiIndex is expected, how many levels the MultiIndex columns should have
    nlevels_atleast : bool, optional
        Whether the MultiIndex has to have at least ``nlevels`` (``True``)
        or exactly match the number of levels (``False``)
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` meets the expected column index format, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception` is ``True`` and ``df`` does not meet the expected column index format

    """
    return _multiindex_check_helper(
        df=df,
        idx_or_col="column",
        expected=expected,
        nlevels=nlevels,
        nlevels_atleast=nlevels_atleast,
        raise_exception=raise_exception,
    )


def _assert_has_columns_any_level(
    df: pd.DataFrame,
    columns_sets: Sequence[Union[list[_Hashable], list[str], pd.Index]],
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if the dataframe has the expected set of column names at any level of a :any:`pandas.MultiIndex`.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        The dataframe to check
    columns_sets : list
        Column set of list of column sets to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected column names at any :any:`pandas.MultiIndex` level,
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``df`` does not have the expected column names

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> df.columns = pd.MultiIndex.from_tuples([("Name", "col1"), ("Name", "col2")])
    >>> _assert_has_columns_any_level(df, [["col1", "col2"]])
    >>> # This raises no error, as df contains all columns in the seconds level

    """
    _assert_has_column_multiindex(df, expected=True, nlevels_atleast=True)
    column_levels = [np.array(df.columns.get_level_values(i)) for i in range(df.columns.nlevels)]
    result = False
    for columns in column_levels:
        for col_set in columns_sets:
            result = result or all(v in columns for v in col_set)

    if result is False:
        if len(columns_sets) == 1:
            helper_str = f"the following columns: {columns_sets[0]}"
        else:
            helper_str = f"one of the following sets of columns: {columns_sets}"
        if raise_exception:
            raise ValidationError(
                f"The dataframe is expected to have {helper_str} at any level of the MultiIndex. Instead it has the "
                f"following MultiIndex columns: {column_levels}"
            )
    return result


def _assert_has_column_levels(
    df: pd.DataFrame,
    column_levels: Iterable[_Hashable],
    match_atleast: Optional[bool] = False,
    match_order: Optional[bool] = False,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if the dataframe has all column level names of a MultiIndex column.

    Parameters
    ----------
    df : :class:`~pandas.DataFrame`
        The dataframe to check
    column_levels : list
        Set of column level names to check
    match_atleast : bool, optional
        Whether the MultiIndex columns have to have at least the specified column levels (``True``)
        or exactly match the column levels (``False``)
    match_order : bool, optional
        Whether to also match the level order
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected column level names, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``df`` does not have the expected index level names

    """
    return _multiindex_level_names_helper(
        df,
        level_names=column_levels,
        idx_or_col="column",
        match_atleast=match_atleast,
        match_order=match_order,
        raise_exception=raise_exception,
    )


def _assert_value_range(
    data: Union[pd.DataFrame, pd.Series],
    value_range: Sequence[Union[int, float]],
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if all values are within the specified range.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check values
    value_range : tuple of numbers
        value range in the format [min_val, max_val]
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if all values in ``data`` are within ``value_range``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValueRangeError`
        if ``raise_exception`` is ``True`` and any value of ``data`` is not within ``value_range``

    """
    max_val = np.nanmax(data)
    min_val = np.nanmin(data)
    if not (min_val >= value_range[0] and max_val <= value_range[1]):
        if raise_exception:
            raise ValueRangeError(
                "Some of the values are out of the expected range. "
                f"Expected were values in the range {value_range}, got values in the range {[min_val, max_val]}. "
                "If values are part of questionnaire scores, "
                "you can convert questionnaire items into the correct range by calling "
                "`biopsykit.questionnaire.utils.convert_scale()`."
            )
        return False
    return True


def _assert_num_columns(
    data: pd.DataFrame, num_cols: Union[int, Sequence[int]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if dataframe has (any of) the required number of columns.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check
    num_cols : int or list of int
        the required number of columns (or any of the required number of columns in case ``num_cols`` is a list)
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` has the required number of columns, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``data`` does not have the required number of columns

    """
    if isinstance(num_cols, int):
        num_cols = [num_cols]

    if not any(len(data.columns) == num for num in num_cols):
        if raise_exception:
            raise ValidationError(
                "The dataframe does not have the required number of columns. "
                f"Expected were any of {num_cols} columns, but has {len(data.columns)} columns."
            )
        return False
    return True


def _assert_len_list(data: Sequence, length: int, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check if a list has the required length.

    Parameters
    ----------
    data : list
        list to check
    length : int
        the required length or the list
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` has the required length, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``data`` does not have the required length

    """
    _assert_is_dtype(data, (list, tuple, np.ndarray))
    if len(data) != length:
        if raise_exception:
            raise ValidationError(
                "The list does not have the required length. "
                f"Expected was length {length}, but it has length {len(data)}."
            )
        return False
    return True


def _assert_dataframes_same_length(
    df_list: Sequence[pd.DataFrame], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if all dataframes have same length.

    Parameters
    ----------
    df_list : list
        list of dataframes to check
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if all dataframes in ``df_list`` have same length, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    :exc:`~biopsykit.exceptions.ValidationError`
        if ``raise_exception`` is ``True`` and ``data`` does not have the required length

    """
    if len({len(df) for df in df_list}) != 1:
        if raise_exception:
            raise ValidationError("Not all dataframes have the same length!")
        return False
    return True


def _assert_sample_columns_int(data: pd.DataFrame) -> None:
    """Assert that the columns of a DataFrame that have "_sample" in their name are of type int."""
    if not any(data.columns.str.contains("_sample")):
        raise ValidationError("DataFrame does not contain any columns with '_sample' in their name!")
    for col in data.columns:
        if "_sample" in col and not pd.api.types.is_integer_dtype(data[col]):
            raise ValidationError(f"Column '{col}' is not of type 'int'!")


def _multiindex_level_names_helper_get_expected_levels(
    ac_levels: Sequence[str],
    ex_levels: Sequence[str],
    match_atleast: Optional[bool] = False,
    match_order: Optional[bool] = False,
) -> bool:
    if match_order:
        if match_atleast:
            ac_levels_slice = ac_levels[: len(ex_levels)]
            expected = ex_levels == ac_levels_slice
        else:
            expected = ex_levels == ac_levels
    elif match_atleast:
        expected = all(level in ac_levels for level in ex_levels)
    else:
        expected = sorted(ex_levels) == sorted(ac_levels)

    return expected


def _multiindex_level_names_helper(
    df: pd.DataFrame,
    level_names: Iterable[_Hashable],
    idx_or_col: str,
    match_atleast: Optional[bool] = False,
    match_order: Optional[bool] = False,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:

    if isinstance(level_names, str):
        level_names = [level_names]

    ex_levels = list(level_names)
    ac_levels = list(df.index.names) if idx_or_col == "index" else list(df.columns.names)

    expected = _multiindex_level_names_helper_get_expected_levels(ac_levels, ex_levels, match_atleast, match_order)

    if not expected:
        if raise_exception:
            raise ValidationError(
                f"The dataframe is expected to have exactly the following {idx_or_col} level names {level_names}, "
                f"but it has {ac_levels}"
            )
        return False
    return True


def _multiindex_check_helper(
    df: pd.DataFrame,
    idx_or_col: str,
    expected: Optional[bool] = True,
    nlevels: Optional[int] = 2,
    nlevels_atleast: Optional[int] = False,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:

    has_multiindex, nlevels_act = _multiindex_check_helper_get_levels(df, idx_or_col)

    if has_multiindex is not expected:
        return _multiindex_check_helper_not_expected(idx_or_col, nlevels, nlevels_act, expected, raise_exception)

    if has_multiindex is True:
        expected = nlevels_act >= nlevels if nlevels_atleast else nlevels_act == nlevels
        if not expected:
            if raise_exception:
                raise ValidationError(
                    f"The dataframe is expected to have a MultiIndex with {nlevels} {idx_or_col} levels. "
                    f"But it has a MultiIndex with {nlevels_act} {idx_or_col} levels."
                )
            return False
    return True


def _multiindex_check_helper_get_levels(df: pd.DataFrame, idx_or_col: str) -> tuple[bool, int]:
    if idx_or_col == "index":
        has_multiindex = isinstance(df.index, pd.MultiIndex)
        nlevels_act = df.index.nlevels
    else:
        has_multiindex = isinstance(df.columns, pd.MultiIndex)
        nlevels_act = df.columns.nlevels

    return has_multiindex, nlevels_act


def _multiindex_check_helper_not_expected(
    idx_or_col: str, nlevels: int, nlevels_act: int, expected: bool, raise_exception: bool
) -> Optional[bool]:
    if not expected:
        if raise_exception:
            raise ValidationError(
                f"The dataframe is expected to have a single level as {idx_or_col}. "
                f"But it has a MultiIndex with {nlevels_act} {idx_or_col} levels."
            )
        return False
    if raise_exception:
        raise ValidationError(
            f"The dataframe is expected to have a MultiIndex with {nlevels} {idx_or_col} levels. "
            f"It has just a single normal {idx_or_col} level."
        )
    return False


def _assert_has_column_prefix(
    columns: Sequence[str], prefix: str, raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether all columns start with the same prefix.

    Parameters
    ----------
    columns : list of str
        list of column names
    prefix : str
        expected prefix of all columns
    raise_exception : bool, optional
        Whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``columns`` all start with ``prefix``, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and one of ``columns`` is not a string or does not start with ``prefix``

    """
    if prefix is None or len(prefix) == 0:
        if raise_exception:
            raise ValidationError("'prefix' is None or empty!")
        return False

    for col in columns:
        return _check_has_column_prefix_single_col(columns, col, prefix, raise_exception)

    return True


def _check_has_column_prefix_single_col(
    columns: Sequence[str], col: Any, prefix: str, raise_exception: bool
) -> Optional[bool]:
    if not _assert_is_dtype(col, str, raise_exception=False):
        if raise_exception:
            raise ValidationError(f"Column '{col}' from {columns} is not a string!")
        return False
    if not col.startswith(prefix):
        if raise_exception:
            raise ValidationError(f"Column '{col}' from {columns} are starting with the required prefix '{prefix}'!")
        return False
    return True
