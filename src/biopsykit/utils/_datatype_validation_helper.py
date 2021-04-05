"""Internal helpers for dataset validation."""
from pathlib import Path
from typing import Union, Tuple, Sequence, List, Iterable, Optional

import pandas as pd

from biopsykit.utils._types import _Hashable, path_t
from biopsykit.utils.exceptions import ValidationError, FileExtensionError


def _assert_file_extension(
    file_name: path_t, expected_extension: Union[str, Sequence[str]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    # ensure pathlib
    file_name = Path(file_name)
    if isinstance(expected_extension, str):
        expected_extension = [expected_extension]
    if file_name.suffix not in expected_extension:
        if raise_exception:
            raise FileExtensionError(
                "The file name extension is expected to be one of {}. "
                "Instead it has the following extension: {}".format(expected_extension, file_name.suffix)
            )
        return False
    return True


def _assert_is_dtype(
    obj, dtype: Union[type, Tuple[type, ...]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if an object has a specific dtype.

    Parameters
    ----------
    obj
        object to check
    dtype
        data type of tuple of data types to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``obj`` is one of the expected data types, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``obj`` is none of the expected data types

    """
    if not isinstance(obj, dtype):
        if raise_exception:
            raise ValidationError(
                "The data object is expected to be one of ({},). But it is a {}".format(dtype, type(obj))
            )
        return False
    return True


def _assert_has_column_multiindex(
    df: pd.DataFrame,
    expected: Optional[bool] = True,
    nlevels: Optional[int] = 2,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if a :any:`pandas.DataFrame` has a :any:`pandas.MultiIndex` as columns.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The dataframe to check
    expected : bool, optional
        Whether the df is expected to have MultiIndex column or not
    nlevels : int, optional
        If MultiIndex is expected, how many levels the MultiIndex columns should have
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` meets the expected column index format, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception` is ``True`` and ``df`` does not meet the expected column index format

    """
    return _multiindex_check_helper(
        df=df, idx_or_col="column", expected=expected, nlevels=nlevels, raise_exception=raise_exception
    )


def _assert_has_multiindex(
    df: pd.DataFrame,
    expected: Optional[bool] = True,
    nlevels: Optional[int] = 2,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    """Check if a :any:`pandas.DataFrame` has a :any:`pandas.MultiIndex` as index.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The dataframe to check
    expected : bool, optional
        Whether the df is expected to have a MultiIndex index or not
    nlevels : int, optional
        If MultiIndex is expected, how many levels the MultiIndex index should have
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` meets the expected index format, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``df`` does not meet the expected index format

    """
    return _multiindex_check_helper(
        df=df, idx_or_col="index", expected=expected, nlevels=nlevels, raise_exception=raise_exception
    )


def _assert_has_index_levels(
    df: pd.DataFrame, index_levels: Iterable[_Hashable], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if the dataframe has all index level names.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The dataframe to check
    index_levels : list
        Set of index level names to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected index level names, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``df`` does not have the expected index level names

    """
    return _multiindex_level_names_helper(df, index_levels, "index", raise_exception)


def _assert_has_columns(
    df: pd.DataFrame, columns_sets: Sequence[Union[List[_Hashable], List[str]]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if the dataframe has at least all columns sets.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The dataframe to check
    columns_sets : list
        Column set of list of column sets to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected column names, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
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
    for col_set in columns_sets:
        result = result or all(v in columns for v in col_set)

    if result is False:
        if len(columns_sets) == 1:
            helper_str = "the following columns: {}".format(columns_sets[0])
        else:
            helper_str = "one of the following sets of columns: {}".format(columns_sets)
        if raise_exception:
            raise ValidationError(
                "The dataframe is expected to have {}. Instead it has the following columns: {}".format(
                    helper_str, list(df.columns)
                )
            )
    return result


def _assert_has_columns_levels(
    df: pd.DataFrame, column_levels: Iterable[_Hashable], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check if the dataframe has all column level names of a MultiIndex column.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The dataframe to check
    column_levels : list
        Set of column level names to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``df`` has the expected column level names, ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``df`` does not have the expected index level names

    """
    return _multiindex_level_names_helper(df, column_levels, "column", raise_exception)


def _multiindex_level_names_helper(
    df: pd.DataFrame, level_names: Iterable[_Hashable], idx_or_col: str, raise_exception: Optional[bool] = True
) -> Optional[bool]:
    ex_levels = list(level_names)
    if idx_or_col == "index":
        ac_levels = list(df.index.names)
    else:
        ac_levels = list(df.columns.names)
    if ex_levels != ac_levels:
        if raise_exception:
            raise ValidationError(
                "The dataframe is expected to have exactly the following {} level names ({}), "
                "but it has {}".format(idx_or_col, level_names, ac_levels)
            )
        return False
    return True


def _multiindex_check_helper(
    df: pd.DataFrame,
    idx_or_col: str,
    expected: Optional[bool] = True,
    nlevels: Optional[int] = 2,
    raise_exception: Optional[bool] = True,
) -> Optional[bool]:
    if idx_or_col == "index":
        has_multiindex = isinstance(df.index, pd.MultiIndex)
        nlevels_act = df.index.nlevels
    else:
        has_multiindex = isinstance(df.columns, pd.MultiIndex)
        nlevels_act = df.columns.nlevels

    if has_multiindex is not expected:
        if expected is False:
            if raise_exception:
                raise ValidationError(
                    "The dataframe is expected to have a single level as {0}. "
                    "But it has a MultiIndex with {1} {0} levels.".format(idx_or_col, nlevels_act)
                )
            return False
        if raise_exception:
            raise ValidationError(
                "The dataframe is expected to have a MultiIndex with {0} {1} levels. "
                "It has just a single normal {1} level.".format(nlevels, idx_or_col)
            )
        return False
    if has_multiindex is True:
        if not nlevels_act == nlevels:
            if raise_exception:
                raise ValidationError(
                    "The dataframe is expected to have a MultiIndex with {0} {1} levels. "
                    "But it has a MultiIndex with {2} {1} levels.".format(nlevels, idx_or_col, nlevels_act)
                )
            return False
    return True
