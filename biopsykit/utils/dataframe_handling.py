from typing import Optional, Callable, Union, Sequence

import numpy as np
import pandas as pd


def int_from_str_idx(data: pd.DataFrame, idx_names: Union[str, Sequence[str]], regex: Union[str, Sequence[str]],
                     func: Optional[Callable] = None) -> pd.DataFrame:
    """
    Extracts an integer from a index level containing string values.

    Parameters
    ----------
    data
    idx_names
    regex
    func : function to apply to the extracted integers, such as a lambda function to increment all integers by 1

    Returns
    -------
    Dataframe with new index
    """

    if type(idx_names) is not type(regex):
        raise ValueError("`idx_names` and `regex` must both be either strings or list of strings!")

    if isinstance(idx_names, str):
        idx_names = [idx_names]

    if isinstance(regex, str):
        regex = [regex]

    if all([idx not in data.index.names for idx in idx_names]):
        raise ValueError("Not all of `{}` in index!".format(idx_names))

    idx_names_old = data.index.names
    data = data.reset_index()
    for idx, reg in zip(idx_names, regex):
        idx_col = data[idx].str.extract(reg).astype(int)[0]
        if func is not None:
            idx_col = func(idx_col)
        data[idx] = idx_col
    data = data.set_index(idx_names_old)
    return data


def int_from_str_col(data: pd.DataFrame, col_name: str, regex: str, func: Optional[Callable] = None) -> Union[
    pd.Series]:
    """
    Extracts an integer from a column containing string values.

    Parameters
    ----------
    data
    col_name
    regex
    func : function to apply to the extracted integers, such as a lambda function to increment all integers by 1

    Returns
    -------
    Series with integers extracted from the specified column
    """

    if col_name not in data.columns:
        raise ValueError("Name `{}` not in columns!".format(col_name))

    column = data[col_name].str.extract(regex).astype(int)[0]
    if func is not None:
        column = func(column)
    return column


def camel_to_snake(name: str):
    """
    Converts string in 'camelCase' to 'snake_case'.

    Parameters
    ----------
    name

    Returns
    -------

    """
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def replace_missing_data(data: pd.DataFrame, target_col: str, source_col: str, dropna: Optional[bool] = False):
    """
    Replaces missing data in one column by data from another column.

    Parameters
    ----------
    data
    target_col
    source_col
    dropna

    Returns
    -------

    """
    data[target_col] = data[target_col].fillna(data[source_col])
    if dropna:
        return data.dropna(subset=[target_col])
    else:
        return data


def convert_nan(
        data: Union[pd.DataFrame, pd.Series],
        inplace: Optional[bool] = False
) -> Union[pd.DataFrame, pd.Series, None]:
    if inplace:
        data.replace([-99.0, -77.0, -66.0, "-99", "-77", "-66"], np.nan, inplace=True)
    else:
        return data.replace([-99.0, -77.0, -66.0, "-99", "-77", "-66"], np.nan, inplace=False)


def multi_xs(data: pd.DataFrame, keys: Sequence[str], level: str) -> pd.DataFrame:
    levels = data.index.names
    data_xs = pd.concat({key: data.xs(key, level=level) for key in keys}, names=[level])
    return data_xs.reorder_levels(levels).sort_index()


def stack_groups_percent(data: pd.DataFrame, hue: str, stacked: str,
                         order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    data_grouped = pd.DataFrame(data.groupby([hue] + [stacked]).size(), columns=["data"])
    data_grouped = data_grouped.groupby(hue).apply(lambda x: 100 * (x / x.sum())).T.stack().T
    if order:
        data_grouped = data_grouped.reindex(order)
    return data_grouped["data"]
