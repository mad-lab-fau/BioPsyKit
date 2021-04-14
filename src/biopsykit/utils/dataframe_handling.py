from typing import Optional, Callable, Union, Sequence

import numpy as np
import pandas as pd
from biopsykit.utils._types import path_t


def int_from_str_idx(
    data: pd.DataFrame,
    idx_names: Union[str, Sequence[str]],
    regex: Union[str, Sequence[str]],
    func: Optional[Callable] = None,
) -> pd.DataFrame:
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


def int_from_str_col(
    data: pd.DataFrame, col_name: str, regex: str, func: Optional[Callable] = None
) -> Union[pd.Series]:
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

    # TODO don't convert if all in capital letters
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


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
    data: Union[pd.DataFrame, pd.Series], inplace: Optional[bool] = False
) -> Union[pd.DataFrame, pd.Series, None]:
    if inplace:
        data.replace([-99.0, -77.0, -66.0, "-99", "-77", "-66"], np.nan, inplace=True)
    else:
        return data.replace([-99.0, -77.0, -66.0, "-99", "-77", "-66"], np.nan, inplace=False)


def multi_xs(data: pd.DataFrame, keys: Union[str, Sequence[str]], level: str) -> pd.DataFrame:
    if isinstance(keys, str):
        keys = [keys]
    levels = data.index.names
    data_xs = pd.concat({key: data.xs(key, level=level) for key in keys}, names=[level])
    return data_xs.reorder_levels(levels).sort_index()


def stack_groups_percent(
    data: pd.DataFrame, hue: str, stacked: str, order: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    data_grouped = pd.DataFrame(data.groupby([hue] + [stacked]).size(), columns=["data"])
    data_grouped = data_grouped.groupby(hue).apply(lambda x: 100 * (x / x.sum())).T.stack().T
    if order:
        data_grouped = data_grouped.reindex(order)
    return data_grouped["data"]


def apply_codebook(path_or_df: Union[path_t, pd.DataFrame], data: pd.DataFrame) -> pd.DataFrame:
    from biopsykit.io import load_questionnaire_data

    if isinstance(path_or_df, pd.DataFrame):
        codebook = path_or_df
    else:
        # ensure pathlib
        codebook = load_questionnaire_data(path_or_df, index_cols="variable")

    for col in data.index.names:
        if col in codebook.index:
            data.rename(index=codebook.loc[col], level=col, inplace=True)

    return data


def wide_to_long(
    data: pd.DataFrame,
    stubname: str,
    levels: Union[str, Sequence[str]],
    sep: Optional[str] = "_",
) -> pd.DataFrame:
    """Convert a dataframe wide-format into long-format.

    In the wide-format dataframe, the index levels to be converted into long-format are expected to be encoded in the
    column names and separated by ``sep``. If multiple levels should be converted into long-format, e.g., for a
    questionnaire with subscales (level `subscale`) that was assessed pre and post (level `time`), then the different
    levels are all encoded into the string. The level order is specified by ``levels``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        pandas DataFrame containing saliva data in wide-format, i.e. one column per saliva sample, one row per subject
    stubname : str
        common name for each column to be converted into long-format. Usually, this is either the name of the
        questionnaire (e.g., "PSS") or the saliva type (e.g., "cortisol")
    levels : str or list of str
        index levels of the resulting long-format dataframe.
    sep : str, optional
        character separating index levels in the column names of the wide-format dataframe

    Returns
    -------
    :class:`pandas.DataFrame`
        pandas DataFrame in long-format


    Examples
    --------
    >>> import pandas as pd
    >>> from biopsykit.utils.dataframe_handling import wide_to_long
    >>> data = pd.DataFrame(
    >>>     columns=[
    >>>         "MDBF_GoodBad_pre", "MDBF_AwakeTired_pre", "MDBF_CalmNervous_pre",
    >>>         "MDBF_GoodBad_post", "MDBF_AwakeTired_post",  "MDBF_CalmNervous_post"
    >>>     ],
    >>>     index=pd.Index(range(0, 5), name="subject")
    >>> )
    >>> data_long = wide_to_long(data, stubname="MDBF", levels=["subscale", "time"], sep="_")
    >>> print(data_long.index.names)
    ['subject', 'subscale', 'time']
    >>> print(data_long.index)
    MultiIndex([(0,  'AwakeTired', 'post'),
            (0,  'AwakeTired',  'pre'),
            (0, 'CalmNervous', 'post'),
            (0, 'CalmNervous',  'pre'),
            (0,     'GoodBad', 'post'),
            (0,     'GoodBad',  'pre'),
            (1,  'AwakeTired', 'post'),
            ...



    """
    if isinstance(levels, str):
        levels = [levels]

    data = data.filter(like=stubname)
    index_cols = list(data.index.names)
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
            i=index_cols + levels[0:i],
            j=level,
            sep=sep,
            suffix=r"\w+",
        )

    # reorder levels and sort
    return data.reorder_levels(index_cols + levels[::-1]).sort_index()
