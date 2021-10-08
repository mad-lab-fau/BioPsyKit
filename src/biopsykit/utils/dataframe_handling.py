"""Module providing various functions for advanced handling of pandas dataframes."""
import re
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_has_index_levels, _assert_is_dtype
from biopsykit.utils.datatype_helper import CodebookDataFrame, is_codebook_dataframe


def int_from_str_idx(
    data: pd.DataFrame,
    idx_levels: Union[str, Sequence[str]],
    regex: Union[str, Sequence[str]],
    func: Optional[Callable] = None,
) -> pd.DataFrame:
    """Extract integers from strings in index levels and set them as new index values.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data with index to extract information from
    idx_levels : str or list of str
        name of index level or list of index level names
    regex : str or list of str
        regex string or list of regex strings to extract integers from strings
    func : function, optional
        function to apply to the extracted integer values. This can, for example, be a lambda function which
        increments all integers by 1. Default: ``None``


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with new index values

    """
    if isinstance(idx_levels, str):
        idx_levels = [idx_levels]
    if isinstance(regex, str):
        regex = [regex]

    if len(idx_levels) != len(regex):
        raise ValueError(
            "Number of values in 'regex' must match number of index levels in 'idx_levels'! "
            "Got idx_levels: {}, regex: {}.".format(idx_levels, regex)
        )

    _assert_is_dtype(data, pd.DataFrame)
    _assert_has_index_levels(data, idx_levels, match_atleast=True, match_order=False)

    idx_names = data.index.names
    data = data.reset_index()
    for idx, reg in zip(idx_levels, regex):
        idx_col = data[idx].str.extract(reg).astype(int)[0]
        if func is not None:
            idx_col = func(idx_col)
        data[idx] = idx_col

    data = data.set_index(idx_names)
    return data


def int_from_str_col(
    data: pd.DataFrame, col_name: str, regex: str, func: Optional[Callable] = None
) -> Union[pd.Series]:
    """Extract integers from strings in the column of a dataframe and return it.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data with column names to extract information from
    col_name : str
        name of column with string values to extract
    regex : str
        regex string used to extract integers from string values
    func : function, optional
        function to apply to the extracted integer values. This can, for example, be a lambda function which
        increments all integers by 1. Default: ``None``


    Returns
    -------
    :class:`~pandas.Series`
        series object with extracted integer values

    """
    _assert_is_dtype(data, pd.DataFrame)
    _assert_has_columns(data, [[col_name]])

    column = data[col_name].str.extract(regex).astype(int)[0]
    if func is not None:
        column = func(column)
    return column


def camel_to_snake(name: str, lower: Optional[bool] = True):
    """Convert string in "camelCase" to "snake_case".

    .. note::
        If all letters in ``name`` are capital letters the string will not be computed into snake_case because
        it is assumed to be an abbreviation.

    Parameters
    ----------
    name : str
        string to convert from camelCase to snake_case
    lower : bool, optional
        ``True`` to convert all capital letters in to lower case ("actual" snake_case), ``False`` to keep
        capital letters, if present


    Returns
    -------
    str
        string converted into snake_case

    Examples
    --------
    >>> from biopsykit.utils.dataframe_handling import camel_to_snake
    >>> camel_to_snake("HelloWorld")
    hello_world
    >>> camel_to_snake("HelloWorld", lower=False)
    Hello_World
    >>> camel_to_snake("ABC")
    ABC

    """
    if not name.isupper():
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
        if lower:
            name = name.lower()
    return name


def replace_missing_data(
    data: pd.DataFrame,
    target_col: str,
    source_col: str,
    dropna: Optional[bool] = False,
    inplace: Optional[bool] = False,
) -> Optional[pd.DataFrame]:
    """Replace missing data in one column by data from another column.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data with values to replace
    target_col : str
        target column, i.e., column in which missing values should be replaced
    source_col : str
        source column, i.e., column values used to replace missing values in ``target_col``
    dropna : bool, optional
        whether to drop rows with missing values in ``target_col`` or not. Default: ``False``
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame` or ``None``
        dataframe with replaced missing values or ``None`` if ``inplace`` is ``True``

    """
    _assert_is_dtype(data, pd.DataFrame)
    if not inplace:
        data = data.copy()

    data[target_col].fillna(data[source_col], inplace=True)
    if dropna:
        data.dropna(subset=[target_col], inplace=True)

    if inplace:
        return None
    return data


def convert_nan(
    data: Union[pd.DataFrame, pd.Series], inplace: Optional[bool] = False
) -> Union[pd.DataFrame, pd.Series, None]:
    """Convert missing values to NaN.

    Data exported from programs like SPSS often uses negative integers to encode missing values because these negative
    numbers are "unrealistic" values. Use this function to convert these negative numbers to
    "actual" missing values: not-a-number (``NaN``).

    Values that will be replaced with ``NaN`` are -66, -77, -99 (integer and string representations).

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        input data
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame` or ``None``
        dataframe with converted missing values or ``None`` if ``inplace`` is ``True``

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))

    if not inplace:
        data = data.copy()
    data.replace([-99.0, -77.0, -66.0, "-99", "-77", "-66"], np.nan, inplace=True)
    if inplace:
        return None
    return data


def multi_xs(
    data: Union[pd.DataFrame, pd.Series],
    keys: Union[str, Sequence[str]],
    level: Union[str, int, Sequence[str], Sequence[int]],
    drop_level: Optional[bool] = True,
) -> Union[pd.DataFrame, pd.Series]:
    """Return cross-section of multiple keys from the dataframe.

    This function internally calls the :meth:`pandas.DataFrame.xs` method, but it can take a list of key arguments
    to return multiple keys at once, in comparison to the original :meth:`~pandas.DataFrame.xs` method which
    only takes one possible key.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        input data to get cross-section from
    keys : str or list of str
        label(s) contained in the index, or partially in a :class:`~pandas.MultiIndex`
    level : str, int, or list of such
        in case of keys partially contained in a :class:`~pandas.MultiIndex`, indicate which index levels are used.
        Levels can be referred by label or position.
    drop_level : bool, optional
        if ``False``, returns object with same levels as self. Default: ``True``


    Returns
    -------
    :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        cross-section from the original dataframe or series

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))
    if isinstance(keys, str):
        keys = [keys]
    levels = data.index.names
    data_xs = pd.concat({key: data.xs(key, level=level, drop_level=drop_level) for key in keys}, names=[level])
    return data_xs.reorder_levels(levels).sort_index()


def stack_groups_percent(
    data: pd.DataFrame, hue: str, stacked: str, order: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Create dataframe with stacked groups.

    To create a stacked bar chart, i.e. a plot with different bar charts along a categorical axis,
    where the variables of each bar chart are stacked along the value axis, the data needs to be rearranged and
    normalized in percent.

    The columns of the resulting dataframe be the categorical values specified by ``hue``,
    the index items will be the variables specified by ``stacked``.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to compute stacked group in percent
    hue : str
        column name of grouping categorical variable. This typically corresponds to the ``x`` axis
        in a stacked bar chart.
    stacked : str
        column name of variable that is stacked along the ``y`` axis
    order : str
        order of categorical variable specified by ``hue``


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe in a format that can be used to create a stacked bar chart

    See Also
    --------
    :func:`~biopsykit.plotting.stacked_barchart`
        function to create a stacked bar chart

    """
    data_grouped = pd.DataFrame(data.groupby([hue] + [stacked]).size(), columns=["data"])
    data_grouped = data_grouped.groupby(hue).apply(lambda x: 100 * (x / x.sum())).T.stack().T
    if order is not None:
        data_grouped = data_grouped.reindex(order)
    return data_grouped["data"]


def apply_codebook(data: pd.DataFrame, codebook: CodebookDataFrame) -> pd.DataFrame:
    """Apply codebook to convert numerical to categorical values.

    The codebook is expected to be a dataframe in a standardized format
    (see :obj:`~biopsykit.utils.datatype_helper.CodebookDataFrame` for further information).



    Parameters
    ----------
    codebook : :obj:`~biopsykit.utils.datatype_helper.CodebookDataFrame`
        path to codebook or dataframe to be used as codebook
    data : :class:`~pandas.DataFrame`
        data to apply codebook on

    Returns
    -------
    :class:`~pandas.DataFrame`
        data with numerical values converted to categorical values

    See Also
    --------
    :func:`~biopsykit.io.load_codebook`
        load Codebook

    Examples
    --------
    >>> codebook = pd.DataFrame(
    >>>     {
    >>>         0: [None, None, "Morning"],
    >>>         1: ["Male", "No", "Intermediate"],
    >>>         2: ["Female", "Not very often", "Evening"],
    >>>         3: [None, "Often", None],
    >>>         4: [None, "Very often", None]
    >>>     },
    >>>     index=pd.Index(["gender", "smoking", "chronotype"], name="variable")
    >>> )
    >>> apply_codebook(codebook, data)

    """
    is_codebook_dataframe(codebook)

    for col in data.index.names:
        if col in codebook.index:
            data.rename(index=codebook.loc[col], level=col, inplace=True)

    for col in data.columns:
        if col in codebook.index:
            data.loc[:, col].replace(codebook.loc[col], inplace=True)

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
    data : :class:`~pandas.DataFrame`
        pandas DataFrame containing saliva data in wide-format, i.e. one column per saliva sample, one row per subject
    stubname : str
        common name for each column to be converted into long-format. Usually, this is either the name of the
        questionnaire (e.g., "PSS") or the saliva type (e.g., "cortisol").
    levels : str or list of str
        index levels of the resulting long-format dataframe.
    sep : str, optional
        character separating index levels in the column names of the wide-format dataframe. Default: ``_``


    Returns
    -------
    :class:`~pandas.DataFrame`
        pandas DataFrame in long-format


    Examples
    --------
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

    if any(col is None for col in index_cols):
        raise ValueError(
            "All index levels of the dataframe need to have names! Please assign names using "
            "'pandas.Index.set_names()' before using this function!"
        )

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
