"""Module containing utility functions for manipulating and processing questionnaire data."""
import warnings
import re

from typing import Optional, Union, Sequence, Tuple, Dict, Any
from inspect import getmembers, isfunction

from typing_extensions import Literal

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_value_range, _assert_len_list
from biopsykit.utils.dataframe_handling import wide_to_long as wide_to_long_utils
from biopsykit.questionnaires import questionnaires


__all__ = [
    "bin_scale",
    "compute_scores",
    "crop_scale",
    "convert_scale",
    "find_cols",
    "zero_pad_columns",
    "invert",
    "to_idx",
    "wide_to_long",
    "get_supported_questionnaires",
]


def find_cols(
    data: pd.DataFrame,
    regex_str: Optional[str] = None,
    starts_with: Optional[str] = None,
    ends_with: Optional[str] = None,
    contains: Optional[str] = None,
    zero_pad_numbers: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    r"""Find columns in dataframe that match a specific pattern.

    This function is useful to find all columns that belong to a questionnaire. Column names can be filtered based on
    one (or a combination of) the following criteria:

    * ``starts_with``: columns have to start with the specified string
    * ``ends_with``: columns have to end with the specified string
    * ``contains``: columns have to contain the specified string

    Optionally, the item numbers in the matching column names can be zero-padded, if they are not already.

    .. note::
        If ``zero_pad_numbers`` is ``True`` then the column names returned by this function will be renamed and might
        thus not match the column names of the original dataframe. To solve this, make sure your orignal dataframe
        already has zero-padded columns (by manually renaming them) or convert column names using
        :func:`~biopsykit.questionnaires.utils.zero_pad_columns`.

    .. warning::
        Zero-padding using :func:`~biopsykit.questionnaires.utils.zero_pad_columns` assumes, by default, that numbers
        are *at the end* of column names. If you want to change that behavior
        (e.g., because the column names have string suffixes), you might need to apply zero-padding manually.



    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with columns to be filtered
    regex_str : str, optional
        regex string to extract column names. If this parameter is passed the other parameters (``starts_with``,
        ``ends_with``, ``contains``) will be ignored. Default: ``None``
    starts_with : str, optional
        string columns have to start with. Default: ``None``
    ends_with : str, optional
        string columns have to end with. Default: ``None``
    contains : str, optional
        string columns have to contain. Default: ``None``
    zero_pad_numbers : bool, optional
        whether to zero-pad numbers in column names. Default: ``True``

    Returns
    -------
    data_filt : :class:`~pandas.DataFrame`
        dataframe with filtered columns that match the specified pattern
    cols : :class:`~pandas.Index`
        columns that match the specified pattern

    Examples
    --------
    >>> import biopsykit as bp
    >>> import pandas as pd
    >>> # Option 1: has to start with "XX"
    >>> data = pd.DataFrame(columns=["XX_{}".format(i) for i in range(1, 11)])
    >>> df, cols = bp.questionnaires.utils.find_cols(data, starts_with="XX")
    >>> print(cols)
    >>> ["XX_01", "XX_02", ..., "XX_10"]
    >>> # Option 2: has to end with "Post"
    >>> data = pd.DataFrame(columns=["XX_1_Pre", "XX_2_Pre", "XX_3_Pre", "XX_1_Post", "XX_2_Post", "XX_3_Post"])
    >>> df, cols = bp.questionnaires.utils.find_cols(data, ends_with="Post")
    >>> print(cols)
    >>> ["XX_01_Post", "XX_02_Post", "XX_03_Post"]
    >>> # Option 3: has to start with "XX" and end with "Post"
    >>> data = pd.DataFrame(columns=["XX_1_Pre", "XX_2_Pre", "XX_3_Pre", "XX_1_Post", "XX_2_Post", "XX_3_Post",
     "YY_1_Pre", "YY_2_Pre", "YY_1_Post", "YY_2_Post"])
    >>> bp.questionnaires.utils.find_cols(data, starts_with="XX", ends_with="Post")
    >>> print(cols)
    >>> # WARNING: this will not zero-pad the questionnaire numbers!
    >>> ["XX_1_Post", "XX_2_Post", "XX_3_Post"]
    >>> # Option 4: pass custom regex string
    >>> data = pd.DataFrame(columns=["XX_1_Pre", "XX_2_Pre", "XX_3_Pre", "XX_1_Post", "XX_2_Post", "XX_3_Post",
     "YY_1_Pre", "YY_2_Pre", "YY_1_Post", "YY_2_Post"])
    >>> bp.questionnaires.utils.find_cols(data, regex_str=r"XX_\d+_\w+")
    >>> print(cols)
    >>> # here, zero-padding will be possible again
    >>> ["XX_01_Post", "XX_02_Post", "XX_03_Post"]
    >>> # Option 5: disable zero-padding
    >>> data = pd.DataFrame(columns=["XX_{}".format(i) for i in range(1, 11)])
    >>> df, cols = bp.questionnaires.utils.find_cols(data, starts_with="XX", zero_pad_numbers=False)
    >>> print(cols)
    >>> ["XX_1", "XX_2", ..., "XX_10"]

    """
    _assert_is_dtype(data, pd.DataFrame)
    data_filt = data.copy()

    if regex_str:
        data_filt = data_filt.filter(regex=regex_str)
    else:
        if starts_with:
            data_filt = data_filt.filter(regex="^" + starts_with)
        if ends_with:
            data_filt = data_filt.filter(regex=ends_with + "$")
        if contains:
            data_filt = data_filt.filter(regex=contains)

    if zero_pad_numbers:
        data_filt = zero_pad_columns(data_filt)
    cols = data_filt.columns

    return data_filt, cols


def zero_pad_columns(data: pd.DataFrame, inplace: Optional[bool] = False) -> Optional[pd.DataFrame]:
    r"""Add zero-padding to numbers at the **end** of column names in a dataframe.

    .. warning::
        By default, this function assumes that numbers are **at the end** of column names. If you need to change that
        behavior (e.g., because the column names have string suffixes), you might need to apply zero-padding manually.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with columns to zero-pad
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame` or ``None``
        dataframe with zero-padded columns or ``None`` if ``inplace`` is ``True``

    """
    _assert_is_dtype(data, pd.DataFrame)
    if not inplace:
        data = data.copy()

    nums = [re.findall(r"(\d+)$", c) for c in data.columns]
    nums = [c[0] if len(c) > 0 else "" for c in nums]
    if len(nums) == 0:
        return pd.DataFrame()
    zfill_num = max(max(list(map(len, nums))), 2)

    data.columns = [re.sub(r"(\d+)$", lambda m: m.group(1).zfill(zfill_num), c) for c in data.columns]

    if inplace:
        return None
    return data


def to_idx(col_idxs: Union[np.array, Sequence[int]]) -> np.ndarray:
    """Convert questionnaire item indices into array indices.

    In questionnaires, items indices start at 1. To avoid confusion in the implementation of questionnaires
    (because array indices start at 0) all questionnaire indices in BioPsyKit also start at 1 and are converted to
    0-based indexing using this function.

    Parameters
    ----------
    col_idxs : list of int
        list of indices to convert to 0-based indexing

    Returns
    -------
    :class:`~numpy.ndarray`
        array with converted indices

    """
    return np.array(col_idxs) - 1


def invert(
    data: Union[pd.DataFrame, pd.Series],
    score_range: Sequence[int],
    cols: Optional[Union[np.array, Sequence[int], Sequence[str]]] = None,
    inplace: Optional[bool] = False,
) -> Optional[Union[pd.DataFrame, pd.Series]]:
    """Invert questionnaire scores.

    In many questionnaires some items need to be inverted (reversed) before sum scores can be computed. This function
    can be used to either invert a single column (Series), selected columns in a dataframe (by specifying columns
    in the ``cols`` parameter), or a complete dataframe.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        questionnaire data to invert
    score_range : list of int
        possible score range of the questionnaire items
    cols : list of str or list of int
        list of column names or column indices
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame` or ``None``
        dataframe with inverted columns or ``None`` if ``inplace`` is ``True``


    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is no dataframe or series
        if ``score_range`` does not have length 2
    :exc:`~biopsykit.utils.exceptions.ValueRangeError`
        if values in ``data`` are not in ``score_range``


    Examples
    --------
    >>> from biopsykit.questionnaires.utils import invert
    >>> data_in = pd.DataFrame({"A": [1, 2, 3, 1], "B": [4, 0, 1, 3], "C": [0, 3, 2, 3], "D": [0, 1, 2, 4]})
    >>> data_out = invert(data_in, score_range=[0, 4])
    >>> data_out["A"]
    >>> [3, 2, 1, 3]
    >>> data_out["B"]
    >>> [0, 4, 3, 1]
    >>> data_out["C"]
    >>> [4, 1, 2, 1]
    >>> data_out["D"]
    >>> [4, 3, 2, 0]
    >>> # Other score range
    >>> data_out = invert(data, score_range=[0, 5])
    >>> data_out["A"]
    >>> [3, 2, 1, 3]
    >>> data_out["B"]
    >>> [1, 5, 4, 2]
    >>> data_out["C"]
    >>> [5, 2, 3, 2]
    >>> data_out["D"]
    >>> [5, 4, 3, 1]
    >>> # Invert only specific columns
    >>> data_out = invert(data, score_range=[0, 4], cols=["A", "C"])
    >>> data_out["A"]
    >>> [3, 2, 1, 3]
    >>> data_out["B"]
    >>> [4, 0, 1, 3]
    >>> data_out["C"]
    >>> [4, 1, 2, 1]
    >>> data_out["D"]
    >>> [0, 1, 2, 4]

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))
    _assert_len_list(score_range, 2)

    if not inplace:
        data = data.copy()

    if isinstance(data, pd.DataFrame):
        data = _invert_dataframe(data, cols, score_range)
    else:
        _assert_value_range(data, score_range)
        data.iloc[:] = score_range[1] - data.iloc[:] + score_range[0]

    if inplace:
        return None
    return data


def _invert_dataframe(
    data: pd.DataFrame, cols: Union[Sequence[str], Sequence[int]], score_range: Sequence[int]
) -> pd.DataFrame:
    if cols is not None:
        if isinstance(cols[0], str):
            _assert_value_range(data[cols], score_range)
            data.loc[:, cols] = score_range[1] - data.loc[:, cols] + score_range[0]
        else:
            _assert_value_range(data.iloc[:, cols], score_range)
            data.iloc[:, cols] = score_range[1] - data.iloc[:, cols] + score_range[0]
    else:
        _assert_value_range(data, score_range)
        data.iloc[:, :] = score_range[1] - data.iloc[:, :] + score_range[0]

    return data


def _invert_subscales(
    data: pd.DataFrame,
    subscales: Dict[str, Sequence[Union[str, int]]],
    idx_dict: Dict[str, Sequence[int]],
    score_range: Sequence[int],
) -> pd.DataFrame:
    """Invert questionnaire scores from a dictionary of questionnaire subscales.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        questionnaire data to invert
    subscales : dict
        dictionary with subscale names (keys) and list of item indices or column names belonging to the
        individual subscales (values)
    idx_dict : dict
        dictionary with subscale names (keys) and indices of items that should be inverted (values)
    score_range : list of int
        possible score range of the questionnaire items

    Returns
    -------
    :class:`~pandas.DataFrame` or ``None``
        dataframe with inverted columns

    See Also
    --------
    invert : invert scores of questionnaire columns

    """
    _assert_is_dtype(data, pd.DataFrame)

    for scale_name, idxs in idx_dict.items():
        if scale_name in subscales:
            data = invert(data, cols=to_idx(np.array(subscales[scale_name])[idxs]), score_range=score_range)
    return data


def convert_scale(
    data: Union[pd.DataFrame, pd.Series],
    offset: int,
    cols: Optional[Union[pd.DataFrame, pd.Series]] = None,
    inplace: Optional[bool] = False,
) -> Optional[Union[pd.DataFrame, pd.Series]]:
    """Convert the score range of questionnaire items.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        questionnaire data to invert
    offset : int
        offset to add to questionnaire items
    cols : list of str or list of int
        list of column names or column indices
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame`, :class:`~pandas.Series`, or ``None``
        dataframe with converted columns or ``None`` if ``inplace`` is ``True``


    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is no dataframe or series

    Examples
    --------
    >>> from biopsykit.questionnaires.utils import convert_scale
    >>> data_in = pd.DataFrame({"A": [1, 2, 3, 1], "B": [4, 0, 1, 3], "C": [0, 3, 2, 3], "D": [0, 1, 2, 4]})
    >>> # convert data from range [0, 4] to range [1, 5]
    >>> data_out = convert_scale(data_in, offset=1)
    >>> data_out["A"]
    >>> [2, 3, 4, 2]
    >>> data_out["B"]
    >>> [5, 1, 2, 4]
    >>> data_out["C"]
    >>> [1, 4, 3, 4]
    >>> data_out["D"]
    >>> [1, 2, 3, 5]
    >>> data_in = pd.DataFrame({"A": [1, 2, 3, 1], "B": [4, 2, 1, 3], "C": [3, 3, 2, 3], "D": [4, 1, 2, 4]})
    >>> # convert data from range [1, 4] to range [0, 3]
    >>> data_out = convert_scale(data_in, offset=-1)
    >>> print(data_out)
    >>> # convert only specific columns
    >>> data_out = convert_scale(data_in, offset=-1, columns=["A", "C"])
    >>> print(data_out)

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))

    if not inplace:
        data = data.copy()

    if isinstance(data, pd.DataFrame):
        data = _convert_scale_dataframe(data, cols, offset)
    else:
        data.iloc[:] = data.iloc[:] + offset

    if inplace:
        return None
    return data


def _convert_scale_dataframe(
    data: pd.DataFrame, cols: Union[Sequence[int], Sequence[str]], offset: int
) -> pd.DataFrame:
    if cols is None:
        data.iloc[:, :] = data.iloc[:, :] + offset
    else:
        if isinstance(cols[0], int):
            data.iloc[:, cols] = data.iloc[:, cols] + offset
        elif isinstance(cols[0], str):
            data.loc[:, cols] = data.loc[:, cols] + offset

    return data


def crop_scale(
    data: Union[pd.DataFrame, pd.Series],
    score_range: Sequence[int],
    set_nan: Optional[bool] = False,
    inplace: Optional[bool] = False,
) -> Optional[Union[pd.DataFrame, pd.Series]]:
    """Crop questionnaire scales, i.e., set values out of range to specific minimum and maximum values or to NaN.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        data to be cropped
    score_range : list of int
        possible score range of the questionnaire items. Values out of ``score_range`` are cropped.
    set_nan : bool, optional
        whether to set values out of range to NaN or to the values specified by ``score_range``. Default: ``False``
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame`, :class:`~pandas.Series`, or ``None``
        dataframe (or series) with cropped scales or ``None`` if ``inplace`` is ``True``

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))
    _assert_len_list(score_range, 2)

    if not inplace:
        data = data.copy()

    if set_nan:
        data.mask((data < score_range[0]) | (data > score_range[1]), inplace=True)
    else:
        data.mask((data < score_range[0]), other=score_range[0], inplace=True)
        data.mask((data > score_range[1]), other=score_range[1], inplace=True)

    if inplace:
        return None
    return data


def bin_scale(
    data: Union[pd.DataFrame, pd.Series],
    bins: Union[int, Sequence[float], pd.IntervalIndex],
    cols: Optional[Union[Sequence[Union[int, str]], Union[int, str]]] = None,
    first_min: Optional[bool] = True,
    last_max: Optional[bool] = False,
    inplace: Optional[bool] = False,
    **kwargs,
) -> Optional[Union[pd.Series, pd.DataFrame]]:
    """Bin questionnaire scales.

    Questionnaire scales are binned using :func:`pandas.cut` according to the bins specified by ``bins``.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        data with scales to be binned
    bins : int or list of float or :class:`~pandas.IntervalIndex``
        The criteria to bin by. ``bins`` can have one of the following types:

        * ``int`` : Defines the number of equal-width bins in the range of ``data``. The range of ``data`` is extended
          by 0.1% on each side to include the minimum and maximum values of ``data``.
        * sequence of scalars : Defines the bin edges allowing for non-uniform width. No extension of the range of
          ``data`` is done.
        * :class:`~pandas.IntervalIndex` : Defines the exact bins to be used. Note that the ``IntervalIndex`` for
          ``bins`` must be non-overlapping.

    cols : list of str or list of int, optional
        column name/index (or list of such) to be binned or ``None`` to use all columns (or if ``data`` is a series).
        Default: ``None``
    first_min : bool, optional
        whether the minimum value should be added as the leftmost edge of the last bin or not.
        Only considered if ``bins`` is a list. Default: ``False``
    last_max : bool, optional
        whether the maximum value should be added as the rightmost edge of the last bin or not.
        Only considered if ``bins`` is a list. Default: ``False``
    inplace : bool, optional
        whether to perform the operation inplace or not. Default: ``False``
    **kwargs
        additional parameters that are passed to :func:`pandas.cut`


    Returns
    -------
    :class:`~pandas.DataFrame`, :class:`~pandas.Series`, or ``None``
        dataframe (or series) with binned scales or ``None`` if ``inplace`` is ``True``


    See Also
    --------
    :func:`pandas.cut`
        Pandas method to bin values into discrete intervals.

    """
    _assert_is_dtype(data, (pd.Series, pd.DataFrame))

    if not inplace:
        data = data.copy()

    # set "labels" argument to False, but only if is wasn't specified by the user yet
    kwargs["labels"] = kwargs.get("labels", False)
    if isinstance(data, pd.Series):
        bins_c = _get_bins(data, bins, None, first_min, last_max)
        c = pd.cut(data.iloc[:], bins=bins_c, **kwargs)
        data.iloc[:] = c
        return data

    cols = _get_cols(data, cols)
    for col in cols:
        bins_c = _get_bins(data, bins, col, first_min, last_max)
        if isinstance(col, int):
            c = pd.cut(data.iloc[:, col], bins=bins_c, **kwargs)
            data.iloc[:, col] = c
        else:
            c = pd.cut(data.loc[:, col], bins=bins_c, **kwargs)
            data.loc[:, col] = c

    if inplace:
        return None
    return data


def wide_to_long(data: pd.DataFrame, quest_name: str, levels: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Convert a dataframe wide-format into long-format.

    .. warning::
        This function is deprecated and will be removed in the future!
        Please use :func:`~biopsykit.utils.dataframe_handling.wide_to_long` instead.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        pandas DataFrame containing saliva data in wide-format, i.e. one column per saliva sample, one row per subject.
    quest_name : str
        questionnaire name, i.e., common name for each column to be converted into long-format.
    levels : str or list of str
        index levels of the resulting long-format dataframe.


    Returns
    -------
    :class:`~pandas.DataFrame`
        pandas DataFrame in long-format

    See Also
    --------
    :func:`~biopsykit.utils.dataframe_handling.wide_to_long`
        convert dataframe from wide to long format

    """
    warnings.warn(
        "'biopsykit.questionnaires.utils.wide_to_long()' is deprecated! "
        "Please update your code to use 'biopsykit.utils.dataframe_handling.wide_to_long()' in the future.",
        category=DeprecationWarning,
    )
    return wide_to_long_utils(data=data, stubname=quest_name, levels=levels)


def compute_scores(
    data: pd.DataFrame,
    quest_dict: Dict[str, Union[Sequence[str], pd.Index]],
    quest_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Compute questionnaire scores from dataframe.

    This function can be used if multiple questionnaires from a dataframe should be computed at once. If the same
    questionnaire was assessed at multiple time points, these scores will be computed separately
    (see ``Notes`` and ``Examples``).
    The questionnaires (and the dataframe columns belonging to the questionnaires) are specified by ``quest_dict``.

    .. note::
        If questionnaires were collected at different time points (e.g., `pre` and `post`), which should all be
        computed, then the dictionary keys need to have the following format: "<questionnaire_name>-<time_point>".

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data
    quest_dict : dict
        dictionary with questionnaire names to be computed (keys) and columns of the questionnaires (values)
    quest_kwargs : dict
        dictionary with optional arguments to be passed to questionnaire functions. The dictionary is expected
        consist of questionnaire names (keys) and ``**kwargs`` dictionaries (values) with arguments per questionnaire

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with computed questionnaire scores

    Examples
    --------
    >>> from biopsykit.questionnaires.utils import compute_scores
    >>> quest_dict = {
    >>>     "PSS": ["PSS_{:02d}".format(i) for i in range(1, 11)], # PSS: one time point
    >>>     "PASA-pre": ["PASA_{:02d}_T0".format(i) for i in range(1, 17)], # PASA: two time points (pre and post)
    >>>     "PASA-post": ["PASA_{:02d}_T1".format(i) for i in range(1, 17)], # PASA: two time points (pre and post)
    >>> }
    >>> compute_scores(data, quest_dict)

    """
    _assert_is_dtype(data, pd.DataFrame)

    df_scores = pd.DataFrame(index=data.index)
    quest_funcs = dict(getmembers(questionnaires, isfunction))
    if quest_kwargs is None:
        quest_kwargs = {}

    for score, columns in quest_dict.items():
        score_orig = score
        score = score.lower()
        suffix = None
        if "-" in score:
            score_split = score.split("-")
            score = score_split[0]
            suffix = score_split[1]

        if score not in quest_funcs:
            raise ValueError(
                "Unknown questionnaire '{}'! Call "
                "'biopsykit.questionnaires.utils.get_supported_questionnaires()' "
                "to get a list of all supported questionnaires.".format(score)
            )
        kwargs = quest_kwargs.get(score_orig, {})
        try:
            df = quest_funcs[score](data[columns], **kwargs)
        except TypeError as e:
            raise TypeError(
                "Error computing questionnaire '{}'. The computation failed with the following "
                "error: \n\n{}.".format(score, str(e))
            ) from e

        if suffix is not None:
            df.columns = ["{}_{}".format(col, suffix) for col in df.columns]
        df_scores = df_scores.join(df)

    return df_scores


def get_supported_questionnaires() -> Dict[str, str]:
    """List all supported (i.e., implemented) questionnaires.

    Returns
    -------
    dict
        dictionary with questionnaire names (keys) and description (values)

    """
    funcs = dict(getmembers(questionnaires, isfunction))
    quests = {}
    for key, value in funcs.items():
        if key.startswith("_"):
            continue
        summary = value.__doc__.split("\n")[0]
        summary = re.findall(r"\*\*(.*)\*\*.", summary)
        if len(summary) == 0:
            continue
        quests[key] = summary[0]

    return quests


def _compute_questionnaire_subscales(
    data: pd.DataFrame,
    score_name: str,
    subscales: Dict[str, Sequence[Union[str, int]]],
    agg_type: Optional[Literal["sum", "mean"]] = "sum",
) -> Dict[str, pd.Series]:
    """Compute questionnaire subscales (helper function).

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data
    score_name : str
        name of the questionnaire
    subscales : dict
        dictionary with subscales to be computed. Keys are subscale names, values are the indices of the items
        belonging to the subscales
    agg_type : str
        whether to compute a ``sum`` or a ``mean`` score. Default: ``sum``

    Returns
    -------
    dict
        dictionary with computed subscales

    """
    _assert_is_dtype(data, pd.DataFrame)

    out = {}
    for key, items in subscales.items():
        if all(np.issubdtype(type(i), np.integer) for i in items):
            # assume column indices, starting at 1 (-> convert to 0-indexed indices first)
            score = _compute_questionnaire_scores_int(data, items, agg_type)
        elif all(isinstance(i, str) for i in items):
            # assume column names
            score = _compute_questionnaire_scores_str(data, items, agg_type)
        else:
            raise ValueError(
                "Subscale columns are either expected as column names (list of strings) or "
                "column indices (list of integers)!"
            )

        out["{}_{}".format(score_name, key)] = score

    return out


def _compute_questionnaire_scores_int(data: pd.DataFrame, items: Sequence[int], agg_type: str):
    if agg_type == "sum":
        return data.iloc[:, to_idx(items)].sum(axis=1)
    return data.iloc[:, to_idx(items)].mean(axis=1)


def _compute_questionnaire_scores_str(data: pd.DataFrame, items: Sequence[str], agg_type: str):
    if agg_type == "sum":
        return data.loc[:, items].sum(axis=1)
    return data.loc[:, items].mean(axis=1)


def _get_cols(
    data: pd.DataFrame, cols: Optional[Union[Sequence[Union[int, str]], Union[int, str]]] = None
) -> Sequence[Union[str, int]]:
    if isinstance(cols, int):
        cols = [cols]
    if isinstance(cols, str):
        cols = [cols]
    if cols is None:
        cols = list(data.columns)
    return cols


def _get_bins(
    data: Union[pd.DataFrame, pd.Series],
    bins: Union[int, Sequence[float]],
    col: Optional[Union[int, str]] = None,
    first_min: Optional[bool] = False,
    last_max: Optional[bool] = False,
) -> Union[int, Sequence[float]]:

    if isinstance(bins, (int, pd.IntervalIndex)):
        return bins

    # ensure list
    bins = list(bins)

    if first_min:
        min_val = _bin_scale_get_min_val(data, col)
        if min_val < min(bins):
            bins = [min_val - 0.01] + bins

    if last_max:
        max_val = _bin_scale_get_max_val(data, col)
        if max_val > max(bins):
            bins = bins + [max_val + 0.01]

    return bins


def _bin_scale_get_min_val(data: Union[pd.DataFrame, pd.Series], col: Union[int, str]) -> float:
    if isinstance(col, int):
        return data.iloc[:, col].min()
    if isinstance(col, str):
        return data[col].min()
    return data.min()


def _bin_scale_get_max_val(data: Union[pd.DataFrame, pd.Series], col: Union[int, str]) -> float:
    if isinstance(col, int):
        return data.iloc[:, col].max()
    if isinstance(col, str):
        return data[col].max()
    return data.max()
