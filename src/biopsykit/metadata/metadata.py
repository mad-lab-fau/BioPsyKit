"""Module containing implementations to compute various metadata parameter.

Each function at least expects a dataframe containing the required columns in a specified order
(see function documentations for specifics) to be passed to the ``data`` argument.

If ``data`` is a dataframe that contains more than the required two columns, e.g., if the complete dataframe
is passed, the required columns can be sliced by specifying them in the ``columns`` parameter.

"""
from typing import Optional, Sequence, Union

import pandas as pd

from biopsykit.utils._datatype_validation_helper import (
    _assert_has_columns,
    _assert_has_index_levels,
    _assert_len_list,
    _assert_value_range,
)

__all__ = ["bmi", "whr", "gender_counts"]

from biopsykit.utils.exceptions import ValidationError


def gender_counts(
    data: pd.DataFrame, gender_col: Optional[str] = None, split_condition: Optional[bool] = False
) -> pd.DataFrame:
    """Get statistics about gender distribution from a dataset.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with subjects
    gender_col : str, optional
        column name containing gender information or ``None`` to use default name ("gender").
    split_condition : bool, optional
        ``True`` to split gender distribution by condition (assumes that an "condition" index level is present in
        ``data``), ``False`` otherwise.
        Default: ``False``

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with absolute and relative gender distribution

    """
    if gender_col is None:
        gender_col = "gender"
    if isinstance(gender_col, str):
        gender_col = [gender_col]

    try:
        _assert_len_list(gender_col, 1)
    except ValidationError as e:
        raise ValidationError(
            f"'gender_col' is excepted to be only one column! Got {len(gender_col)} columns instead."
        ) from e
    _assert_has_columns(data, [gender_col])
    data = data.loc[:, gender_col]
    if split_condition:
        _assert_has_index_levels(data, "condition", match_atleast=True)
        return data.groupby("condition").apply(_gender_counts)
    return _gender_counts(data)


def _gender_counts(data: pd.DataFrame):
    return pd.concat(
        [data.value_counts(sort=False), data.value_counts(normalize=True, sort=False)],
        axis=1,
        keys=["count", "percent"],
    )


def bmi(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **Body Mass Index**.

    This function assumes the required data in the following format:

    * 1st column: weight in kilogram
    * 2nd column: height in centimeter

    If ``data`` is a dataframe that contains more than the required two columns, e.g., if the complete questionnaire
    dataframe is passed, the required columns can be sliced by specifying them in the ``columns`` parameter.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing weight and height information
    columns: sequence of str, optional
        list of column names needed to compute body mass index. Only needed if ``data`` is a dataframe with more than
        the required columns for computing body mass index. Not needed if ``data`` only contains the required columns.
        Default: ``None``

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with body mass index

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValueRangeError`
        if input values or output values are not in the expected range, e.g., because values are provided in the
        wrong unit or columns are in the wrong order

    """
    score_name = "BMI"

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    # weight
    _assert_value_range(data.iloc[:, 0], [10, 300])
    # height
    _assert_value_range(data.iloc[:, 1], [50, 250])

    data = pd.DataFrame(data.iloc[:, 0] / (data.iloc[:, 1] / 100.0) ** 2, columns=[score_name])
    # check if BMI is in a reasonable range
    _assert_value_range(data, [10, 50])
    return data.round(2)


def whr(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **Waist to Hip Ratio**.

    This function assumes the required data in the following format:

    * 1st column: waist circumference
    * 2nd column: hip circumference

    If ``data`` is a dataframe that contains more than the required two columns, e.g., if the complete questionnaire
    dataframe is passed, the required columns can be sliced by specifying them in the ``columns`` parameter.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing waist and hip circumference
    columns: sequence of str, optional
        list of column names needed to compute body mass index. Only needed if ``data`` is a dataframe with more than
        the required columns for computing body mass index. Not needed if ``data`` only contains the required columns.
        Default: ``None``

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with waist to hip ratio

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValueRangeError`
        if input values or output values are not in the expected range, e.g., because values are provided in the
        wrong unit or column are in the wrong order

    """
    score_name = "WHR"

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    data = pd.DataFrame(data.iloc[:, 0] / data.iloc[:, 1], columns=[score_name])
    # check if WHR is in a reasonable range
    _assert_value_range(data, [0.5, 1.5])
    return data.round(3)
