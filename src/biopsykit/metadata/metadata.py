"""Module containing implementations to compute various metadata parameter.

Each function at least expects a dataframe containing the required columns in a specified order
(see function documentations for specifics) to be passed to the ``data`` argument.

If ``data`` is a dataframe that contains more than the required two columns, e.g., if the complete dataframe
is passed, the required columns can be sliced by specifying them in the ``columns`` parameter.

"""
from typing import Optional, Union, Sequence

import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_value_range

__all__ = ["bmi", "whr"]


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
