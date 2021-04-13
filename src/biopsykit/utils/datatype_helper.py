"""A couple of helper functions that easy the use of the typical biopsykit data formats."""
from typing import Dict, Optional, Union, List

import pandas as pd
from biopsykit.utils._datatype_validation_helper import (
    _assert_is_dtype,
    _assert_has_multiindex,
    _assert_has_column_multiindex,
    _assert_has_index_levels,
    _assert_has_columns,
    _assert_has_column_levels,
    _assert_has_column_prefix,
)
from biopsykit.utils.exceptions import ValidationError

__all__ = [
    "HeartRateSubjectDict",
    "SalivaRawDataFrame",
    "SalivaFeatureDataFrame",
    "is_hr_subject_dict",
    "is_raw_saliva_dataframe",
    "is_feature_saliva_dataframe",
]

HeartRateSubjectDict = Dict[str, pd.DataFrame]
"""Dictionary containing time-series data of `one` subject, split into different phases.

A `HeartRateSubjectDict` is a dictionary with the have the following format:

{ phase_1 : hr_dataframe, phase_2 : hr_dataframe, ... }

Each ``hr_dataframe`` is a :class:`pandas.DataFrame` with the following format:
    * `time` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
    * `Heart_Rate` Column: heart rate values

"""

SalivaRawDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing raw saliva data in a standardized format.

Data needs to be in long-format and **must** have a :class:`pandas.MultiIndex` with index level names:
    * `subject`: subject ID; can be number or string
    * `sample`: saliva sample ID; can be number or string

Additionally, the following index levels can be added to identify saliva values, such as:
    * `condition`: subject condition during the study (e.g., "Control" vs. "Condition")
    * `day`: day ID, if saliva samples were collected over multiple days
    * `night`: night ID, if saliva samples were collected over multiple night
    * ...

"""

SalivaFeatureDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing feature computed from saliva data in a standardized format.

The resulting dataframe must at least have a `subject` index level and all column names need to begin with 
the saliva marker type (e.g. "cortisol"), followed by the feature name, separated by underscore '_'
Additionally, the name of the column index needs to be `saliva_feature`.

"""


def is_hr_subject_dict(data: HeartRateSubjectDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a `HeartRateSubjectDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a `HeartRateSubjectDict`, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a `HeartRateSubjectDict`

    See Also
    --------
    `HeartRateSubjectDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for df in data.values():
            _assert_is_dtype(df, pd.DataFrame)
            _assert_has_multiindex(df, expected=False)
            _assert_has_column_multiindex(df, expected=False)
            _assert_has_columns(df, [["Heart_Rate"]])
            _assert_has_index_levels(df, ["time"])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a HeartRateSubjectDict. "
                "The validation failed with the following error:\n\n{}\n"
                "HeartRateSubjectDict's in an old format can be converted into the new format using "
                "`biopsykit.utils.legacy.legacy_convert_hr_subject_dict()`".format(str(e))
            ) from e
        return False
    return True


def is_raw_saliva_dataframe(
    data: SalivaRawDataFrame, saliva_type: Union[str, List[str]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a `SalivaRawDataFrame`.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a `SalivaRawDataFrame`
    saliva_type : str or list of str
        type of saliva data (or list of saliva types) in the dataframe, e.g., "cortisol" or "amylase"
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a `SalivaRawDataFrame`, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a `SalivaRawDataFrame`

    See Also
    --------
    `SalivaRawDataFrame`
        dataframe format

    """
    try:
        if saliva_type is None:
            raise ValidationError("`saliva_type` is None!")
        if isinstance(saliva_type, str):
            saliva_type = [saliva_type]
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_multiindex(data, nlevels=2, nlevels_atleast=True)
        _assert_has_index_levels(data, index_levels=["subject", "sample"], match_atleast=True, match_order=False)
        _assert_has_columns(data, [saliva_type, saliva_type + ["time"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SalivaRawDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_feature_saliva_dataframe(
    data: SalivaFeatureDataFrame, saliva_type: str, raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a `SalivaFeatureDataFrame`.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a `SalivaFeatureDataFrame`
    saliva_type : str or list of str
        type of saliva data in the dataframe, e.g., "cortisol" or "amylase"
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a `SalivaFeatureDataFrame`, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a `SalivaFeatureDataFrame`

    See Also
    --------
    `SalivaFeatureDataFrame+`
        dataframe format

    """
    try:
        if saliva_type is None:
            raise ValidationError("`saliva_type` is None!")
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_index_levels(data, index_levels="subject", match_atleast=True, match_order=False)
        _assert_has_column_levels(data, column_levels="saliva_feature", match_atleast=True, match_order=False)
        _assert_has_column_prefix(data.columns, prefix=saliva_type)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SalivaFeatureDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True
