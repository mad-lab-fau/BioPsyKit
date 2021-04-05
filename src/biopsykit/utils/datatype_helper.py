"""A couple of helper functions that easy the use of the typical biopsykit data formats."""
from typing import Dict, Optional
import pandas as pd
from biopsykit.utils._datatype_validation_helper import (
    _assert_is_dtype,
    _assert_has_multiindex,
    _assert_has_column_multiindex,
    _assert_has_index_levels,
    _assert_has_columns,
)
from biopsykit.utils.exceptions import ValidationError

__all__ = ["HeartRateSubjectDict", "is_hr_subject_dict"]

HeartRateSubjectDict = Dict[str, pd.DataFrame]
"""Dictionary containing time-series data of `one` subject, split into different phases.

A `HeartRateSubjectDict` is a dictionary with the have the following format:

{ phase_1 : hr_dataframe, phase_2 : hr_dataframe, ... }

Each ``hr_dataframe`` is a :class:`pandas.DataFrame` with the following format:
    * `time` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
    * `Heart_Rate` Column: heart rate values

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
                "HeartRateSubjectDict in an old format can be converted into the new format using "
                "`biopsykit.utils.legacy.legacy_convert_hr_subject_dict()`".format(str(e))
            ) from e
        return False
    return True
