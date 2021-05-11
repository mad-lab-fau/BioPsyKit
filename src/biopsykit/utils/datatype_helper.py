"""A couple of helper functions that easy the use of the typical biopsykit data formats."""
from typing import Dict, Optional, Union, List, Any

import numpy as np
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
    "SubjectConditionDataFrame",
    "SubjectConditionDict",
    "HeartRateSubjectDict",
    "SalivaRawDataFrame",
    "SalivaFeatureDataFrame",
    "SalivaMeanSeDataFrame",
    "SleepEndpointDataFrame",
    "SleepEndpointDict",
    "EcgRawDataFrame",
    "is_subject_condition_dataframe",
    "is_subject_condition_dict",
    "is_hr_subject_dict",
    "is_saliva_raw_dataframe",
    "is_saliva_feature_dataframe",
    "is_saliva_mean_se_dataframe",
    "is_sleep_endpoint_dataframe",
    "is_sleep_endpoint_dict",
    "is_ecg_raw_dataframe",
]

SubjectConditionDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing subject IDs and condition assignment in a standardized format.

A ``SubjectConditionDataFrame`` has an index with subject IDs named ``subject`` and a column with the condition 
assignment named ``condition``.  

"""

SubjectConditionDict = Dict[str, np.ndarray]
"""Dictionary containing subject IDs and condition assignment in a standardized format.

A ``SubjectConditionDict`` contains conditions as dictionary keys and a collection of subject IDs 
(list, numpy array, pandas Index) as dictionary values.

"""

HeartRateSubjectDict = Dict[str, pd.DataFrame]
"""Dictionary containing time-series data of `one` subject, split into different phases.

A ``HeartRateSubjectDict`` is a dictionary with the have the following format:

{ phase_1 : hr_dataframe, phase_2 : hr_dataframe, ... }

Each ``hr_dataframe`` is a :class:`pandas.DataFrame` with the following format:
    * ``time`` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
    * ``Heart_Rate`` Column: heart rate values

"""

SalivaRawDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing raw saliva data in a standardized format.

Data needs to be in long-format and **must** have a :class:`pandas.MultiIndex` with index level names:
    * ``subject``: subject ID; can be number or string
    * ``sample``: saliva sample ID; can be number or string

Additionally, the following index levels can be added to identify saliva values, such as:
    * ``condition``: subject condition during the study (e.g., "Control" vs. "Condition")
    * ``day``: day ID, if saliva samples were collected over multiple days
    * ``night``: night ID, if saliva samples were collected over multiple night
    * ...

"""

SalivaFeatureDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing feature computed from saliva data in a standardized format.

The resulting dataframe must at least have a ``subject`` index level and all column names need to begin with 
the saliva marker type (e.g. "cortisol"), followed by the feature name, separated by underscore '_'
Additionally, the name of the column index needs to be `saliva_feature`.

"""

SalivaMeanSeDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing mean and standard error of saliva samples in a standardized format.

The resulting dataframe must at least have a ``sample`` index level and the two columns ``mean`` and ``se``. 
It can have additional index levels, such as ``condition`` or ``time``.

"""

SleepEndpointDict = Dict[str, Any]
"""Dictionary containing sleep endpoints in a standardized format.

The dict entries represent the sleep endpoints and should follow a standardized naming convention,
regardless of the origin (IMU sensor, sleep mattress, psg, etc.).

Required are the entries:
    * ``sleep_onset``: Sleep Onset, i.e., time of falling asleep, in absolute time
    * ``wake_onset``: Wake Onset, i.e., time of awakening, in absolute time
    * ``total_sleep_duration``: Total sleep duration, i.e., time between Sleep Onset and Wake Onset, in minutes

The following entries are common, but not required:
    * ``total_duration``: Total recording time, in minutes
    * ``net_sleep_duration``: Net duration spent sleeping, in minutes
    * ``num_wake_bouts``: Total number of wake bouts
    * ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep, in minutes
    * ``getup_onset_latency``: Get Up Latency, i.e., time in bed after awakening until getting up, in minutes
    * ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep, in minutes

The following entries are, for instance, further possible:
    * ``total_time_light_sleep``: Total time of light sleep, in minutes
    * ``total_time_deep_sleep``: Total time of deep sleep, in minutes
    * ``total_time_rem_sleep``: Total time of REM sleep, in minutes
    * ``total_time_awake``: Total time of being awake, in minutes
    * ``count_snoring_episodes``: Total number of snoring episodes
    * ``total_time_snoring``: Total time of snoring, in minutes
    * ``heart_rate_avg``: Average heart rate during recording, in bpm
    * ``heart_rate_min``: Minimum heart rate during recording, in bpm
    * ``heart_rate_max``: Maximum heart rate during recording, in bpm

"""

SleepEndpointDataFrame = pd.DataFrame
""":class:`pandas.DataFrame` containing sleep endpoints in a standardized format.

The resulting dataframe must at least have a ``date`` index level, 
and, optionally, further index levels like ``night``.

The columns defining the sleep endpoints should follow a standardized naming convention, regardless of the origin
(IMU sensor, sleep mattress, psg, etc.).

Required are the columns:
    * ``sleep_onset``: Sleep Onset, i.e., time of falling asleep, in absolute time
    * ``wake_onset``: Wake Onset, i.e., time of awakening, in absolute time
    * ``total_sleep_duration``: Total sleep duration, i.e., time between Sleep Onset and Wake Onset, in minutes

The following columns are common, but not required:
    * ``total_duration``: Total recording time, in minutes
    * ``net_sleep_duration``: Net duration spent sleeping, in minutes
    * ``num_wake_bouts``: Total number of wake bouts
    * ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep, in minutes
    * ``getup_onset_latency``: Get Up Latency, i.e., time in bed after awakening until getting up, in minutes
    * ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep, in minutes

The following columns are further possible:
    * ``total_time_light_sleep``: Total time of light sleep, in minutes
    * ``total_time_deep_sleep``: Total time of deep sleep, in minutes
    * ``total_time_rem_sleep``: Total time of REM sleep, in minutes
    * ``total_time_awake``: Total time of being awake, in minutes
    * ``count_snoring_episodes``: Total number of snoring episodes
    * ``total_time_snoring``: Total time of snoring, in minutes
    * ``heart_rate_avg``: Average heart rate during recording, in bpm
    * ``heart_rate_min``: Minimum heart rate during recording, in bpm
    * ``heart_rate_max``: Maximum heart rate during recording, in bpm 

"""


EcgRawDataFrame = pd.DataFrame
""":class:`~pandas.DataFrame` containing raw ECG data as time-series of `one` subject.

The dataframe is expected to have the following format:
    * ``ecg`` column: raw ECG samples

"""


def is_subject_condition_dataframe(
    data: SubjectConditionDataFrame, raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a ``SubjectConditionDataFrame``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a ``SubjectConditionDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SubjectConditionDataFrame``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SubjectConditionDataFrame``

    See Also
    --------
    ``SubjectConditionDataFrame``
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_multiindex(data, expected=False)
        _assert_has_index_levels(data, index_levels=["subject"], match_atleast=False, match_order=True)
        _assert_has_columns(data, [["condition"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SubjectConditionDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_subject_condition_dict(data: SubjectConditionDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a ``SubjectConditionDict``.

    Parameters
    ----------
    data : dict
        dict to check if it is a ``SubjectConditionDict``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SubjectConditionDict``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SubjectConditionDict``

    See Also
    --------
    ``SubjectConditionDict``
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for val in data.values():
            _assert_is_dtype(val, (np.ndarray, list, pd.Index))
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SubjectConditionDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_hr_subject_dict(data: HeartRateSubjectDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a ``HeartRateSubjectDict``.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``HeartRateSubjectDict``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``HeartRateSubjectDict``

    See Also
    --------
    ``HeartRateSubjectDict``
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


def is_saliva_raw_dataframe(
    data: SalivaRawDataFrame, saliva_type: Union[str, List[str]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a ``SalivaRawDataFrame``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a ``SalivaRawDataFrame``
    saliva_type : str or list of str
        type of saliva data (or list of saliva types) in the dataframe, e.g., "cortisol" or "amylase"
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SalivaRawDataFrame``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SalivaRawDataFrame``

    See Also
    --------
    ``SalivaRawDataFrame``
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


def is_saliva_feature_dataframe(
    data: SalivaFeatureDataFrame, saliva_type: str, raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a ``SalivaFeatureDataFrame``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a ``SalivaFeatureDataFrame``
    saliva_type : str or list of str
        type of saliva data in the dataframe, e.g., "cortisol" or "amylase"
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SalivaFeatureDataFrame``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SalivaFeatureDataFrame``

    See Also
    --------
    ``SalivaFeatureDataFrame``
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


def is_saliva_mean_se_dataframe(data: SalivaFeatureDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a ``SalivaMeanSeDataFrame``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a ``SalivaMeanSeDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SalivaMeanSeDataFrame``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SalivaMeanSeDataFrame``

    See Also
    --------
    ``SalivaMeanSeDataFrame``
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_index_levels(data, index_levels="sample", match_atleast=True, match_order=False)
        _assert_has_columns(data, [["mean", "se"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SalivaMeanSeDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_sleep_endpoint_dataframe(data: SleepEndpointDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a ``SleepEndpointDataFrame``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a ``SleepEndpointDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SleepEndpointDataFrame``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SleepEndpointDataFrame``

    See Also
    --------
    ``SleepEndpointDataFrame``
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_is_dtype(data.index, pd.DatetimeIndex)
        _assert_has_index_levels(data, index_levels="date", match_atleast=True, match_order=False)
        _assert_has_columns(data, columns_sets=[["sleep_onset", "wake_onset", "total_sleep_duration"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SleepEndpointDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_sleep_endpoint_dict(data: SleepEndpointDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dictionary is a ``SleepEndpointDict``.

    Parameters
    ----------
    data : dict
        data to check if it is a ``SleepEndpointDict``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SleepEndpointDict``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SleepEndpointDict``

    See Also
    --------
    ``SleepEndpointDict``
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        expected_keys = ["date", "sleep_onset", "wake_onset", "total_sleep_duration"]
        if any(col not in list(data.keys()) for col in expected_keys):
            raise ValidationError("Not all of {} are in the dictionary!".format(expected_keys))
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SleepEndpointDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_ecg_raw_dataframe(data: EcgRawDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a ``EcgRawDataFrame``.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        data to check if it is a ``EcgRawDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``EcgRawDataFrame``, ``False`` otherwise
    (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``EcgRawDataFrame``

    See Also
    --------
    ``EcgRawDataFrame``
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, columns_sets=[["ecg"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a EcgRawDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True
