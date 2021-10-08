"""A couple of helper functions that ease the use of the typical biopsykit data formats."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import (
    _assert_has_column_multiindex,
    _assert_has_column_prefix,
    _assert_has_columns,
    _assert_has_columns_any_level,
    _assert_has_index_levels,
    _assert_has_multiindex,
    _assert_is_dtype,
)
from biopsykit.utils.exceptions import ValidationError

__all__ = [
    "CodebookDataFrame",
    "MeanSeDataFrame",
    "SalivaRawDataFrame",
    "SalivaFeatureDataFrame",
    "SalivaMeanSeDataFrame",
    "SleepEndpointDataFrame",
    "SleepEndpointDict",
    "EcgRawDataFrame",
    "EcgResultDataFrame",
    "RPeakDataFrame",
    "HeartRateDataFrame",
    "AccDataFrame",
    "GyrDataFrame",
    "ImuDataFrame",
    "SleepWakeDataFrame",
    "SubjectConditionDataFrame",
    "SubjectConditionDict",
    "PhaseDict",
    "SubjectDataDict",
    "HeartRatePhaseDict",
    "HeartRateSubjectDataDict",
    "HeartRateStudyDataDict",
    "StudyDataDict",
    "MergedStudyDataDict",
    "is_subject_condition_dataframe",
    "is_subject_condition_dict",
    "is_codebook_dataframe",
    "is_mean_se_dataframe",
    "is_phase_dict",
    "is_hr_phase_dict",
    "is_subject_data_dict",
    "is_hr_subject_data_dict",
    "is_study_data_dict",
    "is_merged_study_data_dict",
    "is_saliva_raw_dataframe",
    "is_saliva_feature_dataframe",
    "is_saliva_mean_se_dataframe",
    "is_sleep_endpoint_dataframe",
    "is_sleep_endpoint_dict",
    "is_ecg_raw_dataframe",
    "is_ecg_result_dataframe",
    "is_r_peak_dataframe",
    "is_heart_rate_dataframe",
    "is_acc_dataframe",
    "is_gyr_dataframe",
    "is_imu_dataframe",
    "is_sleep_wake_dataframe",
]

# these subclasses of pd.DataFrame are needed to be added to the type aliases because otherwise, autosphinx does not
# add the docstring to the documentation of the type aliases. Additionally, they can be used internally to highlight
# which alias types are expected at which position


class _SubjectConditionDataFrame(pd.DataFrame):
    pass


class _CodebookDataFrame(pd.DataFrame):
    pass


class _MeanSeDataFrame(pd.DataFrame):
    pass


class _SalivaRawDataFrame(pd.DataFrame):
    pass


class _SalivaFeatureDataFrame(pd.DataFrame):
    pass


class _SalivaMeanSeDataFrame(pd.DataFrame):
    pass


class _EcgRawDataFrame(pd.DataFrame):
    pass


class _EcgResultDataFrame(pd.DataFrame):
    pass


class _HeartRateDataFrame(pd.DataFrame):
    pass


class _RPeakDataFrame(pd.DataFrame):
    pass


class _AccDataFrame(pd.DataFrame):
    pass


class _GyrDataFrame(pd.DataFrame):
    pass


class _ImuDataFrame(pd.DataFrame):
    pass


class _SleepWakeDataFrame(pd.DataFrame):
    pass


class _SleepEndpointDataFrame(pd.DataFrame):
    pass


SubjectConditionDataFrame = Union[_SubjectConditionDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing subject IDs and condition assignment in a standardized format.

A ``SubjectConditionDataFrame`` has an index with subject IDs named ``subject`` and a column with the condition
assignment named ``condition``.
"""

SubjectConditionDict = Dict[str, np.ndarray]
"""Dictionary containing subject IDs and condition assignment in a standardized format.

A ``SubjectConditionDict`` contains conditions as dictionary keys and a collection of subject IDs
(list, numpy array, pandas Index) as dictionary values.
"""

CodebookDataFrame = Union[_CodebookDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` representing a codebook which encodes numerical and categorical values
in a standardized format.

A ``CodebookDataFrame`` has an index level named ``variable``. The column names are the numerical values (0, 1, ...),
the dataframe entries then represent the mapping of numerical value to categorical value for the variable.
"""

MeanSeDataFrame = Union[_MeanSeDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing mean and standard error of time-series data in a standardized format.

The resulting dataframe must at least the two columns ``mean`` and ``se``. It can have additional index levels,
such as ``phase``, ``subphase`` or ``condition``.
"""

SalivaRawDataFrame = Union[_SalivaRawDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing raw saliva data in a standardized format.

Data needs to be in long-format and **must** have a :class:`pandas.MultiIndex` with index level names:

* ``subject``: subject ID; can be number or string
* ``sample``: saliva sample ID; can be number or string

Additionally, the following index levels can be added to identify saliva values, such as:

* ``condition``: subject condition during the study (e.g., "Control" vs. "Condition")
* ``day``: day ID, if saliva samples were collected over multiple days
* ``night``: night ID, if saliva samples were collected over multiple night
* ...

"""

SalivaFeatureDataFrame = Union[_SalivaFeatureDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing feature computed from saliva data in a standardized format.

The resulting dataframe must at least have a ``subject`` index level and all column names need to begin with
the saliva marker type (e.g. "cortisol"), followed by the feature name, separated by underscore '_'
Additionally, the name of the column index needs to be `saliva_feature`.
"""

SalivaMeanSeDataFrame = Union[_SalivaMeanSeDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing mean and standard error of saliva samples in a standardized format.

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
* ``bed_interval_start``: Bed Interval Start, i.e, time when participant went to bed, in absolute time
* ``bed_interval_end``: Bed Interval End, i.e, time when participant left bed, in absolute time
* ``sleep_efficiency``: Sleep Efficiency, defined as the ratio between net sleep duration and sleep duration
  in percent
* ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep, in minutes
* ``getup_latency``: Get Up Latency, i.e., time in bed after awakening until getting up, in minutes
* ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep, in minutes
* ``sleep_bouts``: List with start and end times of sleep bouts
* ``wake_bouts``: List with start and end times of wake bouts
* ``number_wake_bouts``: Total number of wake bouts

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

SleepEndpointDataFrame = Union[_SleepEndpointDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing sleep endpoints in a standardized format.

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
* ``bed_interval_start``: Bed Interval Start, i.e, time when participant went to bed, in absolute time
* ``bed_interval_end``: Bed Interval End, i.e, time when participant left bed, in absolute time
* ``sleep_efficiency``: Sleep Efficiency, defined as the ratio between net sleep duration and sleep duration
  in percent
* ``sleep_onset_latency``: Sleep Onset Latency, i.e., time in bed needed to fall asleep, in minutes
* ``getup_latency``: Get Up Latency, i.e., time in bed after awakening until getting up, in minutes
* ``wake_after_sleep_onset``: Wake After Sleep Onset (WASO), i.e., total time awake after falling asleep, in minutes
* ``number_wake_bouts``: Total number of wake bouts

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


EcgRawDataFrame = Union[_EcgRawDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing raw ECG data of `one` subject.

The dataframe is expected to have the following columns:

* ``ecg``: Raw ECG signal

"""

EcgResultDataFrame = Union[_EcgResultDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing processed ECG data of `one` subject.

The dataframe is expected to have the following columns:

* ``ECG_Raw``: Raw ECG signal
* ``ECG_Clean``: Cleaned (filtered) ECG signal
* ``ECG_Quality``: ECG signal quality indicator in the range of [0, 1]
* ``ECG_R_Peaks``: 1.0 where R peak was detected in the ECG signal, 0.0 else
* ``R_Peak_Outlier``: 1.0 when a detected R peak was classified as outlier, 0.0 else
* ``Heart_Rate``: Computed Heart rate interpolated to signal length

"""

HeartRateDataFrame = Union[_HeartRateDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing heart rate time series data of `one` subject.

The dataframe is expected to have the following columns:

* ``Heart_Rate``: Heart rate data. Can either be instantaneous heart rate or resampled heart rate

"""

RPeakDataFrame = Union[_RPeakDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing R-peak locations of `one` subject extracted from ECG data.

The dataframe is expected to have the following columns:

* ``R_Peak_Quality``: Signal quality indicator (of the raw ECG signal) in the range of [0, 1]
* ``R_Peak_Idx``: Array index of detected R peak in the raw ECG signal
* ``RR_Interval``: Interval between the current and the successive R peak in seconds
* ``R_Peak_Outlier``: 1.0 when a detected R peak was classified as outlier, 0.0 else

"""

AccDataFrame = Union[_AccDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing 3-d acceleration data.

The dataframe is expected to have one of the following column sets:

* ["acc_x", "acc_y", "acc_z"]: one level column index
* [("acc", "x"), ("acc", "y"), ("acc", "z")]: two-level column index, first level specifying the channel
  (acceleration), second level specifying the axes

"""

GyrDataFrame = Union[_GyrDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing 3-d gyroscope data.

The dataframe is expected to have one of the following column sets:

* ["gyr_x", "gyr_y", "gyr_z"]: one level column index
* [("gyr", "x"), ("gyr", "y"), ("gyr", "z")]: two-level column index, first level specifying the channel
  (gyroscope), second level specifying the axes

"""

ImuDataFrame = Union[_ImuDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing 3-d inertial measurement (IMU) (acceleration and gyroscope) data.

Hence, an ``ImuDataFrame`` must both be a ``AccDataFrame`` **and** a ``GyrDataFrame``.

The dataframe is expected to have one of the following column sets:

* ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]: one level column index
* [("acc", "x"), ("acc", "y"), ("acc", "z"), ("gyr", "x"), ("gyr", "y"), ("gyr", "z")]:
  two-level column index, first level specifying the channel (acceleration and gyroscope),
  second level specifying the axes

"""

SleepWakeDataFrame = Union[_SleepWakeDataFrame, pd.DataFrame]
""":class:`~pandas.DataFrame` containing sleep/wake predictions.

The dataframe is expected to have at least the following column(s):

* ["sleep_wake"]: sleep/wake predictions where 1 indicates sleep and 0 indicates wake

"""

PhaseDict = Dict[str, pd.DataFrame]
"""Dictionary containing general time-series data of **one single subject** split into **different phases**.

A ``PhaseDict`` is a dictionary with the following format:

{ "phase_1" : dataframe, "phase_2" : dataframe, ... }

Each ``dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* Index: :class:`pandas.DatetimeIndex` with timestamps, name of index level: ``time``

"""

HeartRatePhaseDict = Dict[str, HeartRateDataFrame]
"""Dictionary containing time-series heart rate data of **one single subject** split into **different phases**.

A ``HeartRatePhaseDict`` is a dictionary with the following format:

{ "phase_1" : hr_dataframe, "phase_2" : hr_dataframe, ... }

Each ``hr_dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* ``time`` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
* ``Heart_Rate`` Column: heart rate values

"""

SubjectDataDict = Dict[str, PhaseDict]
"""Dictionary representing time-series data from **multiple subjects** collected during a psychological protocol.

A ``SubjectDataDict`` is a nested dictionary with time-series data from multiple subjects, each containing data
from different phases. It is expected to have the level order `subject`, `phase`:

| {
|     "subject1" : { "phase_1" : dataframe, "phase_2" : dataframe, ... },
|     "subject2" : { "phase_1" : dataframe, "phase_2" : dataframe, ... },
|     ...
| }

This dictionary can, for instance, be rearranged to a :obj:`biopsykit.utils.datatype_helper.StudyDataDict`,
where the level order is reversed: `phase`, `subject`.
"""

HeartRateSubjectDataDict = Union[Dict[str, HeartRatePhaseDict], Dict[str, Dict[str, HeartRatePhaseDict]]]
"""Dictionary with time-series heart rate data from **multiple subjects** collected during a psychological protocol.

A ``HeartRateSubjectDataDict`` is a nested dictionary with time-series heart rate data from multiple subjects,
each containing data from different phases. It is expected to have the level order `subject`, `phase`:

| {
|     "subject1" : { "phase_1" : hr_dataframe, "phase_2" : hr_dataframe, ... },
|     "subject2" : { "phase_1" : hr_dataframe, "phase_2" : hr_dataframe, ... },
|     ...
| }

Each ``hr_dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* ``time`` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
* ``Heart_Rate`` Column: heart rate values

This dictionary can, for instance, be rearranged to a :obj:`~biopsykit.utils.datatype_helper.HeartRateStudyDataDict`,
where the level order is reversed: `phase`, `subject`.

"""

StudyDataDict = Dict[str, Dict[str, pd.DataFrame]]
"""Dictionary with data from **multiple phases** collected during a psychological protocol.

A ``StudyDataDict`` is a nested dictionary with time-series data from multiple phases, each phase containing data
from different subjects. It is expected to have the level order `phase`, `subject`:

| {
|     "phase_1" : { "subject1" : dataframe, "subject2" : dataframe, ... },
|     "phase_2" : { "subject1" : dataframe, "subject2" : dataframe, ... },
|     ...
| }

This dict results from rearranging a :obj:`biopsykit.utils.datatype_helper.SubjectDataDict` by calling
:func:`~biopsykit.utils.data_processing.rearrange_subject_data_dict`.
"""


HeartRateStudyDataDict = Dict[str, Dict[str, HeartRateDataFrame]]
"""Dictionary with heart rate data from **multiple phases** collected during a psychological protocol.

A ``HeartRateStudyDataDict`` is a nested dictionary with time-series heart rate data from multiple phases,
each phase containing data from different subjects. It is expected to have the level order `phase`, `subject`:

| {
|     "phase_1" : { "subject1" : hr_dataframe, "subject2" : hr_dataframe, ... },
|     "phase_2" : { "subject1" : hr_dataframe, "subject2" : hr_dataframe, ... },
|     ...
| }

Each ``hr_dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* ``time`` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
* ``Heart_Rate`` Column: heart rate values

This dict results from rearranging a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict` by calling
:func:`~biopsykit.utils.data_processing.rearrange_subject_data_dict`.
"""

MergedStudyDataDict = Dict[str, pd.DataFrame]
"""Dictionary with merged time-series data of **multiple subjects**, split into **different phases**.

A ``MergedStudyDataDict`` is a dictionary with the following format:

| {
|     "phase_1" : merged_dataframe,
|     "phase_2" : merged_dataframe,
|     ...
| }

This dict results from merging the inner dictionary into one dataframe by calling
:func:`~biopsykit.utils.data_processing.merge_study_data_dict`.

.. note::
    Merging the inner dictionaries requires that the dataframes of all subjects have same length within each phase.

Each ``merged_dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* Index: time. Name of index level: ``time``
* Columns: time series data per subject, each subject has its own column.
  Name of the column index level: ``subject``
"""


def is_subject_condition_dataframe(
    data: SubjectConditionDataFrame, raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``SubjectConditionDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SubjectConditionDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SubjectConditionDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame`
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
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDict`.

    Parameters
    ----------
    data : dict
        dict to check if it is a ``SubjectConditionDict``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SubjectConditionDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SubjectConditionDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDict`
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


def is_codebook_dataframe(data: CodebookDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.CodebookDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``CodebookDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value


    Returns
    -------
    ``True`` if ``data`` is a ``CodebookDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)


    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``CodebookDataFrame``


    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.CodebookDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_index_levels(data, index_levels="variable", match_atleast=True, match_order=False)
        if not np.issubdtype(data.columns.dtype, np.integer):
            raise ValidationError(
                "The dtypes of columns in a CodebookDataFrame are expected to be of type int, but it is {}.".format(
                    data.columns.dtype
                )
            )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a CodebookDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_mean_se_dataframe(data: MeanSeDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.MeanSeDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``MeanSeDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value


    Returns
    -------
    ``True`` if ``data`` is a ``MeanSeDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)


    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``MeanSeDataFrame``


    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.MeanSeDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        if data.columns.nlevels == 1:
            _assert_has_columns(data, [["mean", "se"]])
        else:
            _assert_has_columns_any_level(data, [["mean", "se"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MeanSeDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_hr_phase_dict(data: HeartRatePhaseDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``HeartRatePhaseDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``HeartRatePhaseDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for df in data.values():
            is_heart_rate_dataframe(df)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a HeartRatePhaseDict. "
                "The validation failed with the following error:\n\n{}\n"
                "HeartRatePhaseDicts in an old format can be converted into the new format using "
                "`biopsykit.utils.legacy_helper.legacy_convert_hr_phase_dict()`".format(str(e))
            ) from e
        return False
    return True


def is_phase_dict(data: PhaseDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a :obj:`~biopsykit.utils.datatype_helper.PhaseDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``PhaseDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``PhaseDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.PhaseDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for df in data.values():
            _assert_is_dtype(df, pd.DataFrame)
            _assert_has_multiindex(df, expected=False)
            _assert_has_column_multiindex(df, expected=False)
            _assert_has_index_levels(df, ["time"])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a PhaseDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_hr_subject_data_dict(data: HeartRateSubjectDataDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``HeartRateSubjectDataDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``HeartRateSubjectDataDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for data_dict in data.values():
            is_hr_phase_dict(data_dict)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a HeartRateSubjectDataDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_study_data_dict(data: StudyDataDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``StudyDataDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``StudyDataDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for data_dict in data.values():
            _assert_is_dtype(data_dict, dict)
            for df in data_dict.values():
                _assert_is_dtype(df, pd.DataFrame)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a StudyDataDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_subject_data_dict(data: SubjectDataDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SubjectDataDict```
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SubjectDataDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict``
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for data_dict in data.values():
            _assert_is_dtype(data_dict, dict)
            for df in data_dict.values():
                _assert_is_dtype(df, pd.DataFrame)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SubjectDataDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_merged_study_data_dict(data: MergedStudyDataDict, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether a dict is a :obj:`~biopsykit.utils.datatype_helper.MergedStudyDataDict`.

    Parameters
    ----------
    data : dict
        dict to check
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``MergedStudyDataDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``MergedStudyDataDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.MergedStudyDataDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for df in data.values():
            _assert_is_dtype(df, pd.DataFrame)
            _assert_has_multiindex(df, expected=False)
            _assert_has_column_multiindex(df, expected=False)
            _assert_has_index_levels(df, ["time"])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MergedStudyDataDict. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_saliva_raw_dataframe(
    data: SalivaRawDataFrame, saliva_type: Union[str, List[str]], raise_exception: Optional[bool] = True
) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``SalivaRawDataFrame``
    saliva_type : str or list of str
        type of saliva data (or list of saliva types) in the dataframe, e.g., "cortisol" or "amylase"
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SalivaRawDataFrame```
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SalivaRawDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
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
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``SalivaFeatureDataFrame``
    saliva_type : str or list of str
        type of saliva data in the dataframe, e.g., "cortisol" or "amylase"
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SalivaFeatureDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SalivaFeatureDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SalivaFeatureDataFrame`
        dataframe format

    """
    try:
        if saliva_type is None:
            raise ValidationError("`saliva_type` is None!")
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_index_levels(data, index_levels="subject", match_atleast=True, match_order=False)
        # _assert_has_column_levels(data, column_levels="saliva_feature", match_atleast=True, match_order=False)
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
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SalivaMeanSeDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``SalivaMeanSeDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SalivaMeanSeDataFrame```
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SalivaMeanSeDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SalivaMeanSeDataFrame``
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
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``SleepEndpointDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SleepEndpointDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SleepEndpointDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDataFrame`
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
    """Check whether dictionary is a :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDict`.

    Parameters
    ----------
    data : dict
        data to check if it is a ``SleepEndpointDict``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SleepEndpointDict``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SleepEndpointDict``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SleepEndpointDict`
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
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.EcgRawDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``EcgRawDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``EcgRawDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``EcgRawDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.EcgRawDataFrame`
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


def is_ecg_result_dataframe(data: EcgRawDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``EcgResultDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``EcgResultDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``EcgResultDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            columns_sets=[
                ["ECG_Raw", "ECG_Clean", "ECG_Quality", "ECG_R_Peaks", "R_Peak_Outlier"],
                ["ECG_Raw", "ECG_Clean", "ECG_Quality", "ECG_R_Peaks", "R_Peak_Outlier", "Heart_Rate"],
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a EcgResultDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_heart_rate_dataframe(data: HeartRateDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.HeartRateDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``HeartRateDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``HeartRateDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``HeartRateDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.HeartRateDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, columns_sets=[["Heart_Rate"]])
        _assert_has_multiindex(data, expected=False)
        _assert_has_column_multiindex(data, expected=False)
        _assert_has_index_levels(data, ["time"])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a HeartRateDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_r_peak_dataframe(data: EcgRawDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.RPeakDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``RPeakDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``RPeakDataFrame```
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``RPeakDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.RPeakDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            columns_sets=[
                ["R_Peak_Idx", "RR_Interval"],
                ["R_Peak_Quality", "R_Peak_Idx", "RR_Interval"],
                ["R_Peak_Quality", "R_Peak_Idx", "RR_Interval", "R_Peak_Outlier"],
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a RPeakDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_acc_dataframe(data: AccDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.AccDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``AccDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``AccDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``AccDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.AccDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            columns_sets=[
                ["acc_x", "acc_y", "acc_z"],
                [("acc", "x"), ("acc", "y"), ("acc", "z")],
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a AccDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_gyr_dataframe(data: GyrDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.GyrDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``GyrDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``GyrDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``GyrDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.GyrDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            columns_sets=[
                ["gyr_x", "gyr_y", "gyr_z"],
                [("gyr", "x"), ("gyr", "y"), ("gyr", "z")],
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a GyrDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_imu_dataframe(data: GyrDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.ImuDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``ImuDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``ImuDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``ImuDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.ImuDataFrame`
        dataframe format

    """
    try:
        is_acc_dataframe(data, raise_exception=True)
        is_gyr_dataframe(data, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a ImuDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_sleep_wake_dataframe(data: SleepWakeDataFrame, raise_exception: Optional[bool] = True) -> Optional[bool]:
    """Check whether dataframe is a :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``SleepWakeDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``SleepWakeDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``SleepWakeDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, [["sleep_wake"]])
        if not all(data["sleep_wake"].between(0, 1, inclusive=True)):
            raise ValidationError(
                "Invalid values for sleep/wake prediction! Sleep/wake scores are expected to be "
                "in the interval [0, 1]."
            )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SleepWakeDataFrame. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True
