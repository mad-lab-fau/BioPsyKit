"""A couple of helper functions that ease the use of the typical biopsykit data formats."""

from typing import Any

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
    _assert_sample_columns_int,
)
from biopsykit.utils.exceptions import ValidationError

__all__ = [
    "ECG_RESULT_DATAFRAME_COLUMNS",
    "HEART_RATE_DATAFRAME_COLUMNS",
    "R_PEAK_DATAFRAME_COLUMNS",
    "Acc1dDataFrame",
    "Acc3dDataFrame",
    "BPointDataFrame",
    "BiomarkerRawDataFrame",
    "CPointDataFrame",
    "CodebookDataFrame",
    "EcgRawDataFrame",
    "EcgResultDataFrame",
    "Gyr1dDataFrame",
    "Gyr3dDataFrame",
    "HeartRateDataFrame",
    "HeartRatePhaseDict",
    "HeartRateStudyDataDict",
    "HeartRateSubjectDataDict",
    "HeartbeatSegmentationDataFrame",
    "IcgRawDataFrame",
    "ImuDataFrame",
    "MeanSeDataFrame",
    "MergedStudyDataDict",
    "PepResultDataFrame",
    "PhaseDict",
    "QPeakDataFrame",
    "RPeakDataFrame",
    "SalivaFeatureDataFrame",
    "SalivaMeanSeDataFrame",
    "SalivaRawDataFrame",
    "SleepEndpointDataFrame",
    "SleepEndpointDict",
    "SleepWakeDataFrame",
    "StudyDataDict",
    "SubjectConditionDataFrame",
    "SubjectConditionDict",
    "SubjectDataDict",
    "is_acc1d_dataframe",
    "is_acc3d_dataframe",
    "is_b_point_dataframe",
    "is_biomarker_raw_dataframe",
    "is_c_point_dataframe",
    "is_codebook_dataframe",
    "is_ecg_raw_dataframe",
    "is_ecg_result_dataframe",
    "is_gyr1d_dataframe",
    "is_gyr3d_dataframe",
    "is_heart_rate_dataframe",
    "is_heartbeat_segmentation_dataframe",
    "is_hr_phase_dict",
    "is_hr_subject_data_dict",
    "is_icg_raw_dataframe",
    "is_imu_dataframe",
    "is_mean_se_dataframe",
    "is_merged_study_data_dict",
    "is_pep_result_dataframe",
    "is_phase_dict",
    "is_q_peak_dataframe",
    "is_r_peak_dataframe",
    "is_saliva_feature_dataframe",
    "is_saliva_mean_se_dataframe",
    "is_saliva_raw_dataframe",
    "is_sleep_endpoint_dataframe",
    "is_sleep_endpoint_dict",
    "is_sleep_wake_dataframe",
    "is_study_data_dict",
    "is_subject_condition_dataframe",
    "is_subject_condition_dict",
    "is_subject_data_dict",
]

ECG_RESULT_DATAFRAME_COLUMNS = ["ECG_Raw", "ECG_Clean", "ECG_Quality", "ECG_R_Peaks", "R_Peak_Outlier"]
HEART_RATE_DATAFRAME_COLUMNS = ["Heart_Rate"]
R_PEAK_DATAFRAME_COLUMNS = ["R_Peak_Quality", "R_Peak_Idx", "RR_Interval", "R_Peak_Outlier"]

PEP_RESULT_DATAFRAME_COLUMNS = [
    "heartbeat_start_sample",
    "heartbeat_end_sample",
    "r_peak_sample",
    "q_peak_sample",
    "b_point_sample",
    "pep_sample",
    "pep_ms",
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


class _BiomarkerRawDataFrame(pd.DataFrame):
    pass


class _SalivaRawDataFrame(pd.DataFrame):
    pass


class _SalivaFeatureDataFrame(pd.DataFrame):
    pass


class _SalivaMeanSeDataFrame(pd.DataFrame):
    pass


class _IcgRawDataFrame(pd.DataFrame):
    pass


class _EcgRawDataFrame(pd.DataFrame):
    pass


class _EcgResultDataFrame(pd.DataFrame):
    pass


class _HeartRateDataFrame(pd.DataFrame):
    pass


class _RPeakDataFrame(pd.DataFrame):
    pass


class _Acc1dDataFrame(pd.DataFrame):
    pass


class _Acc3dDataFrame(pd.DataFrame):
    pass


class _Gyr1dDataFrame(pd.DataFrame):
    pass


class _Gyr3dDataFrame(pd.DataFrame):
    pass


class _ImuDataFrame(pd.DataFrame):
    pass


class _SleepWakeDataFrame(pd.DataFrame):
    pass


class _SleepEndpointDataFrame(pd.DataFrame):
    pass


class _HeartbeatSegmentationDataFrame(pd.DataFrame):
    pass


class _QPeakDataFrame(pd.DataFrame):
    pass


class _BPointDataFrame(pd.DataFrame):
    pass


class _CPointDataFrame(pd.DataFrame):
    pass


class _PepResultDataFrame(pd.DataFrame):
    pass


SubjectConditionDataFrame = _SubjectConditionDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing subject IDs and condition assignment in a standardized format.

A ``SubjectConditionDataFrame`` has an index with subject IDs named ``subject`` and a column with the condition
assignment named ``condition``.
"""

SubjectConditionDict = dict[str, np.ndarray]
"""Dictionary containing subject IDs and condition assignment in a standardized format.

A ``SubjectConditionDict`` contains conditions as dictionary keys and a collection of subject IDs
(list, numpy array, pandas Index) as dictionary values.
"""

CodebookDataFrame = _CodebookDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` representing a codebook which encodes numerical and categorical values
in a standardized format.

A ``CodebookDataFrame`` has an index level named ``variable``. The column names are the numerical values (0, 1, ...),
the dataframe entries then represent the mapping of numerical value to categorical value for the variable.
"""

MeanSeDataFrame = _MeanSeDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing mean and standard error of time-series data in a standardized format.

The resulting dataframe must at least the two columns ``mean`` and ``se``. It can have additional index levels,
such as ``phase``, ``subphase`` or ``condition``.
"""

BiomarkerRawDataFrame = _BiomarkerRawDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing raw biomarker data in a standardized format.

Data needs to be in long-format and **must** have a :class:`pandas.MultiIndex` with index level names:

* ``subject``: subject ID; can be number or string
* ``sample``: saliva sample ID; can be number or string

Additionally, the following index levels can be added to identify saliva values, such as:

* ``condition``: subject condition during the study (e.g., "Control" vs. "Condition")
* ``day``: day ID, if saliva samples were collected over multiple days
* ``night``: night ID, if saliva samples were collected over multiple night
* ...

"""

SalivaRawDataFrame = _BiomarkerRawDataFrame | pd.DataFrame
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

SalivaFeatureDataFrame = _SalivaFeatureDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing feature computed from saliva data in a standardized format.

The resulting dataframe must at least have a ``subject`` index level and all column names need to begin with
the saliva marker type (e.g. "cortisol"), followed by the feature name, separated by underscore '_'
Additionally, the name of the column index needs to be `saliva_feature`.
"""

SalivaMeanSeDataFrame = _SalivaMeanSeDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing mean and standard error of saliva samples in a standardized format.

The resulting dataframe must at least have a ``sample`` index level and the two columns ``mean`` and ``se``.
It can have additional index levels, such as ``condition`` or ``time``.
"""

SleepEndpointDict = dict[str, Any]
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

SleepEndpointDataFrame = _SleepEndpointDataFrame | pd.DataFrame
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

IcgRawDataFrame = _IcgRawDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing raw ICG data of `one` subject.

The dataframe is expected to have one of the following columns:

* ``icg``: Raw ICG signal
* ``icg_der``: Derivative of the ICG signal

"""


EcgRawDataFrame = _EcgRawDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing raw ECG data of `one` subject.

The dataframe is expected to have the following columns:

* ``ecg``: Raw ECG signal

"""

EcgResultDataFrame = _EcgResultDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing processed ECG data of `one` subject.

The dataframe is expected to have the following columns:

* ``ECG_Raw``: Raw ECG signal
* ``ECG_Clean``: Cleaned (filtered) ECG signal
* ``ECG_Quality``: ECG signal quality indicator in the range of [0, 1]
* ``ECG_R_Peaks``: 1.0 where R peak was detected in the ECG signal, 0.0 else
* ``R_Peak_Outlier``: 1.0 when a detected R peak was classified as outlier, 0.0 else
* ``Heart_Rate``: Computed Heart rate interpolated to signal length

"""

HeartRateDataFrame = _HeartRateDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing heart rate time series data of `one` subject.

The dataframe is expected to have the following columns:

* ``Heart_Rate``: Heart rate data. Can either be instantaneous heart rate or resampled heart rate

"""

RPeakDataFrame = _RPeakDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing R-peak locations of `one` subject extracted from ECG data.

The dataframe is expected to have the following columns:

* ``R_Peak_Quality``: Signal quality indicator (of the raw ECG signal) in the range of [0, 1]
* ``R_Peak_Idx``: Array index of detected R peak in the raw ECG signal
* ``RR_Interval``: Interval between the current and the successive R peak in seconds
* ``R_Peak_Outlier``: 1.0 when a detected R peak was classified as outlier, 0.0 else

"""

Acc1dDataFrame = _Acc1dDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing 1-d acceleration data.

The dataframe is expected to have one of the following column sets:

* ["acc"]: one level column index
* ["acc_norm"]: one level column index

"""

Acc3dDataFrame = _Acc3dDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing 3-d acceleration data.

The dataframe is expected to have one of the following column sets:

* ["acc_x", "acc_y", "acc_z"]: one level column index
* [("acc", "x"), ("acc", "y"), ("acc", "z")]: two-level column index, first level specifying the channel
  (acceleration), second level specifying the axes

"""

Gyr1dDataFrame = _Gyr1dDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing 1-d gyroscope data.

The dataframe is expected to have one of the following column sets:

* ["gyr"]: one level column index
* ["gyr_norm"]: one level column index

"""

Gyr3dDataFrame = _Gyr3dDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing 3-d gyroscope data.

The dataframe is expected to have one of the following column sets:

* ["gyr_x", "gyr_y", "gyr_z"]: one level column index
* [("gyr", "x"), ("gyr", "y"), ("gyr", "z")]: two-level column index, first level specifying the channel
  (gyroscope), second level specifying the axes

"""

ImuDataFrame = _ImuDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing 6-d inertial measurement (IMU) (acceleration and gyroscope) data.

Hence, an ``ImuDataFrame`` must both be a ``AccDataFrame`` **and** a ``GyrDataFrame``.

The dataframe is expected to have one of the following column sets:

* ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]: one level column index
* [("acc", "x"), ("acc", "y"), ("acc", "z"), ("gyr", "x"), ("gyr", "y"), ("gyr", "z")]:
  two-level column index, first level specifying the channel (acceleration and gyroscope),
  second level specifying the axes

"""

SleepWakeDataFrame = _SleepWakeDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing sleep/wake predictions.

The dataframe is expected to have at least the following column(s):

* ["sleep_wake"]: sleep/wake predictions where 1 indicates sleep and 0 indicates wake

"""

HeartbeatSegmentationDataFrame = _HeartbeatSegmentationDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing results of heartbeat segmentation.

The dataframe is expected to have *at least* the following columns:

* ``start_sample``: Start sample of segmented heartbeat
* ``end_sample``: End sample of segmented heartbeat
* ``r_peak_sample``: R-peak sample of segmented heartbeat

"""

QPeakDataFrame = _QPeakDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing Q-peak locations extracted from ECG data.

The dataframe is expected to have *at least* the following columns:

* ``q_peak_sample``: The sample index of the Q-peak in the ECG signal

Optionally, the dataframe can contain additional columns, such as:

* ``nan_reason``: Reason why the Q-peak was set to NaN (e.g., "r_peak_nan", "no_zero_crossing")

"""

BPointDataFrame = _BPointDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing B-Point locations extracted from ICG data.

The dataframe is expected to have *at least* the following columns:

* ``b_point_sample``: The sample index of the B-point in the ICG signal

Optionally, the dataframe can contain additional columns, such as:

* ``nan_reason``: Reason why the B-point was set to NaN (e.g., "c_point_nan", "no_zero_crossing")

"""

CPointDataFrame = _CPointDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing C-Point locations extracted from ICG data.

The dataframe is expected to have *at least* the following columns:

* ``c_point_sample``: The sample index of the C-point in the ICG signal

Optionally, the dataframe can contain additional columns, such as:

* ``nan_reason``: Reason why the C-point was set to NaN (e.g., "no_local_maxima")

"""

PepResultDataFrame = _PepResultDataFrame | pd.DataFrame
""":class:`~pandas.DataFrame` containing results of PEP extraction.

The dataframe is expected to have *at least* the following columns:

* ``heartbeat_start_sample``: Start sample of segmented heartbeat
* ``heartbeat_end_sample``: End sample of segmented heartbeat
* ``r_peak_sample``: R-peak sample of segmented heartbeat
* ``q_peak_sample``: Q-peak sample of segmented heartbeat
* ``b_point_sample``: B-point sample of segmented heartbeat
* ``pep_sample``: Pre-ejection period (PEP) in samples
* ``pep_ms``: Pre-ejection period (PEP) in milliseconds



Additionally, the dataframe can contain the following columns:

* ``rr_interval_sample``: RR interval between the previous and the current heartbeat in samples
* ``rr_interval_ms``: RR interval between the previous and the current heartbeat in milliseconds
* ``heart_rate_bpm``: Heart rate in beats per minute, derived from RR interval
* ``nan_reason``: Reason why the PEP was set to NaN (e.g., "r_peak_nan", "no_zero_crossing")

"""


PhaseDict = dict[str, pd.DataFrame]
"""Dictionary containing general time-series data of **one single subject** split into **different phases**.

A ``PhaseDict`` is a dictionary with the following format:

{ "phase_1" : dataframe, "phase_2" : dataframe, ... }

Each ``dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* Index: :class:`pandas.DatetimeIndex` with timestamps, name of index level: ``time``

"""

HeartRatePhaseDict = dict[str, HeartRateDataFrame]
"""Dictionary containing time-series heart rate data of **one single subject** split into **different phases**.

A ``HeartRatePhaseDict`` is a dictionary with the following format:

{ "phase_1" : hr_dataframe, "phase_2" : hr_dataframe, ... }

Each ``hr_dataframe`` is a :class:`~pandas.DataFrame` with the following format:

* ``time`` Index: :class:`pandas.DatetimeIndex` with heart rate sample timestamps
* ``Heart_Rate`` Column: heart rate values

"""

SubjectDataDict = dict[str, PhaseDict]
"""Dictionary representing time-series data from **multiple subjects** collected during a psychological protocol.

A ``SubjectDataDict`` is a nested dictionary with time-series data from multiple subjects, each containing data
from different phases. It is expected to have the level order `subject`, `phase`:

| {
|     "subject1" : { "phase_1" : dataframe, "phase_2" : dataframe, ... },
|     "subject2" : { "phase_1" : dataframe, "phase_2" : dataframe, ... },
|     ...
| }

This dictionary can, for instance, be rearranged to a :obj:`biopsykit.utils.dtypes.StudyDataDict`,
where the level order is reversed: `phase`, `subject`.
"""

HeartRateSubjectDataDict = dict[str, HeartRatePhaseDict] | dict[str, dict[str, HeartRatePhaseDict]]
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

This dictionary can, for instance, be rearranged to a :obj:`~biopsykit.utils.dtypes.HeartRateStudyDataDict`,
where the level order is reversed: `phase`, `subject`.

"""

StudyDataDict = dict[str, dict[str, pd.DataFrame]]
"""Dictionary with data from **multiple phases** collected during a psychological protocol.

A ``StudyDataDict`` is a nested dictionary with time-series data from multiple phases, each phase containing data
from different subjects. It is expected to have the level order `phase`, `subject`:

| {
|     "phase_1" : { "subject1" : dataframe, "subject2" : dataframe, ... },
|     "phase_2" : { "subject1" : dataframe, "subject2" : dataframe, ... },
|     ...
| }

This dict results from rearranging a :obj:`biopsykit.utils.dtypes.SubjectDataDict` by calling
:func:`~biopsykit.utils.data_processing.rearrange_subject_data_dict`.
"""


HeartRateStudyDataDict = dict[str, dict[str, HeartRateDataFrame]]
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

This dict results from rearranging a :obj:`~biopsykit.utils.dtypes.HeartRateSubjectDataDict` by calling
:func:`~biopsykit.utils.data_processing.rearrange_subject_data_dict`.
"""

MergedStudyDataDict = dict[str, pd.DataFrame]
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


def is_subject_condition_dataframe(data: SubjectConditionDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SubjectConditionDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SubjectConditionDataFrame`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_subject_condition_dict(data: SubjectConditionDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SubjectConditionDict`.

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
    :obj:`~biopsykit.utils.dtypes.SubjectConditionDict`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_codebook_dataframe(data: CodebookDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.CodebookDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.CodebookDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_index_levels(data, index_levels="variable", match_atleast=True, match_order=False)
        if not np.issubdtype(data.columns.dtype, np.integer):
            raise ValidationError(
                f"The dtypes of columns in a CodebookDataFrame are expected to be of type int, "
                f"but it is {data.columns.dtype}."
            )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a CodebookDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_mean_se_dataframe(data: MeanSeDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.MeanSeDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.MeanSeDataFrame`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_hr_phase_dict(data: HeartRatePhaseDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether a dict is a :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict`.

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
    :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict`
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
                f"The validation failed with the following error:\n\n{e!s}\n"
                "HeartRatePhaseDicts in an old format can be converted into the new format using "
                "`biopsykit.utils.legacy_helper.legacy_convert_hr_phase_dict()`"
            ) from e
        return False
    return True


def is_phase_dict(data: PhaseDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether a dict is a :obj:`~biopsykit.utils.dtypes.PhaseDict`.

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
    :obj:`~biopsykit.utils.dtypes.PhaseDict`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_hr_subject_data_dict(data: HeartRateSubjectDataDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether a dict is a :obj:`~biopsykit.utils.dtypes.HeartRateSubjectDataDict`.

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
    :obj:`~biopsykit.utils.dtypes.HeartRateSubjectDataDict`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_study_data_dict(data: StudyDataDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether a dict is a :obj:`~biopsykit.utils.dtypes.StudyDataDict`.

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
    :obj:`~biopsykit.utils.dtypes.StudyDataDict`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_subject_data_dict(data: SubjectDataDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether a dict is a :obj:`~biopsykit.utils.dtypes.SubjectDataDict`.

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
    :obj:`~biopsykit.utils.dtypes.SubjectDataDict``
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_merged_study_data_dict(data: MergedStudyDataDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether a dict is a :obj:`~biopsykit.utils.dtypes.MergedStudyDataDict`.

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
    :obj:`~biopsykit.utils.dtypes.MergedStudyDataDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        for df in data.values():
            _assert_is_dtype(df, pd.DataFrame)
            _assert_has_multiindex(df, expected=False)
            _assert_has_index_levels(df, ["time"])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MergedStudyDataDict. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_biomarker_raw_dataframe(
    data: BiomarkerRawDataFrame, biomarker_type: str | list[str], raise_exception: bool | None = True
) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SalivaRawDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SalivaRawDataFrame`
        dataframe format

    """
    try:
        if biomarker_type is None:
            raise ValidationError("`saliva_type` is None!")
        if isinstance(biomarker_type, str):
            biomarker_type = [biomarker_type]
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_multiindex(data, nlevels=2, nlevels_atleast=True)
        _assert_has_index_levels(data, index_levels=["subject", "sample"], match_atleast=True, match_order=False)
        _assert_has_columns(data, [biomarker_type, [*biomarker_type, "time"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a BiomarkerRawDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_saliva_raw_dataframe(
    data: SalivaRawDataFrame, saliva_type: str | list[str], raise_exception: bool | None = True
) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SalivaRawDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SalivaRawDataFrame`
        dataframe format

    """
    return is_biomarker_raw_dataframe(data, saliva_type, raise_exception)


def is_saliva_feature_dataframe(
    data: SalivaFeatureDataFrame, saliva_type: str, raise_exception: bool | None = True
) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SalivaFeatureDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SalivaFeatureDataFrame`
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_saliva_mean_se_dataframe(data: SalivaFeatureDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SalivaMeanSeDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SalivaMeanSeDataFrame``
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_sleep_endpoint_dataframe(data: SleepEndpointDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SleepEndpointDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SleepEndpointDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_is_dtype(data.index, pd.DatetimeIndex)
        _assert_has_index_levels(data, index_levels="date", match_atleast=True, match_order=False)
        _assert_has_columns(data, column_sets=[["sleep_onset", "wake_onset", "total_sleep_duration"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SleepEndpointDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_sleep_endpoint_dict(data: SleepEndpointDict, raise_exception: bool | None = True) -> bool | None:
    """Check whether dictionary is a :obj:`~biopsykit.utils.dtypes.SleepEndpointDict`.

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
    :obj:`~biopsykit.utils.dtypes.SleepEndpointDict`
        dictionary format

    """
    try:
        _assert_is_dtype(data, dict)
        expected_keys = ["date", "sleep_onset", "wake_onset", "total_sleep_duration"]
        if any(col not in list(data.keys()) for col in expected_keys):
            raise ValidationError(f"Not all of {expected_keys} are in the dictionary!")
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SleepEndpointDict. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_icg_raw_dataframe(data: IcgRawDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.IcgRawDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``IcgRawDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``IcgRawDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``IcgRawDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.IcgRawDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, column_sets=[["icg_der"], ["icg"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a IcgRawDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_ecg_raw_dataframe(data: EcgRawDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.EcgRawDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.EcgRawDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, column_sets=[["ecg"]])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a EcgRawDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_ecg_result_dataframe(data: EcgRawDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.EcgResultDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.EcgResultDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[
                ECG_RESULT_DATAFRAME_COLUMNS,
                ECG_RESULT_DATAFRAME_COLUMNS + HEART_RATE_DATAFRAME_COLUMNS,
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a EcgResultDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_heart_rate_dataframe(data: HeartRateDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.HeartRateDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.HeartRateDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, column_sets=[HEART_RATE_DATAFRAME_COLUMNS])
        _assert_has_multiindex(data, expected=False)
        _assert_has_column_multiindex(data, expected=False)
        _assert_has_index_levels(data, ["time"])
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a HeartRateDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_r_peak_dataframe(data: EcgRawDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.RPeakDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.RPeakDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[
                ["R_Peak_Idx", "RR_Interval"],
                ["R_Peak_Quality", "R_Peak_Idx", "RR_Interval"],
                R_PEAK_DATAFRAME_COLUMNS,
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a RPeakDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_acc1d_dataframe(data: Acc3dDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.Acc1dDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``Acc1dDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``Acc1dDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``Acc1dDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.Acc1dDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[["acc"], ["acc_norm"]],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a Acc1dDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_acc3d_dataframe(data: Acc3dDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.Acc3dDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``Acc3dDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``Acc3dDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``Acc3dDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.Acc3dDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[
                ["acc_x", "acc_y", "acc_z"],
                [("acc", "x"), ("acc", "y"), ("acc", "z")],
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a Acc3dDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_gyr1d_dataframe(data: Gyr3dDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.Gyr1dDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``Gyr1dDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``Gyr1dDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``Gyr1dDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.Gyr1dDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[["gyr"], ["gyr_norm"]],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a Gyr1dDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_gyr3d_dataframe(data: Gyr3dDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.Gyr3dDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``Gyr3dDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``Gyr3dDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``Gyr3dDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.Gyr3dDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[
                ["gyr_x", "gyr_y", "gyr_z"],
                [("gyr", "x"), ("gyr", "y"), ("gyr", "z")],
            ],
        )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a Gyr3dDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_imu_dataframe(data: Gyr3dDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.ImuDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.ImuDataFrame`
        dataframe format

    """
    try:
        is_acc3d_dataframe(data, raise_exception=True)
        is_gyr3d_dataframe(data, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a ImuDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_sleep_wake_dataframe(data: SleepWakeDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.SleepWakeDataFrame`.

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
    :obj:`~biopsykit.utils.dtypes.SleepWakeDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, [["sleep_wake"]])
        if not all(data["sleep_wake"].between(0, 1, inclusive=True)):
            raise ValidationError(
                "Invalid values for sleep/wake prediction! Sleep/wake scores are expected to be in the interval [0, 1]."
            )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SleepWakeDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_heartbeat_segmentation_dataframe(
    data: HeartbeatSegmentationDataFrame, raise_exception: bool | None = True
) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``HeartbeatSegmentationDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``HeartbeatSegmentationDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``HeartbeatSegmentationDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.HeartbeatSegmentationDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, [["start_sample", "end_sample", "r_peak_sample"]])
        # assert that columns with "_sample" in the end are of type int
        _assert_sample_columns_int(data)
        _assert_has_index_levels(data, "heartbeat_id", match_atleast=True, match_order=False)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a HeartbeatSegmentationDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_c_point_dataframe(data: CPointDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.CPointDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``CPointDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``CPointDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``CPointDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.CPointDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, [["c_point_sample"], ["c_point_sample", "nan_reason"]])
        # assert that columns with "_sample" in the end are of type int
        _assert_sample_columns_int(data)
        _assert_has_index_levels(data, "heartbeat_id", match_atleast=True, match_order=False)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a CPointDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_b_point_dataframe(data: BPointDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.BPointDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``BPointDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``BPointDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``BPointDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.BPointDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, [["b_point_sample"], ["b_point_sample", "nan_reason"]])
        # assert that columns with "_sample" in the end are of type int
        _assert_sample_columns_int(data)
        _assert_has_index_levels(data, "heartbeat_id", match_atleast=True, match_order=False)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a BPointDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_q_peak_dataframe(data: QPeakDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.QPeakDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``QPeakDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``QPeakDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``QPeakDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.QPeakDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(data, [["q_peak_sample"], ["q_peak_sample", "nan_reason"]])
        # assert that columns with "_sample" in the end are of type int
        _assert_sample_columns_int(data)
        _assert_has_index_levels(data, "heartbeat_id", match_atleast=True, match_order=False)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a QPeakDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_pep_result_dataframe(data: PepResultDataFrame, raise_exception: bool | None = True) -> bool | None:
    """Check whether dataframe is a :obj:`~biopsykit.utils.dtypes.PepResultDataFrame`.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to check if it is a ``PepResultDataFrame``
    raise_exception : bool, optional
        whether to raise an exception or return a bool value

    Returns
    -------
    ``True`` if ``data`` is a ``PepResultDataFrame``
    ``False`` otherwise (if ``raise_exception`` is ``False``)

    Raises
    ------
    ValidationError
        if ``raise_exception`` is ``True`` and ``data`` is not a ``PepResultDataFrame``

    See Also
    --------
    :obj:`~biopsykit.utils.dtypes.PepResultDataFrame`
        dataframe format

    """
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_columns(
            data,
            column_sets=[
                PEP_RESULT_DATAFRAME_COLUMNS,
                [*PEP_RESULT_DATAFRAME_COLUMNS, "rr_interval_sample", "rr_interval_ms", "heart_rate_bpm", "nan_reason"],
            ],
        )
        _assert_sample_columns_int(data)
        _assert_has_index_levels(data, "heartbeat_id", match_atleast=True, match_order=False)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a PepResultDataFrame. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True
