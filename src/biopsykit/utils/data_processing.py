"""Module providing various functions for processing more complex structured data (e.g., collected during a study)."""
import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import interpolate

from biopsykit.utils._datatype_validation_helper import (
    _assert_dataframes_same_length,
    _assert_has_index_levels,
    _assert_has_multiindex,
    _assert_is_dtype,
)
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.dtypes import (
    MeanSeDataFrame,
    MergedStudyDataDict,
    StudyDataDict,
    SubjectConditionDataFrame,
    SubjectConditionDict,
    SubjectDataDict,
    _MeanSeDataFrame,
    is_mean_se_dataframe,
    is_merged_study_data_dict,
    is_study_data_dict,
    is_subject_condition_dataframe,
    is_subject_condition_dict,
    is_subject_data_dict,
)
from biopsykit.utils.functions import se


def _split_data_series(data: pd.DataFrame, time_intervals: pd.Series, include_start: bool) -> dict[str, pd.DataFrame]:
    if time_intervals.index.nlevels > 1:
        # multi-index series => second level contains start/end times of phases
        time_intervals = time_intervals.unstack().T
        time_intervals = {key: tuple(value.values()) for key, value in time_intervals.to_dict().items()}
    else:
        if include_start:
            time_intervals["Start"] = data.index[0].to_pydatetime().time()
        # time_intervals.sort_values(inplace=True)
        time_intervals = {
            name: (start, end)
            for name, start, end in zip(time_intervals.index, time_intervals[:-1], time_intervals[1:])
        }
    return time_intervals


def split_data(
    data: pd.DataFrame,
    time_intervals: Union[pd.DataFrame, pd.Series, dict[str, Sequence[str]]],
    include_start: Optional[bool] = False,
) -> dict[str, pd.DataFrame]:
    """Split data into different phases based on time intervals.

    The start and end times of the phases are prodivded via the ``time_intervals`` parameter and can either be a
    :class:`~pandas.Series`, 1 row of a :class:`~pandas.DataFrame`, or a dictionary with start and end times per phase.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        data to be split
    time_intervals : dict or :class:`~pandas.Series` or :class:`~pandas.DataFrame`
        time intervals indicating where the data should be split. This can be:

        * :class:`~pandas.Series` object or 1 row of a :class:`~pandas.DataFrame` with `start` times of each phase.
          The phase names are then derived from the `index` names in case of a :class:`~pandas.Series` or from the
          `columns` names in case of a :class:`~pandas.DataFrame`.
        * dictionary with phase names (keys) and tuples with start and end times of the phase (values)

    include_start: bool, optional
        ``True`` to include data from the beginning of the recording to the start of the first phase as the
        first phase (this phase will be named "Start"), ``False`` to discard this data. Default: ``False``


    Returns
    -------
    dict
        dictionary with data split into different phases


    Examples
    --------
    >>> from biopsykit.utils.data_processing import split_data
    >>> # read pandas dataframe from csv file and split data based on time interval dictionary
    >>> data = pd.read_csv("path-to-file.csv")
    >>> # Example 1: define time intervals (start and end) of the different recording phases as dictionary
    >>> time_intervals = {"Part1": ("09:00", "09:30"), "Part2": ("09:30", "09:45"), "Part3": ("09:45", "10:00")}
    >>> data_dict = split_data(data=data, time_intervals=time_intervals)
    >>> # Example 2: define time intervals as pandas Series. Here, only start times of the are required, it is assumed
    >>> # that the phases are back to back
    >>> time_intervals = pd.Series(data=["09:00", "09:30", "09:45", "10:00"], index=["Part1", "Part2", "Part3", "End"])
    >>> data_dict = split_data(data=data, time_intervals=time_intervals)
    >>>
    >>> # Example: Get Part 2 of data_dict
    >>> print(data_dict['Part2'])

    """
    _assert_is_dtype(time_intervals, (pd.DataFrame, pd.Series, dict))

    if isinstance(time_intervals, pd.DataFrame):
        if len(time_intervals) > 1:
            raise ValueError("Only dataframes with 1 row allowed!")
        time_intervals = time_intervals.iloc[0]

    if isinstance(time_intervals, pd.Series):
        time_intervals = _split_data_series(data, time_intervals, include_start)
    elif include_start:
        time_intervals["Start"] = (
            data.index[0].to_pydatetime().time(),
            next(iter(time_intervals.values())),
        )

    data_dict = {name: data.between_time(*start_end) for name, start_end in time_intervals.items()}
    data_dict = {name: data for name, data in data_dict.items() if not data.empty}
    return data_dict


def exclude_subjects(
    excluded_subjects: Union[Sequence[str], Sequence[int]], index_name: Optional[str] = "subject", **kwargs
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Exclude subjects from dataframes.

    This function can be used to exclude subject IDs for later analysis from different kinds of dataframes, such as:

    * dataframes with subject condition information
      (:obj:`~biopsykit.utils.dtypes.SubjectConditionDataFrame`)
    * dataframes with time log information
    * dataframes with (processed) data (e.g., :obj:`biopsykit.utils.dtypes.SalivaRawDataFrame` or
      :obj:`~biopsykit.utils.dtypes.MeanSeDataFrame`)

    All dataframes can be supplied at once via ``**kwargs``.

    Parameters
    ----------
    excluded_subjects : list of str or int
        list with subjects IDs to be excluded
    index_name : str, optional
        name of dataframe index level with subject IDs. Default: "subject"
    **kwargs :
        data to be cleaned as key-value pairs

    Returns
    -------
    :class:`~pandas.DataFrame` or dict of such
        dictionary with cleaned versions of the dataframes passed to the function via ``**kwargs``
        or dataframe if function was only called with one single dataframe

    """
    cleaned_data: dict[str, pd.DataFrame] = {}

    for key, data in kwargs.items():
        _assert_is_dtype(data, pd.DataFrame)
        if index_name in data.index.names:
            level_values = data.index.get_level_values(index_name)
            if (level_values.dtype == object and all(isinstance(s, str) for s in excluded_subjects)) or (
                level_values.dtype == int and all(isinstance(s, int) for s in excluded_subjects)
            ):
                cleaned_data[key] = _exclude_single_subject(data, excluded_subjects, index_name, key)
            else:
                raise ValueError(f"{key}: dtypes of index and subject ids to be excluded do not match!")
        else:
            raise ValueError(f"No '{index_name}' level in index!")
    if len(cleaned_data) == 1:
        cleaned_data = next(iter(cleaned_data.values()))
    return cleaned_data


def _exclude_single_subject(
    data: pd.DataFrame, excluded_subjects: Union[Sequence[str], Sequence[int]], index_name: str, dataset_name: str
) -> pd.DataFrame:
    # dataframe index and subjects are both strings or both integers
    try:
        if isinstance(data.index, pd.MultiIndex):
            # MultiIndex => specify index level
            return data.drop(index=excluded_subjects, level=index_name)
        # Regular Index
        return data.drop(index=excluded_subjects)
    except KeyError:
        warnings.warn(f"Not all subjects of {excluded_subjects} exist in '{dataset_name}'!")
        return data


def normalize_to_phase(subject_data_dict: SubjectDataDict, phase: Union[str, pd.DataFrame]) -> SubjectDataDict:
    """Normalize time series data per subject to the phase specified by ``normalize_to``.

    The result is the relative change (of, for example, heart rate) compared to the mean value in ``phase``.

    Parameters
    ----------
    subject_data_dict : :class:`~biopsykit.utils.dtypes.SubjectDataDict`
        ``SubjectDataDict``, i.e., a dictionary with a :class:`~biopsykit.utils.dtypes.PhaseDict`
        for each subject
    phase : str or :class:`~pandas.DataFrame`
        phase to normalize all other data to. If ``phase`` is a string then it is interpreted as the name of a phase
        present in ``subject_data_dict``. If ``phase`` is a DataFrame then the data will be normalized (per subject)
        to the mean value of the DataFrame.

    Returns
    -------
    dict
        dictionary with normalized data per subject

    """
    _assert_is_dtype(phase, (str, pd.DataFrame))
    dict_subjects_norm = {}
    for subject_id, data in subject_data_dict.items():
        bl_mean = data[phase].mean() if isinstance(phase, str) else phase.mean()
        dict_subjects_norm[subject_id] = {p: (df - bl_mean) / bl_mean * 100 for p, df in data.items()}

    return dict_subjects_norm


def resample_sec(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Resample input data to a frequency of 1 Hz.

    .. note::
        For resampling the index of ``data`` either be has to be a :class:`~pandas.DatetimeIndex`
        or a :class:`~pandas.Index` with time information in seconds.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame` or :class:`~pandas.Series`
        data to resample. Index of data needs to be a :class:`~pandas.DatetimeIndex`


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with data resampled to 1 Hz


    Raises
    ------
    ValueError
        If ``data`` is not a DataFrame or Series

    """
    _assert_is_dtype(data, (pd.DataFrame, pd.Series))

    column_name = data.columns if isinstance(data, pd.DataFrame) else [data.name]

    if isinstance(data.index, pd.DatetimeIndex):
        x_old = np.array((data.index - data.index[0]).total_seconds())
    else:
        x_old = np.array(data.index - data.index[0])
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = sanitize_input_1d(data)

    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=pd.Index(x_new, name="time"), columns=column_name)


def resample_dict_sec(
    data_dict: dict[str, Any],
) -> dict[str, Any]:
    """Resample all data in the dictionary to 1 Hz data.

    This function recursively looks for all dataframes in the dictionary and resamples data to 1 Hz using
    :func:`~biopsykit.utils.data_processing.resample_sec`.


    Parameters
    ----------
    data_dict : dict
        nested dictionary with data to be resampled


    Returns
    -------
    dict
        nested dictionary with data resampled to 1 Hz


    See Also
    --------
    :func:`~biopsykit.utils.data_processing.resample_sec`
        resample dataframe to 1 Hz

    """
    result_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            result_dict[key] = resample_sec(value)
        elif isinstance(value, dict):
            result_dict[key] = resample_dict_sec(value)
        else:
            raise TypeError("Invalid input format!")
    return result_dict


def select_dict_phases(subject_data_dict: SubjectDataDict, phases: Union[str, Sequence[str]]) -> SubjectDataDict:
    """Select specific phases from :obj:`~biopsykit.utils.dtypes.SubjectDataDict`.

    Parameters
    ----------
    subject_data_dict : :obj:`~biopsykit.utils.dtypes.SubjectDataDict`
        ``SubjectDataDict``, i.e. a dictionary with :obj:`~biopsykit.utils.dtypes.PhaseDict` for each subject
    phases : list of str
        list of phases to select

    Returns
    -------
    :obj:`~biopsykit.utils.dtypes.SubjectDataDict`
        ``SubjectDataDict`` containing only the phases of interest

    """
    is_subject_data_dict(subject_data_dict)
    if isinstance(phases, str):
        phases = [phases]
    return {
        subject: {phase: dict_subject[phase] for phase in phases} for subject, dict_subject in subject_data_dict.items()
    }


def rearrange_subject_data_dict(
    subject_data_dict: SubjectDataDict,
) -> StudyDataDict:
    """Rearrange ``SubjectDataDict`` to ``StudyDataDict``.

    A :obj:`~biopsykit.utils.dtypes.StudyDataDict` is constructed from a
    :obj:`~biopsykit.utils.dtypes.SubjectDataDict` by swapping outer (subject IDs) and inner
    (phase names) dictionary keys.

    The **input** needs to be a :obj:`~biopsykit.utils.dtypes.SubjectDataDict`,
    a nested dictionary in the following format:

    | {
    |   "subject1" : { "phase_1" : dataframe, "phase_2" : dataframe, ... },
    |   "subject2" : { "phase_1" : dataframe, "phase_2" : dataframe, ... },
    |   ...
    | }

    The **output** format will be the following:

    | {
    |   "phase_1" : { "subject1" : dataframe, "subject2" : dataframe, ... },
    |   "phase_2" : { "subject1" : dataframe, "subject2" : dataframe, ... },
    |   ...
    | }


    Parameters
    ----------
    subject_data_dict : :obj:`~biopsykit.utils.dtypes.SubjectDataDict`
        ``SubjectDataDict``, i.e. a dictionary with data from multiple subjects, each containing data from
        multiple phases (in form of a :obj:`~biopsykit.utils.dtypes.PhaseDict`)


    Returns
    -------
    :obj:`~biopsykit.utils.dtypes.StudyDataDict`
        rearranged ``SubjectDataDict``

    """
    dict_flipped = {}
    phases = [np.array(dict_phase.keys()) for dict_phase in subject_data_dict.values()]
    if not all(phases[0] == p for p in phases):
        raise ValueError(
            "Error rearranging the dictionary! Not all 'PhaseDict's have the same phases. "
            "To rearrange the 'SubjectDataDict', "
            "the dictionaries of all subjects need to have the exact same phases!"
        )

    for subject, phase_dict in subject_data_dict.items():
        for phase, df in phase_dict.items():
            dict_flipped.setdefault(phase, dict.fromkeys(subject_data_dict.keys()))
            dict_flipped[phase][subject] = df

    return dict_flipped


def cut_phases_to_shortest(study_data_dict: StudyDataDict, phases: Optional[Sequence[str]] = None) -> StudyDataDict:
    """Cut time-series data to shortest duration of a subject in each phase.

    To overlay time-series data from multiple subjects in an `ensemble plot` it is beneficial if all data have
    the same length. For that reason, data can be cut to the same length using this function.

    Parameters
    ----------
    study_data_dict : :obj:`~biopsykit.utils.dtypes.StudyDataDict`
        ``StudyDataDict``, i.e. a dictionary with data from multiple phases, each phase containing data from
        different subjects.
    phases : list of str, optional
        list of phases if only a subset of phases should be cut or ``None`` to cut all phases.
        Default: ``None``

    Returns
    -------
    :obj:`~biopsykit.utils.dtypes.StudyDataDict`
        ``StudyDataDict`` with data cut to the shortest duration in each phase

    """
    is_study_data_dict(study_data_dict)

    if phases is None:
        phases = study_data_dict.keys()

    dict_cut = {}
    for phase in phases:
        min_dur = min(len(df) for df in study_data_dict[phase].values())
        dict_cut[phase] = {subject: df.iloc[0:min_dur].copy() for subject, df in study_data_dict[phase].items()}

    is_study_data_dict(study_data_dict)
    return dict_cut


def merge_study_data_dict(
    study_data_dict: StudyDataDict, dict_levels: Optional[Sequence[str]] = None
) -> MergedStudyDataDict:
    """Merge inner dictionary level of :obj:`~biopsykit.utils.dtypes.StudyDataDict` into one dataframe.

    This function removes the inner level of the nested ``StudyDataDict`` by merging data from all subjects
    into one dataframe for each phase.

    .. note::
        To merge data from different subjects into one dataframe the data are all expected to have the same length!
        If this is not the case, all data needs to be cut to equal length first, e.g. using
        :func:`~biopsykit.utils.data_processing.cut_phases_to_shortest`.


    Parameters
    ----------
    study_data_dict : :obj:`~biopsykit.utils.dtypes.StudyDataDict`
        ``StudyDataDict``, i.e. a dictionary with data from multiple phases, each phase containing data from
        different subjects.
    dict_levels : list of str
        list with names of dictionary levels.


    Returns
    -------
    :obj:`~biopsykit.utils.dtypes.MergedStudyDataDict`
        ``MergedStudyDataDict`` with data of all subjects merged into one dataframe for each phase

    """
    is_study_data_dict(study_data_dict)

    if dict_levels is None:
        dict_levels = ["phase", "subject"]

    dict_merged = {}
    for phase, dict_phase in study_data_dict.items():
        _assert_dataframes_same_length(list(dict_phase.values()))
        df_merged = pd.concat(dict_phase, names=dict_levels[1:], axis=1)
        df_merged.columns = df_merged.columns.droplevel(-1)
        dict_merged[phase] = df_merged

    is_merged_study_data_dict(dict_merged)
    return dict_merged


def split_dict_into_subphases(
    data_dict: dict[str, Any],
    subphases: dict[str, int],
) -> Union[dict[str, dict[str, Any]]]:
    """Split dataframes in a nested dictionary into subphases.

    By further splitting a dataframe into subphases a new dictionary level is created. The new dictionary level
    then contains the subphases with their data.

    .. note::
        If the duration of the last subphase is unknown (e.g., because it has variable length) this can be
        indicated by setting the duration of this subphase to 0.
        The duration of this subphase will then be inferred from the data.

    Parameters
    ----------
    data_dict : dict
        dictionary with an arbitrary number of outer level (e.g., conditions, phases, etc.) as keys and
        dataframes with data to be split into subphases as values
    subphases : dict
        dictionary with subphase names (keys) and subphase durations (values) in seconds

    Returns
    -------
    dict
        dictionary where each dataframe in the dictionary is split into the subphases specified by ``subphases``

    """
    result_dict = {}
    for key, value in data_dict.items():
        _assert_is_dtype(value, (dict, pd.DataFrame))
        if isinstance(value, dict):
            # nested dictionary
            result_dict[key] = split_dict_into_subphases(value, subphases)
        else:
            subphase_times = get_subphase_durations(value, subphases)
            subphase_dict = {}
            for subphase, times in zip(subphases.keys(), subphase_times):
                if isinstance(value.index, pd.DatetimeIndex):
                    # slice the current subphase by dropping the preceding subphases
                    mask_drop = value.index < value.index[0] + pd.Timedelta(seconds=times[0])
                    value_cpy = value[~mask_drop]
                    mask_keep = value_cpy.index <= value.index[0] + pd.Timedelta(seconds=times[1])
                    value_cpy = value_cpy[mask_keep]
                    subphase_dict[subphase] = value_cpy
                else:
                    subphase_dict[subphase] = value.iloc[times[0] : times[1]]
            result_dict[key] = subphase_dict
    return result_dict


def get_subphase_durations(
    data: pd.DataFrame, subphases: dict[str, Union[int, tuple[int, int]]]
) -> Sequence[tuple[int, int]]:
    """Compute subphase durations from dataframe.

    The subphases can be specified in two different ways:

    * If the dictionary entries in ``subphases`` are integer, it's assumed that subphases are consecutive,
      i.e., each subphase begins right after the previous one, and the entries indicate the *durations* of each
      subphase. The start and end times of each subphase will then be computed from the subphase durations.
    * If the dictionary entries in ``subphases`` are tuples, it's assumed that the start and end times of each
      subphase are directly provided.

    .. note::
        If the duration of the last subphase is unknown (e.g., because it has variable length) this can be
        indicated by setting the duration of this subphase to 0.
        The duration of this subphase will then be inferred from the data.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with data from one phase. Used to compute the duration of the last subphase if this subphase
        is expected to have variable duration.
    subphases : dict
        dictionary with subphase names as keys and subphase durations (as integer) or start and end
        times (as tuples of integer) as values in seconds

    Returns
    -------
    list
        list with start and end times of each subphase in seconds relative to beginning of the phase


    Examples
    --------
    >>> from biopsykit.utils.data_processing import get_subphase_durations
    >>> # Option 1: Subphases consecutive, subphase durations provided
    >>> get_subphase_durations(data, {"Start": 60, "Middle": 120, "End": 60})
    >>> # Option 2: Subphase start and end times provided
    >>> get_subphase_durations(data, {"Start": (0, 50), "Middle": (60, 160), "End": (180, 240)})

    """
    subphase_durations = np.array(list(subphases.values()))
    if subphase_durations.ndim == 1:
        # 1d array => subphase values are integer => they are consecutive and each entry is the duration
        # of the subphase, so the start and end times of each subphase must be computed
        times_cum = np.cumsum(subphase_durations)
        if subphase_durations[-1] == 0:
            # last subphase has duration 0 => end of last subphase is length of dataframe
            times_cum[-1] = len(data)
        subphase_times = list(zip([0, *list(times_cum)], times_cum))
    else:
        # 2d array => subphase values are tuples => start end end time of each subphase are already provided and do
        # not need to be computed
        subphase_times = subphase_durations
    return subphase_times


def add_subject_conditions(
    data: pd.DataFrame, condition_list: Union[SubjectConditionDict, SubjectConditionDataFrame]
) -> pd.DataFrame:
    """Add subject conditions to dataframe.

    This function expects a dataframe with data from multiple subjects and information on which subject
    belongs to which condition.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe where new index level ``condition`` with subject conditions should be added to
    condition_list : ``SubjectConditionDict`` or ``SubjectConditionDataFrame``
        :obj:`~biopsykit.utils.dtypes.SubjectConditionDict` or
        :obj:`~biopsykit.utils.dtypes.SubjectConditionDataFrame` with information on which subject belongs to
        which condition


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with new index level ``condition`` indicating which subject belongs to which condition

    """
    if is_subject_condition_dataframe(condition_list, raise_exception=False):
        condition_list = condition_list.groupby("condition").groups
    is_subject_condition_dict(condition_list)
    return pd.concat({cond: data.loc[subjects] for cond, subjects in condition_list.items()}, names=["condition"])


def split_subject_conditions(
    data_dict: dict[str, Any], condition_list: Union[SubjectConditionDict, SubjectConditionDataFrame]
) -> dict[str, dict[str, Any]]:
    """Split dictionary with data based on conditions subjects were assigned to.

    This function adds a new outer dictionary level with the different conditions as keys and dictionaries
    belonging to the conditions as values. For that, it expects a dictionary with data from multiple subjects and
    information on which subject belongs to which condition.


    Parameters
    ----------
    data_dict : dict
        (nested) dictionary with data which should be split based on the conditions subjects belong to
    condition_list : ``SubjectConditionDict`` or ``SubjectConditionDataFrame``
        :obj:`~biopsykit.utils.dtypes.SubjectConditionDict` or
        :obj:`~biopsykit.utils.dtypes.SubjectConditionDataFrame` with information on which subject belongs to
        which condition


    Returns
    -------
    dict
        dictionary with additional outer level indicating which subject belongs to which condition

    """
    if is_subject_condition_dataframe(condition_list, raise_exception=False):
        condition_list = condition_list.groupby("condition").groups
    is_subject_condition_dict(condition_list)
    return {cond: _splits_subject_conditions(data_dict, subjects) for cond, subjects in condition_list.items()}


def _splits_subject_conditions(data_dict: dict[str, Any], subject_list: Sequence[str]):
    _assert_is_dtype(data_dict, (dict, pd.DataFrame))
    if isinstance(data_dict, pd.DataFrame):
        return data_dict[subject_list]
    return {key: _splits_subject_conditions(value, subject_list) for key, value in data_dict.items()}


def mean_per_subject_dict(data: dict[str, Any], dict_levels: Sequence[str], param_name: str) -> pd.DataFrame:
    """Compute mean values of time-series data from a nested dictionary.

    This function computes the mean value of time-series data in a nested dictionary per subject and combines it into
    a joint dataframe. The dictionary will be traversed recursively and can thus have arbitrary depth.
    The most inner level must contain dataframes with time-series data of which mean values will be computed.
    The names of the dictionary levels are specified by ``dict_levels``.


    Parameters
    ----------
    data: dict
        nested dictionary with data on which mean should be computed. The number of nested levels must match the
        number of levels specified in ``dict_levels``.
    dict_levels : list of str
        list with names of dictionary levels.
    param_name : str
        type of data of which mean values will be computed from.
        This will also be the column name in the resulting dataframe.


    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with ``dict_levels`` as index levels and mean values of time-series data as column values

    """
    result_data = {}

    one_col_df = False
    for key, value in data.items():
        _assert_is_dtype(value, (dict, pd.DataFrame))
        if isinstance(value, dict):
            if len(dict_levels) <= 1:
                raise ValueError("Invalid number of 'dict_levels' specified!")
            # nested dictionary
            key_len = 1 if isinstance(key, (str, int)) else len(key)
            result_data[key] = mean_per_subject_dict(value, dict_levels[key_len:], param_name)
        else:
            value.columns.name = "subject"
            if len(value.columns) == 1:
                one_col_df = True
            df = pd.DataFrame(value.mean(axis=0), columns=[param_name])
            result_data[key] = df

    key_lengths = list({1 if isinstance(k, (str, int)) else len(k) for k in result_data})
    if len(key_lengths) != 1:
        raise ValueError("Inconsistent dictionary key lengths!")
    key_lengths = key_lengths[0]
    names = dict_levels[0:key_lengths]
    if isinstance(names, str):
        names = [names]
    ret = pd.concat(result_data, names=names)
    if one_col_df:
        ret.index = ret.index.droplevel(-1)
    return ret


def mean_se_per_phase(data: pd.DataFrame) -> MeanSeDataFrame:
    """Compute mean and standard error over all subjects in a dataframe.

    .. note::
        The dataframe in ``data`` is expected to have a :class:`~pandas.MultiIndex` with at least two levels,
        one of the levels being the level "subject"!

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe with :class:`~pandas.MultiIndex` from which to compute mean and standard error

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with mean and standard error over all subjects

    """
    # expect dataframe to have at least 2 levels, one of it being "subject"
    _assert_has_multiindex(data, expected=True, nlevels=2, nlevels_atleast=True)
    _assert_has_index_levels(data, ["subject"], match_atleast=True)

    # group by all columns except the "subject" column
    group_cols = list(data.index.names)
    group_cols.remove("subject")

    data = data.groupby(group_cols, sort=False).agg([np.mean, se])
    is_mean_se_dataframe(data)

    return _MeanSeDataFrame(data)
