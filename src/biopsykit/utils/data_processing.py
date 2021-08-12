"""Module providing various functions for processing more complex structured data (e.g., collected during a study)."""
import warnings
from typing import Sequence, Union, Dict, Optional, Tuple, Any

# from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from scipy import interpolate

from biopsykit.utils._datatype_validation_helper import (
    _assert_is_dtype,
    _assert_dataframes_same_length,
    _assert_has_index_levels,
    _assert_has_multiindex,
)
from biopsykit.utils.functions import se
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.datatype_helper import (
    SubjectConditionDict,
    SubjectConditionDataFrame,
    is_subject_condition_dataframe,
    is_subject_condition_dict,
    SubjectDataDict,
    StudyDataDict,
    MergedStudyDataDict,
    is_merged_study_data_dict,
    is_subject_data_dict,
    is_study_data_dict,
    MeanSeDataFrame,
    _MeanSeDataFrame,
    is_mean_se_dataframe,
)


def _split_data_series(data: pd.DataFrame, time_intervals: pd.Series, include_start: bool) -> Dict[str, pd.DataFrame]:
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
    time_intervals: Union[pd.DataFrame, pd.Series, Dict[str, Sequence[str]]],
    include_start: Optional[bool] = False,
) -> Dict[str, pd.DataFrame]:
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
    else:
        if include_start:
            time_intervals["Start"] = (
                data.index[0].to_pydatetime().time(),
                list(time_intervals.values())[0][0],
            )

    data_dict = {name: data.between_time(*start_end) for name, start_end in time_intervals.items()}
    data_dict = {name: data for name, data in data_dict.items() if not data.empty}
    return data_dict


def exclude_subjects(
    excluded_subjects: Union[Sequence[str], Sequence[int]], index_name: Optional[str] = "subject", **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Exclude subjects from dataframes.

    This function can be used to exclude subject IDs for later analysis from different kinds of dataframes, such as:

    * dataframes with subject condition information
      (:obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame`)
    * dataframes with time log information
    * dataframes with (processed) data (e.g., :obj:`biopsykit.utils.datatype_helper.SalivaRawDataFrame` or
      :obj:`~biopsykit.utils.datatype_helper.MeanSeDataFrame`)

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
    cleaned_data: Dict[str, pd.DataFrame] = {}

    for key, data in kwargs.items():
        _assert_is_dtype(data, pd.DataFrame)
        if index_name in data.index.names:
            level_values = data.index.get_level_values(index_name)
            if (level_values.dtype == np.object and all([isinstance(s, str) for s in excluded_subjects])) or (
                level_values.dtype == np.int and all([isinstance(s, int) for s in excluded_subjects])
            ):
                cleaned_data[key] = _exclude_single_subject(data, excluded_subjects, index_name)
            raise ValueError("{}: dtypes of index and subject ids to be excluded do not match!".format(key))
        raise ValueError("No '{}' level in index!".format(index_name))
    if len(cleaned_data) == 1:
        cleaned_data = list(cleaned_data.values())[0]
    return cleaned_data


def _exclude_single_subject(
    data: pd.DataFrame,
    excluded_subjects: Union[Sequence[str], Sequence[int]],
    index_name: str,
):
    # dataframe index and subjects are both strings or both integers
    try:
        if isinstance(data.index, pd.MultiIndex):
            # MultiIndex => specify index level
            return data.drop(index=excluded_subjects, level=index_name)
        # Regular Index
        return data.drop(index=excluded_subjects)
    except KeyError:
        warnings.warn("Not all subjects of {} exist in the dataset!".format(excluded_subjects))


def normalize_to_phase(subject_data_dict: SubjectDataDict, phase: Union[str, pd.DataFrame]) -> SubjectDataDict:
    """Normalize time series data per subject to the phase specified by ``normalize_to``.

    The result is the relative change (of, for example, heart rate) compared to the mean value in ``phase``.

    Parameters
    ----------
    subject_data_dict : :class:`~biopsykit.utils.datatype_helper.SubjectDataDict`
        ``SubjectDataDict``, i.e., a dictionary with a :class:`~biopsykit.utils.datatype_helper.PhaseDict`
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
        if isinstance(phase, str):
            bl_mean = data[phase].mean()
        else:
            bl_mean = phase.mean()
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

    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    else:
        column_name = [data.name]

    if isinstance(data.index, pd.DatetimeIndex):
        x_old = np.array((data.index - data.index[0]).total_seconds())
    else:
        x_old = np.array(data.index - data.index[0])
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = sanitize_input_1d(data)

    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=pd.Index(x_new, name="time"), columns=column_name)


def resample_dict_sec(
    data_dict: Dict[str, Any],
) -> Dict[str, Any]:
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
            raise ValueError("Invalid input format!")
    return result_dict


def select_dict_phases(subject_data_dict: SubjectDataDict, phases: Sequence[str]) -> SubjectDataDict:
    """Select specific phases from :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict`.

    Parameters
    ----------
    subject_data_dict : :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict`
        ``SubjectDataDict``, i.e. a dictionary with :obj:`~biopsykit.utils.datatype_helper.PhaseDict` for each subject
    phases : list of str
        list of phases to select

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict`
        ``SubjectDataDict`` containing only the phases of interest

    """
    is_subject_data_dict(subject_data_dict)
    return {
        subject: {phase: dict_subject[phase] for phase in phases} for subject, dict_subject in subject_data_dict.items()
    }


def rearrange_subject_data_dict(
    subject_data_dict: SubjectDataDict,
) -> StudyDataDict:
    """Rearrange :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict` to \
    :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`.

    A ``StudyDataDict`` is constructed from a ``SubjectDataDict`` by swapping outer (subject IDs) and inner
    (phase names) dictionary keys.

    The **input** needs to be a :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict`,
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
    subject_data_dict : :obj:`~biopsykit.utils.datatype_helper.SubjectDataDict`
        ``SubjectDataDict``, i.e. a dictionary with data from multiple subjects, each containing data from
        multiple phases (in form of a :obj:`~biopsykit.utils.datatype_helper.PhaseDict`)


    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`
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
    study_data_dict : :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`
        ``StudyDataDict``, i.e. a dictionary with data from multiple phases, each phase containing data from
        different subjects.
    phases : list of str, optional
        list of phases if only a subset of phases should be cut or ``None`` to cut all phases.
        Default: ``None``

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`
        ``StudyDataDict`` with data cut to the shortest duration in each phase

    """
    is_study_data_dict(study_data_dict)

    if phases is None:
        phases = study_data_dict.keys()

    dict_cut = {}
    for phase in phases:
        min_dur = min([len(df) for df in study_data_dict[phase].values()])
        dict_cut[phase] = {subject: df.iloc[0:min_dur].copy() for subject, df in study_data_dict[phase].items()}

    is_study_data_dict(study_data_dict)
    return dict_cut


def merge_study_data_dict(study_data_dict: StudyDataDict) -> MergedStudyDataDict:
    """Merge inner dictionary level of :obj:`~biopsykit.utils.datatype_helper.StudyDataDict` into one dataframe.

    This function removes the inner level of the nested ``StudyDataDict`` by merging data from all subjects
    into one dataframe for each phase.

    .. note::
        To merge data from different subjects into one dataframe the data are all expected to have the same length!
        If this is not the case, all data needs to be cut to equal length first, e.g. using
        :func:`~biopsykit.utils.data_processing.cut_phases_to_shortest`.


    Parameters
    ----------
    study_data_dict : :obj:`~biopsykit.utils.datatype_helper.StudyDataDict`
        ``StudyDataDict``, i.e. a dictionary with data from multiple phases, each phase containing data from
        different subjects.


    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.MergedStudyDataDict`
        ``MergedStudyDataDict`` with data of all subjects merged into one dataframe for each phase

    """
    is_study_data_dict(study_data_dict)

    dict_merged = {}
    for phase, dict_phase in study_data_dict.items():
        _assert_dataframes_same_length(list(dict_phase.values()))
        df_merged = pd.concat(dict_phase, names=["subject"], axis=1)
        df_merged.columns = df_merged.columns.droplevel(1)
        dict_merged[phase] = df_merged

    is_merged_study_data_dict(dict_merged)
    return dict_merged


def split_dict_into_subphases(
    data_dict: Dict[str, Any],
    subphases: Dict[str, int],
) -> Union[Dict[str, Dict[str, Any]]]:
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
                subphase_dict[subphase] = value.iloc[times[0] : times[1]]
            result_dict[key] = subphase_dict
    return result_dict


def get_subphase_durations(data: pd.DataFrame, subphases: Dict[str, int]) -> Sequence[Tuple[int, int]]:
    """Compute subphase durations from dataframe.

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
        dictionary with subphase names (keys) and subphase durations (values) in seconds

    Returns
    -------
    list
        list with start and end times of each subphase in seconds relative to beginning of the phase

    """
    subphase_durations = list(subphases.values())
    times_cum = np.cumsum(subphase_durations)
    if subphase_durations[-1] == 0:
        # last subphase has duration 0 => end of last subphase is length of dataframe
        times_cum[-1] = len(data)
    subphase_times = list(zip([0] + list(times_cum), times_cum))
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
        :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDict` or
        :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame` with information on which subject belongs to
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
    data_dict: Dict[str, Any], condition_list: Union[SubjectConditionDict, SubjectConditionDataFrame]
) -> Dict[str, Dict[str, Any]]:
    """Split dictionary with data based on conditions subjects were assigned to.

    This function adds a new outer dictionary level with the different conditions as keys and dictionaries
    belonging to the conditions as values. For that, it expects a dictionary with data from multiple subjects and
    information on which subject belongs to which condition.


    Parameters
    ----------
    data_dict : dict
        (nested) dictionary with data which should be split based on the conditions subjects belong to
    condition_list : ``SubjectConditionDict`` or ``SubjectConditionDataFrame``
        :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDict` or
        :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame` with information on which subject belongs to
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


def _splits_subject_conditions(data_dict: Dict[str, Any], subject_list: Sequence[str]):
    _assert_is_dtype(data_dict, (dict, pd.DataFrame))
    if isinstance(data_dict, pd.DataFrame):
        return data_dict[subject_list]
    return {key: _splits_subject_conditions(value, subject_list) for key, value in data_dict.items()}


# def param_subphases(
#     ecg_processor: Optional["EcgProcessor"] = None,
#     dict_ecg: Optional[Dict[str, pd.DataFrame]] = None,
#     dict_rpeaks: Optional[Dict[str, pd.DataFrame]] = None,
#     subphases: Optional[Sequence[str]] = None,
#     subphase_durations: Optional[Sequence[int]] = None,
#     param_types: Optional[Union[str, Sequence[str]]] = "all",
#     sampling_rate: Optional[int] = 256,
#     include_total: Optional[bool] = True,
#     title: Optional[str] = None,
# ) -> pd.DataFrame:
#     """
#     Computes specified parameters (HRV / RSA / ...) over phases and subphases **for one subject**.
#
#     To use this function, either simply pass an ``EcgProcessor`` object or two dictionaries
#     ``dict_ecg`` and ``dict_rpeaks`` resulting from ``EcgProcessor.ecg_process()``.
#
#     Parameters
#     ----------
#     ecg_processor : EcgProcessor, optional
#         `EcgProcessor` object
#     dict_ecg : dict, optional
#         dict with dataframes of processed ECG signals. Output from `EcgProcessor.ecg_process()`.
#     dict_rpeaks : dict, optional
#         dict with dataframes of processed R peaks. Output from `EcgProcessor.ecg_process()`.
#     subphases : list of int
#         list of subphase names
#     subphase_durations : list of str
#         list of subphase durations
#     param_types : list or str, optional
#         list with parameter types to compute or 'all' to compute all available parameters. Choose from a subset of
#         ['hrv', 'rsa'] to compute HRV and RSA parameters, respectively.
#     sampling_rate : float, optional
#         Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz
#     include_total : bool, optional
#         ``True`` to also compute parameters over the complete phases (in addition to only over subphases),
#         ``False`` to only compute parameters over the single subphases. Default: ``True``
#     title : str, optional
#         Optional title of the processing progress bar. Default: ``None``
#
#     Returns
#     -------
#     pd.DataFrame
#         dataframe with computed parameters over the single subphases
#     """
#     import biopsykit.signals.ecg as ecg
#
#     if ecg_processor is None and dict_rpeaks is None and dict_ecg is None:
#         raise ValueError("Either `ecg_processor` or `dict_rpeaks` and `dict_ecg` must be passed as arguments!")
#
#     if subphases is None or subphase_durations is None:
#         raise ValueError("Both `subphases` and `subphase_durations` are required as parameter!")
#
#     # TODO change
#     # get all desired parameter types
#     possible_param_types = {
#         "hrv": ecg.EcgProcessor.hrv_process,
#         # "rsp": ecg.EcgProcessor.rsp_rsa_process,
#     }
#     if param_types == "all":
#         param_types = possible_param_types
#
#     if isinstance(param_types, str):
#         param_types = {param_types: possible_param_types[param_types]}
#     if not all([param in possible_param_types for param in param_types]):
#         raise ValueError(
#             "`param_types` must all be of {}, not {}".format(possible_param_types.keys(), param_types.keys())
#         )
#
#     param_types = {param: possible_param_types[param] for param in param_types}
#
#     if ecg_processor:
#         sampling_rate = ecg_processor.sampling_rate
#         dict_rpeaks = ecg_processor.rpeaks
#         dict_ecg = ecg_processor.ecg_result
#
#     if "rsp" in param_types and dict_ecg is None:
#         raise ValueError("`dict_ecg` must be passed if param_type is {}!".format(param_types))
#
#     index_name = "subphase"
#     # dict to store results. one entry per parameter and a list of dataframes per MIST phase
#     # that will later be concated to one large dataframes
#     dict_df_subphases = {param: list() for param in param_types}
#
#     # iterate through all phases in the data
#     for (phase, rpeaks), (ecg_phase, ecg_data) in tqdm(zip(dict_rpeaks.items(), dict_ecg.items()), desc=title):
#         rpeaks = rpeaks.copy()
#         ecg_data = ecg_data.copy()
#
#         # dict to store intermediate results of subphases. one entry per parameter with a
#         # list of dataframes per subphase that will later be concated to one dataframe per MIST phase
#         dict_subphases = {param: list() for param in param_types}
#         if include_total:
#             # compute HRV, RSP over complete phase
#             for param_type, param_func in param_types.items():
#                 dict_subphases[param_type].append(
#                     param_func(
#                         ecg_signal=ecg_data,
#                         rpeaks=rpeaks,
#                         index="Total",
#                         index_name=index_name,
#                         sampling_rate=sampling_rate,
#                     )
#                 )
#
#         if phase not in ["Part1", "Part2"]:
#             # skip Part1, Part2 for subphase parameter analysis (parameters in total are computed above)
#             for subph, dur in zip(subphases, subphase_durations):
#                 # get the first xx seconds of data (i.e., get only the current subphase)
#                 if dur > 0:
#                     df_subph_rpeaks = rpeaks.first("{}S".format(dur))
#                 else:
#                     # duration of 0 seconds = Feedback Interval, don't cut the beginning,
#                     # use all remaining data
#                     df_subph_rpeaks = rpeaks
#                 # ECG does not need to be sliced because rpeaks are already sliced and
#                 # will select only the relevant ECG signal parts anyways
#                 df_subph_ecg = ecg_data
#
#                 for param_type, param_func in param_types.items():
#                     # compute HRV, RSP over subphases
#                     dict_subphases[param_type].append(
#                         param_func(
#                             ecg_signal=df_subph_ecg,
#                             rpeaks=df_subph_rpeaks,
#                             index=subph,
#                             index_name=index_name,
#                             sampling_rate=sampling_rate,
#                         )
#                     )
#
#                 # remove the currently analyzed subphase of data
#                 # (so that the next subphase is first in the next iteration)
#                 rpeaks = rpeaks.drop(df_subph_rpeaks.index)
#
#         for param in dict_subphases:
#             # concat dataframe of all subphases to one dataframe per MIST phase and add to parameter dict
#             dict_df_subphases[param].append(pd.concat(dict_subphases[param]))
#
#     # concat all dataframes together to one big result dataframes
#     return pd.concat(
#         [pd.concat(dict_df, keys=dict_rpeaks.keys(), names=["phase"]) for dict_df in dict_df_subphases.values()],
#         axis=1,
#     )


def mean_per_subject_dict(data: Dict[str, Any], dict_levels: Sequence[str], param_name: str) -> pd.DataFrame:
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
            result_data[key] = mean_per_subject_dict(value, dict_levels[1:], param_name)
        else:
            if len(value.columns) == 1:
                one_col_df = True
            df = pd.DataFrame(value.mean(axis=0), columns=[param_name])
            result_data[key] = df

    ret = pd.concat(result_data, names=[dict_levels[0]])
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

    data = data.groupby(group_cols).agg([np.mean, se])
    is_mean_se_dataframe(data)

    return _MeanSeDataFrame(data)
