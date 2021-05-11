import warnings
from numbers import Number
from typing import Sequence, Union, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from biopsykit.utils.datatype_helper import (
    SubjectConditionDict,
    SubjectConditionDataFrame,
    is_subject_condition_dataframe,
    is_subject_condition_dict,
)
from nilspodlib import Dataset
from tqdm.notebook import tqdm

from biopsykit.utils.time import tz, utc


def split_data(
    time_intervals: Union[pd.DataFrame, pd.Series, Dict[str, Sequence[str]]],
    dataset: Optional[Dataset] = None,
    data: Optional[pd.DataFrame] = None,
    timezone: Optional[Union[str, pytz.timezone]] = tz,
    include_start: Optional[bool] = False,
) -> Dict[str, pd.DataFrame]:
    """
    Splits the data into parts based on time intervals.

    Parameters
    ----------
    time_intervals : dict or pd.Series or pd.DataFrame
        time intervals indicating where the data should be split.
        Can either be a pandas Series or 1 row of a pandas Dataframe with the `start` times of the single phases
        (the names of the phases are then derived from the index in case of a Series or column names in case of a
        Dataframe) or a dictionary with tuples indicating start and end times of the phases
        (the names of the phases are then derived from the dict keys)
    dataset : Dataset, optional
        NilsPodLib dataset object to be split
    data : pd.DataFrame, optional
        data to be split
    timezone : str or pytz.timezone, optional
        timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')
    include_start: bool, optional
        ``True`` to include the data from the beginning of the recording to the first time interval as the
        first interval, ``False`` otherwise. Default: ``False``

    Returns
    -------
    dict
        Dictionary containing split data

    Examples
    --------
    >>> from biopsykit.utils.data_processing import split_data
    >>> # Example 1: define time intervals (start and end) of the different recording phases as dictionary
    >>> time_intervals = {"Part1": ("09:00", "09:30"), "Part2": ("09:30", "09:45"), "Part3": ("09:45", "10:00")}
    >>> # Example 2: define time intervals as pandas Series. Here, only start times of the are required, it is assumed
    >>> # that the phases are back to back
    >>> time_intervals = pd.Series(data=["09:00", "09:30", "09:45", "10:00"], index=["Part1", "Part2", "Part3", "End"])
    >>>
    >>> # read pandas dataframe from csv file and split data based on time interval dictionary
    >>> df = pd.read_csv(path_to_file)
    >>> data_dict = split_data(time_intervals, data=data)
    >>>
    >>> # Example: Get Part 2 of data_dict
    >>> print(data_dict['Part2'])
    """
    data_dict: Dict[str, pd.DataFrame] = {}
    if dataset is None and data is None:
        raise ValueError("Either 'dataset' or 'df' must be specified as parameter!")
    if dataset:
        if isinstance(timezone, str):
            # convert to pytz object
            timezone = pytz.timezone(timezone)
        data = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utc).tz_convert(tz=timezone)
    if isinstance(time_intervals, pd.DataFrame):
        if len(time_intervals) > 1:
            raise ValueError("Only dataframes with 1 row allowed!")
        time_intervals = time_intervals.iloc[0]

    if isinstance(time_intervals, pd.Series):
        if time_intervals.index.nlevels > 1:
            # multi-index series => second level contains start/end times of phases
            time_intervals = time_intervals.unstack().T
            time_intervals = {key: tuple(value.values()) for key, value in time_intervals.to_dict().items()}
        else:
            if include_start:
                time_intervals["Start"] = data.index[0].to_pydatetime().time()
            time_intervals.sort_values(inplace=True)
            for name, start, end in zip(time_intervals.index, np.pad(time_intervals, (0, 1)), time_intervals[1:]):
                data_dict[name] = data.between_time(start, end)

    if isinstance(time_intervals, dict):
        if include_start:
            time_intervals["Start"] = (
                data.index[0].to_pydatetime().time(),
                list(time_intervals.values())[0][0],
            )
        data_dict = {name: data.between_time(*start_end) for name, start_end in time_intervals.items()}
    return data_dict


def exclude_subjects(
    excluded_subjects: Union[Sequence[str], Sequence[int]], id_column: Optional[str] = "subject", **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    cleaned_data: Dict[str, pd.DataFrame] = {}

    for key, data in kwargs.items():
        if id_column in data.index.names:
            if (
                data.index.get_level_values(id_column).dtype == np.object
                and all([isinstance(s, str) for s in excluded_subjects])
            ) or (
                data.index.get_level_values(id_column).dtype == np.int
                and all([isinstance(s, int) for s in excluded_subjects])
            ):
                # dataframe index and subjects are both strings or both integers
                try:
                    if isinstance(data.index, pd.MultiIndex):
                        # MultiIndex => specify index level
                        cleaned_data[key] = data.drop(index=excluded_subjects, level=id_column)
                    else:
                        # Regular Index
                        cleaned_data[key] = data.drop(index=excluded_subjects)
                except KeyError:
                    warnings.warn("Not all subjects of {} exist in the dataset!".format(excluded_subjects))
            else:
                raise ValueError("{}: dtypes of index and subject ids to be excluded do not match!".format(key))
        else:
            raise ValueError("No '{}' level in index!".format(id_column))
    if len(cleaned_data) == 1:
        cleaned_data = list(cleaned_data.values())[0]
    return cleaned_data


def concat_phase_dict(
    dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
    phases: Optional[Sequence[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Rearranges a 'HR subject dict' (a nested dictionary containing heart rate data, see below and
    ``utils.load_hr_excel_all_subjects()`` for further information) into a 'Phase dict', i.e. a dictionary with
    one dataframe per Phase where each dataframe contains column-wise HR data for all subjects.

    The **input** needs to be a 'HR subject dict', a nested dictionary with the following format:
    {
        "<Subject_1>" : {
            "<Phase_1>" : hr_dataframe,
            "<Phase_2>" : hr_dataframe,
            ...
        },
        "<Subject_2>" : {
            "<Phase_1>" : hr_dataframe,
            "<Phase_2>" : hr_dataframe,
            ...
        },
        ...
    }

    The **output** format will be the following:

    { "<Phase>" : hr_dataframe, 1 subject per column }

    E.g., see ``biopsykit.protocols.mist.MIST.concat_phase_dict()`` for further information.

    Parameters
    ----------
    dict_hr_subject : dict
        'HR subject dict', i.e. a nested dict with heart rate data per phase and subject
    phases : list, optional
        list of phase names. If `None` is passed, phases are inferred from the keys of the first subject

    Returns
    -------
    dict
        'Phase dict', i.e. a dict with heart rate data of all subjects per phase

    """

    if phases is None:
        phases = list(dict_hr_subject.values())[0].keys()

    dict_phase: Dict[str, pd.DataFrame] = {key: pd.DataFrame(columns=list(dict_hr_subject.keys())) for key in phases}
    for subj in dict_hr_subject:
        dict_bl = dict_hr_subject[subj]
        for phase in phases:
            dict_phase[phase][subj] = dict_bl[phase]["Heart_Rate"]

    return dict_phase


def split_subphases(
    data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
    subphase_names: Sequence[str],
    subphase_times: Sequence[Tuple[int, int]],
    is_group_dict: Optional[bool] = False,
) -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Splits a `Phase dict` (or a dict of such, in case of multiple groups, see ``bp.signals.utils.concat_phase_dict``)
    into a `Subphase dict` (see below for further explanation).

    The **input** is a `Phase dict`, i.e. a dictionary with data (e.g. heart rate) per Phase
    in the following format:

    { <"Phase"> : dataframe, 1 subject per column }

    If multiple groups are present, then the expected input is nested, i.e. a dict of 'Phase dicts',
    with one entry per group.

    The **output** is a `Subphase dict`, i.e. a nested dictionary with data (e.g. heart rate) per Subphase in the
    following format:

    { <"Phase"> : { <"Subphase"> : dataframe, 1 subject per column } }

    If multiple groups are present, then the output is nested, i.e. a dict of 'Subphase dicts',
    with one entry per group.


    Parameters
    ----------
    data : dict
        'Phase dict' or nested dict of 'Phase dicts' if `is_group_dict` is ``True``
    subphase_names : list
        List with names of subphases
    subphase_times : list
        List with start and end times (as tuples) of each subphase in seconds or list with subphase durations
    is_group_dict : bool, optional
        ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``

    Returns
    -------
    dict
        'Subphase dict' with course of data per Phase, Subphase and Subject, respectively or
        nested dict of 'Subphase dicts' if `is_group_dict` is ``True``

    """
    if isinstance(subphase_times[0], Number):
        times_cum = np.cumsum(np.array(subphase_times))
        subphase_times = [(start, end) for start, end in zip(np.append([0], times_cum[:-1]), times_cum)]

    if is_group_dict:
        # recursively call this function for each group
        return {
            group: split_subphases(dict_group, subphase_names=subphase_names, subphase_times=subphase_times)
            for group, dict_group in data.items()
        }
    else:
        phase_dict = {}
        # split data into subphases for each Phase
        for phase, df in data.items():
            phase_dict[phase] = {subph: df[start:end] for subph, (start, end) in zip(subphase_names, subphase_times)}
        return phase_dict


def split_groups(
    phase_dict: Dict[str, pd.DataFrame], condition_list: Union[SubjectConditionDict, SubjectConditionDataFrame]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Splits 'Phase dict' into group dict, i.e. one 'Phase dict' per group.

    Parameters
    ----------
    phase_dict : dict
        'Phase dict' to be split in groups. See ``bp.signals.utils.concat_phase_dict`` for further information
    condition_list : :class:`~biopsykit.datatype_helper.SubjectConditionDict` or
    :class:`~biopsykit.datatype_helper.SubjectConditionDataFrame`
        dataframe or dictionary of group membership in standardized format

    Returns
    -------
    dict
        nested group dict with one 'Phase dict' per group

    """
    if is_subject_condition_dataframe(condition_list, raise_exception=False):
        condition_list = condition_list.groupby("condition").groups
    is_subject_condition_dict(condition_list)
    return {
        condition: {key: df[condition_list[condition]] for key, df in phase_dict.items()}
        for condition in condition_list.keys()
    }


def param_subphases(
    ecg_processor: Optional["EcgProcessor"] = None,
    dict_ecg: Optional[Dict[str, pd.DataFrame]] = None,
    dict_rpeaks: Optional[Dict[str, pd.DataFrame]] = None,
    subphases: Optional[Sequence[str]] = None,
    subphase_durations: Optional[Sequence[int]] = None,
    param_types: Optional[Union[str, Sequence[str]]] = "all",
    sampling_rate: Optional[int] = 256,
    include_total: Optional[bool] = True,
    title: Optional[str] = None,
) -> pd.DataFrame:
    """
    Computes specified parameters (HRV / RSA / ...) over phases and subphases **for one subject**.

    To use this function, either simply pass an ``EcgProcessor`` object or two dictionaries
    ``dict_ecg`` and ``dict_rpeaks`` resulting from ``EcgProcessor.ecg_process()``.

    Parameters
    ----------
    ecg_processor : EcgProcessor, optional
        `EcgProcessor` object
    dict_ecg : dict, optional
        dict with dataframes of processed ECG signals. Output from `EcgProcessor.ecg_process()`.
    dict_rpeaks : dict, optional
        dict with dataframes of processed R peaks. Output from `EcgProcessor.ecg_process()`.
    subphases : list of int
        list of subphase names
    subphase_durations : list of str
        list of subphase durations
    param_types : list or str, optional
        list with parameter types to compute or 'all' to compute all available parameters. Choose from a subset of
        ['hrv', 'rsa'] to compute HRV and RSA parameters, respectively.
    sampling_rate : float, optional
        Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz
    include_total : bool, optional
        ``True`` to also compute parameters over the complete phases (in addition to only over subphases),
        ``False`` to only compute parameters over the single subphases. Default: ``True``
    title : str, optional
        Optional title of the processing progress bar. Default: ``None``

    Returns
    -------
    pd.DataFrame
        dataframe with computed parameters over the single subphases
    """
    import biopsykit.signals.ecg as ecg

    if ecg_processor is None and dict_rpeaks is None and dict_ecg is None:
        raise ValueError("Either `ecg_processor` or `dict_rpeaks` and `dict_ecg` must be passed as arguments!")

    if subphases is None or subphase_durations is None:
        raise ValueError("Both `subphases` and `subphase_durations` are required as parameter!")

    # get all desired parameter types
    possible_param_types = {
        "hrv": ecg.EcgProcessor.hrv_process,
        "rsp": ecg.EcgProcessor.rsp_rsa_process,
    }
    if param_types == "all":
        param_types = possible_param_types

    if isinstance(param_types, str):
        param_types = {param_types: possible_param_types[param_types]}
    if not all([param in possible_param_types for param in param_types]):
        raise ValueError(
            "`param_types` must all be of {}, not {}".format(possible_param_types.keys(), param_types.keys())
        )

    param_types = {param: possible_param_types[param] for param in param_types}

    if ecg_processor:
        sampling_rate = ecg_processor.sampling_rate
        dict_rpeaks = ecg_processor.rpeaks
        dict_ecg = ecg_processor.ecg_result

    if "rsp" in param_types and dict_ecg is None:
        raise ValueError("`dict_ecg` must be passed if param_type is {}!".format(param_types))

    index_name = "subphase"
    # dict to store results. one entry per parameter and a list of dataframes per MIST phase
    # that will later be concated to one large dataframes
    dict_df_subphases = {param: list() for param in param_types}

    # iterate through all phases in the data
    for (phase, rpeaks), (ecg_phase, ecg_data) in tqdm(zip(dict_rpeaks.items(), dict_ecg.items()), desc=title):
        rpeaks = rpeaks.copy()
        ecg_data = ecg_data.copy()

        # dict to store intermediate results of subphases. one entry per parameter with a
        # list of dataframes per subphase that will later be concated to one dataframe per MIST phase
        dict_subphases = {param: list() for param in param_types}
        if include_total:
            # compute HRV, RSP over complete phase
            for param_type, param_func in param_types.items():
                dict_subphases[param_type].append(
                    param_func(
                        ecg_signal=ecg_data,
                        rpeaks=rpeaks,
                        index="Total",
                        index_name=index_name,
                        sampling_rate=sampling_rate,
                    )
                )

        if phase not in ["Part1", "Part2"]:
            # skip Part1, Part2 for subphase parameter analysis (parameters in total are computed above)
            for subph, dur in zip(subphases, subphase_durations):
                # get the first xx seconds of data (i.e., get only the current subphase)
                if dur > 0:
                    df_subph_rpeaks = rpeaks.first("{}S".format(dur))
                else:
                    # duration of 0 seconds = Feedback Interval, don't cut the beginning,
                    # use all remaining data
                    df_subph_rpeaks = rpeaks
                # ECG does not need to be sliced because rpeaks are already sliced and
                # will select only the relevant ECG signal parts anyways
                df_subph_ecg = ecg_data

                for param_type, param_func in param_types.items():
                    # compute HRV, RSP over subphases
                    dict_subphases[param_type].append(
                        param_func(
                            ecg_signal=df_subph_ecg,
                            rpeaks=df_subph_rpeaks,
                            index=subph,
                            index_name=index_name,
                            sampling_rate=sampling_rate,
                        )
                    )

                # remove the currently analyzed subphase of data
                # (so that the next subphase is first in the next iteration)
                rpeaks = rpeaks.drop(df_subph_rpeaks.index)

        for param in dict_subphases:
            # concat dataframe of all subphases to one dataframe per MIST phase and add to parameter dict
            dict_df_subphases[param].append(pd.concat(dict_subphases[param]))

    # concat all dataframes together to one big result dataframes
    return pd.concat(
        [pd.concat(dict_df, keys=dict_rpeaks.keys(), names=["phase"]) for dict_df in dict_df_subphases.values()],
        axis=1,
    )


def mean_per_subject_nested_dict(
    data: Dict[str, Dict[str, pd.DataFrame]],
    param_name: str,
    is_group_dict: Optional[bool] = False,
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data: nested dictionary
    param_name : str
        Name of the parameter to compute mean from. Corresponds to the column name of the resulting dataframe
    is_group_dict: bool, optional

    Returns
    -------
        dataframe
    """
    df_result = _mean_per_subject_nested_dict(data, param_name)

    # name index levels correctly (depending on number of levels)
    nlevels = df_result.index.nlevels
    if is_group_dict:
        nlevels -= 1

    index_names = ["phase"]
    if nlevels == 3:
        index_names = ["phase", "subphase"]

    if is_group_dict:
        index_names = ["condition"] + index_names

    index_names = index_names + ["subject"]
    df_result.index.names = index_names

    # last index level is always 'subject' => reorder index levels to that 'subject' is the first
    # (except when 'condition' index is present, then this is the first, 'subject' is second)
    index_names = [index_names[-1]] + index_names[:-1]
    if "condition" in index_names:
        index_names.remove("condition")
        index_names = ["condition"] + index_names
    return df_result.reorder_levels(index_names).sort_index()


def _mean_per_subject_nested_dict(data: Dict[str, Dict[str, pd.DataFrame]], param_name: str) -> pd.DataFrame:
    result_data = {}

    for phase, data_phase in data.items():
        if isinstance(data_phase, dict):
            # nested dictionary
            result_data[phase] = _mean_per_subject_nested_dict(data_phase, param_name)
        elif isinstance(data_phase, pd.DataFrame):
            df = pd.DataFrame(data_phase.mean(), columns=[param_name])
            result_data[phase] = df
        else:
            raise ValueError("Invalid input!")

    return pd.concat(result_data)


def mean_se_nested_dict(
    data: Dict[str, Dict[str, pd.DataFrame]],
    subphases: Optional[Sequence[str]] = None,
    is_group_dict: Optional[bool] = False,
    std_type: Optional[str] = "se",
) -> pd.DataFrame:
    """
    Computes mean and standard error (se) or standard deviation (std) for a nested dictionary.

    As input either
    (a) a 'Subject dict' (e.g. like returned from bp.signals.ecg.io.load_combine_hr_all_subjects()),
    (b) a 'Subphase dict' (for only one group), or
    (c) a dict of 'Subphase dict', one dict per group (for multiple groups, if ``is_group_dict`` is ``True``)
    can be passed (see ``utils.split_subphases`` for more explanation). Both dictionaries are outputs from
    ``utils.split_subphases``.

    The input dict structure is expected to look like one of these examples:
        (a) { "<Subject>" : { "<Phase>" : dataframe with values } }
        (b) { "<Phase>" : dataframe with values, 1 subject per column }
        (c) { "<Phase>" : { "<Subphase>" : dataframe with values, 1 subject per column } }
        (d) { "<Group>" : { "<Phase>" : dataframe with values } }
        (e) { "<Group>" : { "<Phase>" : { "<Subphase>" : dataframe with values, 1 subject per column } } }

    The output is a 'mse dataframe', a pandas dataframe with:
        * columns: ['mean', 'se'] for mean and standard error or ['mean', 'std'] for mean and standard deviation
        * rows: MultiIndex with level 0 = Condition, level 1 = Phases and level 2 = Subphases (depending on input).

    Parameters
    ----------
    data : dict
        nested dictionary containing data to be reduced (e.g. heart rate)
    subphases : list, optional
        list of subphase names or ``None`` to use default subphase names. Default: ``None``
    is_group_dict : boolean, optional
        ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
        Default: ``False``
    std_type : str, optional
        'std' to compute standard deviation, 'se' to compute standard error. Default: 'se'

    Returns
    -------
    pd.DataFrame
        'mse dataframe' with MultiIndex (order: condition, phase, subphase, depending on input data)

    Examples
    --------
    >>> from biopsykit.utils.data_processing import mean_se_nested_dict
    >>> # Example (a): Nested dictionary with outer-keys = Subjects, inner-keys = Phases, inner-values = pandas dataframe with 1 column
    >>> # Construct dictionary (as example)
    >>> dict_subject = {
    >>>     'Vp_01': {
    >>>         'Phase1': pd.DataFrame([1, 2, 3, 4, 5]),
    >>>         'Phase2': pd.DataFrame([6, 7, 8, 9, 10]),
    >>>         'Phase3': pd.DataFrame([11, 12, 13, 14, 15])
    >>>     },
    >>>     'Vp_02': {
    >>>         'Phase1': pd.DataFrame([1, 2, 3, 4, 5]),
    >>>         'Phase2': pd.DataFrame([6, 7, 8, 9, 10]),
    >>>         'Phase3': pd.DataFrame([11, 12, 13, 14, 15])
    >>>     },
    >>>     # ...
    >>> }
    >>> df_mse = mean_se_nested_dict(dict_subject)
    >>> print(df_mse)
    >>> # Output = DataFrame with
    >>> #   Row Index:       [Phase1, Phase2, Phase3]
    >>> #   Column Index:    [mean, se]
    >>>
    >>>
    >>> # Example (b): Nested dictionary with outer-keys = Phases, inner-keys = Subphases, inner-values = pandas dataframe with multiple columns, 1 column per subject
    >>> # Construct dictionary (as example)
    >>> dict_subject = {
    >>>     'Phase1': {
    >>>         'Subphase1': pd.DataFrame({'Vp_01': [1, 2, 3, 4, 5], 'Vp_02': [6, 7, 8, 9, 10], 'Vp_03': [11, 12, 13, 14, 15]}),
    >>>         'Subphase2': pd.DataFrame({'Vp_01': [1, 2, 3, 4, 5], 'Vp_02': [6, 7, 8, 9, 10], 'Vp_03': [11, 12, 13, 14, 15]}),
    >>>         'Subphase3': pd.DataFrame({'Vp_01': [1, 2, 3, 4, 5], 'Vp_02': [6, 7, 8, 9, 10], 'Vp_03': [11, 12, 13, 14, 15]})
    >>>     },
    >>>     'Phase2': {
    >>>         'Subphase1': pd.DataFrame({'Vp_01': [1, 2, 3, 4, 5], 'Vp_02': [6, 7, 8, 9, 10], 'Vp_03': [11, 12, 13, 14, 15]}),
    >>>         'Subphase2': pd.DataFrame({'Vp_01': [1, 2, 3, 4, 5], 'Vp_02': [6, 7, 8, 9, 10], 'Vp_03': [11, 12, 13, 14, 15]}),
    >>>         'Subphase3': pd.DataFrame({'Vp_01': [1, 2, 3, 4, 5], 'Vp_02': [6, 7, 8, 9, 10], 'Vp_03': [11, 12, 13, 14, 15]})
    >>>     },
    >>>     # ...
    >>> }
    >>> df_mse = mean_se_nested_dict(dict_subject)
    >>> print(df_mse)
    >>> # Output = DataFrame with
    >>> #   Row Index: MultiIndex with 1st level = Phases, 2nd level = Subphases
    >>> #   Column Index:    [mean, se]

    """

    # TODO improve interface: automatically check for groups, phases, subphases, always return a dataframe (instead of a dict when data for different groups is passed), name index levels

    if std_type not in ["std", "se"]:
        raise ValueError("Invalid argument for 'std_type'! Must be one of {}, not {}.".format(["std", "se"], std_type))

    if is_group_dict:
        # return pd.concat({group: mean_se_nested_dict(dict_group, subphases, std_type=std_type) for group, dict_group in
        #                           data.items()})
        return {
            group: mean_se_nested_dict(dict_group, subphases, std_type=std_type) for group, dict_group in data.items()
        }
    else:
        if subphases is None:
            # compute mean value per nested dictionary entry
            dict_mean = {}
            for key, dict_val in data.items():
                if isinstance(dict_val, dict):
                    # passed dict was case (c) or (e) explained in docstring
                    dict_mean[key] = pd.DataFrame({subkey: dict_val[subkey].mean() for subkey in dict_val})
                else:
                    # passed dict was case (d) explained in docstring
                    dict_mean[key] = dict_val.mean()
        else:
            dict_mean = {
                key: pd.DataFrame({subph: dict_val[subph].mean() for subph in subphases})
                for key, dict_val in data.items()
            }

        if (np.array([len(df) for df in dict_mean.values()]) == 1).all():
            # Dataframes with one row => concat on this axis
            df_mean = pd.concat(dict_mean)
        else:
            df_mean = pd.concat(dict_mean.values(), axis=1, keys=dict_mean.keys())

        if isinstance(df_mean.index, pd.MultiIndex):
            # if resulting index is a MultiIndex drop the second index level because it's redundant
            df_mean.index = df_mean.index.droplevel(1)

        if std_type == "se":
            return pd.concat(
                [df_mean.mean(), df_mean.std() / np.sqrt(df_mean.shape[0])],
                axis=1,
                keys=["mean", "se"],
            )
        else:
            return pd.concat([df_mean.mean(), df_mean.std()], axis=1, keys=["mean", "std"])
