from typing import Union, Tuple, Optional, Dict, Sequence

import pandas as pd
import numpy as np
import neurokit2 as nk
from numpy.lib.stride_tricks import as_strided

from scipy import signal
from scipy import interpolate

from tqdm.notebook import tqdm

from gaitmap.utils.array_handling import sliding_window_view

import biopsykit.utils as utils


def interpolate_sec(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Interpolates input data to a frequency of 1 Hz.

    *Note*: This function requires the index of the dataframe or series to be a datetime index.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        data to interpolate. Index of data needs to be 'pd.DateTimeIndex'

    Returns
    -------
    pd.DataFrame
        dataframe with data interpolated to seconds


    Raises
    ------
    ValueError
        if no dataframe or series is passed, or if the dataframe/series has no datetime index

    """

    from scipy import interpolate
    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    elif isinstance(data, pd.Series):
        column_name = [data.name]
    else:
        raise ValueError("Only 'pd.DataFrame' or 'pd.Series' allowed as input!")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index of data needs to be 'pd.DateTimeIndex'!")

    x_old = np.array((data.index - data.index[0]).total_seconds())
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = utils.sanitize_input_1d(data)
    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    x_new = pd.Index(x_new, name="Time")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=column_name)


def interpolate_dict_sec(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Interpolates all data in the dictionary to 1Hz data (see `interpolate_sec` for further information).

    Parameters
    ----------
    data_dict : dict
        nested data dict with heart rate data

    Returns
    -------
    dict
        nested data dict with heart rate data interpolated to seconds
    """

    return {
        subject_id: {
            phase: interpolate_sec(df_hr) for phase, df_hr in dict_hr.items()
        } for subject_id, dict_hr in data_dict.items()
    }


def interpolate_and_cut(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_dict = interpolate_dict_sec(data_dict)
    durations = np.array([[len(df) for phase, df in dict_hr.items()] for dict_hr in data_dict.values()])
    phase_names = np.array([np.array(list(val.keys())) for key, val in data_dict.items()])
    if (phase_names[0] == phase_names).all():
        phase_names = phase_names[0]
    else:
        raise ValueError("Phases are not the same for all subjects!")

    # minimal duration of each Phase
    min_dur = {phase: dur for phase, dur in zip(phase_names, np.min(durations, axis=0))}

    for subject_id, dict_hr in data_dict.items():
        dict_cut = {}
        for phase in phase_names:
            dict_cut[phase] = dict_hr[phase][0:min_dur[phase]]
        data_dict[subject_id] = dict_cut

    return data_dict


def concat_phase_dict(dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
                      phases: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """
    Rearranges a 'HR subject dict' (see ``utils.load_hr_excel_all_subjects``) into a 'Phase dict', i.e. a dictionary
    with one dataframe per Phase where each dataframe contains column-wise HR data for all subjects.

    The **output** format will be the following:

    { "<Phase>" : hr_dataframe, 1 subject per column }

    E.g., see ``biopsykit.protocols.mist.MIST.concat_phase_dict()`` for further information.

    Parameters
    ----------
    dict_hr_subject : dict
        'HR subject dict', i.e. a nested dict with heart rate data per phase and subject
    phases : list
        list of phase names

    Returns
    -------
    dict
        'Phase dict', i.e. a dict with heart rate data of all subjects per phase

    """

    dict_phase: Dict[str, pd.DataFrame] = {key: pd.DataFrame(columns=list(dict_hr_subject.keys())) for key in
                                           phases}
    for subj in dict_hr_subject:
        dict_bl = dict_hr_subject[subj]
        for phase in phases:
            dict_phase[phase][subj] = dict_bl[phase]['ECG_Rate']

    return dict_phase


def split_subphases(
        data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
        subphase_names: Sequence[str],
        subphase_times: Sequence[Tuple[int, int]],
        is_group_dict: Optional[bool] = False
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
        List with start and end times of each subphase in seconds
    is_group_dict : bool, optional
        ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``

    Returns
    -------
    dict
        'Subphase dict' with course of data per Phase, Subphase and Subject, respectively or
        nested dict of 'Subphase dicts' if `is_group_dict` is ``True``

    """
    if is_group_dict:
        # recursively call this function for each group
        return {
            group: split_subphases(
                dict_group, subphase_names=subphase_names, subphase_times=subphase_times
            ) for group, dict_group in data.items()
        }
    else:
        phase_dict = {}
        # split data into subphases for each Phase
        for phase, df in data.items():
            phase_dict[phase] = {subph: df[start:end] for subph, (start, end) in
                                 zip(subphase_names, subphase_times)}
        return phase_dict


def split_groups(phase_dict: Dict[str, pd.DataFrame],
                 condition_dict: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Splits 'Phase dict' into group dict, i.e. one 'Phase dict' per group.

    Parameters
    ----------
    phase_dict : dict
        'Phase dict' to be split in groups. See ``bp.signals.utils.concat_phase_dict`` for further information
    condition_dict : dict
        dictionary of group membership. Keys are the different groups, values are lists of subject IDs that belong
        to the respective group

    Returns
    -------
    dict
        nested group dict with one 'Phase dict' per group

    """
    return {
        condition: {key: df[condition_dict[condition]] for key, df in phase_dict.items()} for condition
        in condition_dict.keys()
    }


def param_subphases(
        ecg_processor: Optional['EcgProcessor'] = None,
        dict_ecg: Optional[Dict[str, pd.DataFrame]] = None,
        dict_rpeaks: Optional[Dict[str, pd.DataFrame]] = None,
        subphases: Optional[Sequence[str]] = None,
        subphase_durations: Optional[Sequence[int]] = None,
        param_types: Optional[Union[str, Sequence[str]]] = 'all',
        sampling_rate: Optional[int] = 256, include_total: Optional[bool] = True,
        title: Optional[str] = None
) -> pd.DataFrame:
    """
    Computes specified parameters (HRV / RSA / ...) over phases and subphases.

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
    possible_param_types = {'hrv': ecg.EcgProcessor.hrv_process, 'rsp': ecg.EcgProcessor.rsp_rsa_process}
    if param_types == 'all':
        param_types = possible_param_types

    if isinstance(param_types, str):
        param_types = {param_types: possible_param_types[param_types]}
    if not all([param in possible_param_types for param in param_types]):
        raise ValueError(
            "`param_types` must all be of {}, not {}".format(possible_param_types.keys(), param_types.keys()))

    param_types = {param: possible_param_types[param] for param in param_types}

    if ecg_processor:
        sampling_rate = ecg_processor.sampling_rate
        dict_rpeaks = ecg_processor.rpeaks
        dict_ecg = ecg_processor.ecg_result

    if 'rsp' in param_types and dict_ecg is None:
        raise ValueError("`dict_ecg` must be passed if param_type is {}!".format(param_types))

    index_name = "Subphase"
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
                    param_func(ecg_signal=ecg_data, rpeaks=rpeaks, index="Total", index_name=index_name,
                               sampling_rate=sampling_rate))

        if phase not in ["Part1", "Part2"]:
            # skip Part1, Part2 for subphase parameter analysis (parameters in total are computed above)
            for subph, dur in zip(subphases, subphase_durations):
                # get the first xx seconds of data (i.e., get only the current subphase)
                if dur > 0:
                    df_subph_rpeaks = rpeaks.first('{}S'.format(dur))
                else:
                    # duration of 0 seconds = Feedback Interval, don't cut slice the beginning,
                    # use all remaining data
                    df_subph_rpeaks = rpeaks
                # ECG does not need to be sliced because rpeaks are already sliced and
                # will select only the relevant ECG signal parts anyways
                df_subph_ecg = ecg_data

                for param_type, param_func in param_types.items():
                    # compute HRV, RSP over subphases
                    dict_subphases[param_type].append(
                        param_func(ecg_signal=df_subph_ecg, rpeaks=df_subph_rpeaks, index=subph,
                                   index_name=index_name,
                                   sampling_rate=sampling_rate))

                # remove the currently analyzed subphase of data
                # (so that the next subphase is first in the next iteration)
                rpeaks = rpeaks[~rpeaks.index.isin(df_subph_rpeaks.index)]

        for param in dict_subphases:
            # concat dataframe of all subphases to one dataframe per MIST phase and add to parameter dict
            dict_df_subphases[param].append(pd.concat(dict_subphases[param]))

    # concat all dataframes together to one big result dataframes
    return pd.concat(
        [pd.concat(dict_df, keys=dict_rpeaks.keys(), names=["Phase"]) for dict_df in dict_df_subphases.values()],
        axis=1)


def mean_se_nested_dict(
        data: Dict[str, Dict[str, pd.DataFrame]],
        subphases: Optional[Sequence[str]] = None,
        is_group_dict: Optional[bool] = False,
        std_type: Optional[str] = 'se'
) -> pd.DataFrame:
    """
    Computes mean and standard error (se) or standard deviation (std) for a nested dictionary.

    As input either
    (a) a 'Subject dict' (e.g. like returned from bp.signals.ecg.io.load_combine_hr_all_subjects()),
    (b) a 'Subphase dict' (for only one group), or
    (c) a dict of 'Subphase dict', one dict per group (for multiple groups, if ``is_group_dict`` is ``True``)
    can be passed (see ``utils.split        _subphases`` for more explanation). Both dictionaries are outputs from
    ``utils.split_subphases``.

    The input dict structure is expected to look like one of these examples:
        (a) { "<Subject>" : { "<Phase>" : dataframe with values } }
        (b) { "<Phase>" : { "<Subphase>" : dataframe with values, 1 subject per column } }
        (c) { "<Group>" : { "<Phase>" : { "<Subphase>" : dataframe with values, 1 subject per column } } }

    The output is a 'mse dataframe' (or a dict of such, in case of multiple groups), a pandas dataframe with:
        * columns: ['mean', 'se'] for mean and standard error or ['mean', 'std] for mean and standard deviation
        * rows: MultiIndex with level 0 = Phases and level 1 = Subphases.

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
    dict or pd.DataFrame
        'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.

    Examples
    --------
    >>> import biopsykit.signals.su as su
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
    >>> df_mse = su.mean_se_nested_dict(dict_subject)
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
    >>> df_mse = su.mean_se_nested_dict(dict_subject)
    >>> print(df_mse)
    >>> # Output = DataFrame with
    >>> #   Row Index: MultiIndex with 1st level = Phases, 2nd level = Subphases
    >>> #   Column Index:    [mean, se]

    """

    if std_type not in ['std', 'se']:
        raise ValueError("Invalid argument for 'std_type'! Must be one of {}, not {}.".format(['std', 'se'], std_type))

    if is_group_dict:
        return {group: mean_se_nested_dict(dict_group, subphases, std_type=std_type) for group, dict_group in
                data.items()}
    else:
        if subphases is None:
            # compute mean value per nested dictionary entry
            dict_mean = {key: pd.DataFrame({subkey: dict_val[subkey].mean() for subkey in dict_val})
                         for key, dict_val in data.items()}
        else:
            dict_mean = {key: pd.DataFrame({subph: dict_val[subph].mean() for subph in subphases})
                         for key, dict_val in data.items()}

        if (np.array([len(df) for df in dict_mean.values()]) == 1).all():
            # Dataframes with one row => concat on this axis
            df_mean = pd.concat(dict_mean)
        else:
            df_mean = pd.concat(dict_mean.values(), axis=1, keys=dict_mean.keys())

        if isinstance(df_mean.index, pd.MultiIndex):
            # if resulting index is a MultiIndex drop the second index level because it's redundant
            df_mean.index = df_mean.index.droplevel(1)
        if std_type == 'se':
            return pd.concat([df_mean.mean(), df_mean.std() / np.sqrt(df_mean.shape[0])], axis=1, keys=["mean", "se"])
        else:
            return pd.concat([df_mean.mean(), df_mean.std()], axis=1, keys=["mean", "std"])


def find_extrema_in_radius(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                           indices: Union[pd.DataFrame, pd.Series, np.ndarray], radius: Union[int, Tuple[int, int]],
                           extrema_type: Optional[str] = "min") -> np.ndarray:
    """
    Finds extrema values (min or max) within a given radius.

    Parameters
    ----------
    data : array_like
        input data
    indices : array_like
        array with indices for which to search for extrema values around
    radius: int or tuple of int
        radius around `indices` to search for extrema. If `radius` is an ``int`` then search for extrema equally
        in both directions in the interval [index - radius, index + radius].
        If `radius` is a ``tuple`` then search for extrema in the interval [ index - radius[0], index + radius[1] ]
    extrema_type : {'min', 'max'}, optional
        extrema type to be searched for. Default: 'min'

    Returns
    -------
    array_like
        array containing the indices of the found extrema values in the given radius around `indices`.
        Has the same length as `indices`.

    Examples
    --------
    >>> import biopsykit.signal as signal
    >>> data = pd.read_csv("data.csv")
    >>> indices = np.array([16, 25, 40, 57, 86, 100])
    >>>
    >>> radius = 4
    >>> # search minima in 'data' in a 4 sample 'radius' around each entry of 'indices'
    >>> signal.find_extrema_in_radius(data, indices, radius)
    >>>
    >>> radius = (5, 0)
    >>> # search maxima in 'data' in a 5 samples before each entry of 'indices'
    >>> signal.find_extrema_in_radius(data, indices, radius, extrema_type='max')
    """
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}

    if extrema_type not in extrema_funcs:
        raise ValueError("`extrema_type` must be one of {}, not {}".format(list(extrema_funcs.keys()), extrema_type))
    extrema_func = extrema_funcs[extrema_type]

    # ensure numpy
    data = utils.sanitize_input_1d(data)
    indices = utils.sanitize_input_1d(indices)
    indices = indices.astype(int)
    # possible start offset if beginning of array needs to be padded to ensure radius
    start_padding = 0

    # determine upper and lower limit
    if isinstance(radius, tuple):
        lower_limit = radius[0]
    else:
        lower_limit = radius
    if isinstance(radius, tuple):
        upper_limit = radius[-1]
    else:
        upper_limit = radius

    # round up and make sure it's an integer
    lower_limit = np.ceil(lower_limit).astype(int)
    upper_limit = np.ceil(upper_limit).astype(int)

    # pad end/start of array if last_index+radius/first_index-radius is longer/shorter than array
    if len(data) - np.max(indices) <= upper_limit:
        data = np.pad(data, (0, upper_limit), constant_values=np.nan)
    if np.min(indices) < lower_limit:
        start_padding = lower_limit
        data = np.pad(data, (lower_limit, 0), constant_values=np.nan)

    # initialize window array
    windows = np.zeros(shape=(len(indices), lower_limit + upper_limit + 1))
    for i, index in enumerate(indices):
        # get windows around index
        windows[i] = data[index - lower_limit + start_padding:index + upper_limit + start_padding + 1]

    return extrema_func(windows, axis=1) + indices - lower_limit


def remove_outlier_and_interpolate(data: np.ndarray, outlier_mask: np.ndarray, x_old: Optional[np.ndarray] = None,
                                   desired_length: Optional[int] = None) -> np.ndarray:
    """
    Sets all detected outlier to nan, imputes them by linear interpolation their neighbors and interpolates
    the resulting values to a desired length.

    Parameters
    ----------
    data : array_like
        input data
    outlier_mask: array_like
        outlier mask. has to be the same length like `data`. ``True`` entries indicate outliers
    x_old : array_like, optional
        x values of the input data to interpolate or ``None`` if no interpolation should be performed. Default: ``None``
    desired_length : int, optional
        desired length of the output signal or ``None`` to keep input length. Default: ``None``

    Returns
    -------
    np.array
        Outlier-removed and interpolated data

    Raises
    ------
    ValueError
        if `data` and `outlier_mask` don't have the same length or if `x_old` is ``None`` when `desired_length`
        is passed as parameter

    """
    # ensure numpy
    data = np.array(data)
    outlier_mask = np.array(outlier_mask)

    if len(data) != len(outlier_mask):
        raise ValueError("`data` and `outlier_mask` need to have same length!")
    if x_old is None and desired_length:
        raise ValueError("`x_old` must also be passed when `desired_length` is passed!")
    x_old = np.array(x_old)

    # remove outlier
    data[outlier_mask] = np.nan
    # fill outlier by linear interpolation of neighbors
    data = pd.Series(data).interpolate(limit_direction='both').values
    # interpolate signal
    x_new = np.linspace(x_old[0], x_old[-1], desired_length)
    return nk.signal_interpolate(x_old, data, x_new, method='linear')


def sliding_window(
        data: Union[np.array, pd.Series, pd.DataFrame],
        window_samples: Optional[int] = None,
        window_sec: Optional[int] = None,
        sampling_rate: Optional[Union[int, float]] = 0,
        overlap_samples: Optional[int] = None,
        overlap_percent: Optional[float] = None
):
    # check input
    data = utils.sanitize_input_nd(data, ncols=(1, 3))

    window, overlap = sanitize_sliding_window_input(
        window_samples=window_samples,
        window_sec=window_sec,
        sampling_rate=sampling_rate,
        overlap_samples=overlap_samples,
        overlap_percent=overlap_percent
    )

    return sliding_window_view(data, window_length=window, overlap=overlap, nan_padding=True)


def sanitize_sliding_window_input(
        window_samples: Optional[int] = None,
        window_sec: Optional[int] = None,
        sampling_rate: Optional[Union[int, float]] = 0,
        overlap_samples: Optional[int] = None,
        overlap_percent: Optional[float] = None
) -> Tuple[int, int]:
    if all([x is None for x in (window_samples, window_sec)]):
        raise ValueError("Either `window_samples` or `window_sec` in combination with "
                         "`sampling_rate` must be supplied as parameter!")

    if window_samples is None:
        if sampling_rate == 0:
            raise ValueError("Sampling rate must be specified when `window_sec` is used!")
        window = int(sampling_rate * window_sec)
    else:
        window = window_samples

    if overlap_samples is not None:
        overlap = overlap_samples
    elif overlap_percent is not None:
        overlap = int(overlap_percent * window)
    else:
        overlap = window - 1

    return window, overlap


def downsample(data: np.ndarray, sampling_rate: Union[int, float],
               final_sampling_rate: Union[int, float]) -> np.ndarray:
    if (sampling_rate / final_sampling_rate) % 1 == 0:
        return signal.decimate(data, int(sampling_rate / final_sampling_rate), axis=0)
    else:
        # aliasing filter
        b, a = signal.cheby1(N=8, rp=0.05, Wn=0.8 / (sampling_rate / final_sampling_rate))
        data_lp = signal.filtfilt(a=a, b=b, x=data)
        # interpolation
        x_old = np.linspace(0, len(data_lp), num=len(data_lp), endpoint=False)
        x_new = np.linspace(0, len(data_lp), num=int(len(data_lp) / (sampling_rate / final_sampling_rate)),
                            endpoint=False)
        interpol = interpolate.interp1d(x=x_old, y=data_lp)
        return interpol(x_new)


def check_ecg_input(ecg_processor: 'EcgProcessor', key: str, ecg_signal: pd.DataFrame, rpeaks: pd.DataFrame) -> bool:
    """
    Checks valid input, i.e. if either `ecg_processor` **and** `key` are supplied as arguments *or* `ecg_signal` **and**
    `rpeaks`. Used as helper method for several functions.

    Parameters
    ----------
    ecg_processor : EcgProcessor
        `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
    key : str
        Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
    ecg_signal : str
        dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
    rpeaks : str
        dataframe with R peaks. Output of `EcgProcessor.ecg_process()`

    Returns
    -------
    ``True`` if correct input was supplied, raises ValueError otherwise

    Raises
    ------
    ValueError
        if invalid input supplied
    """

    if all([x is None for x in [ecg_processor, key, ecg_signal, rpeaks]]):
        raise ValueError(
            "Either `ecg_processor` and `key` or `rpeaks` and `ecg_signal` must be passed as arguments!")
    if ecg_processor:
        if key is None:
            raise ValueError("`key` must be passed as argument when `ecg_processor` is passed!")

    return True
