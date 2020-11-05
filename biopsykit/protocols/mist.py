from typing import Dict, Tuple, Union, Optional, Sequence

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import biopsykit.signals.ecg as ecg
from biopsykit.signals.ecg import EcgProcessor
import biopsykit.utils as utils

mist_params = {
    # MIST Phases
    'phases': ["MIST1", "MIST2", "MIST3"],
    # MIST Subphases
    'subphases': ['BL', 'AT', 'FB'],
    # duration of subphases BL, AT in seconds
    'subphases.duration': [60, 240],
}

hr_ensemble_params = {
    'colormap': utils.cmap_fau_blue('3'),
    'line_styles': ['-', '--', ':'],
    'background.color': ['#e0e0e0', '#9e9e9e', '#757575'],
    'background.alpha': [0.6, 0.7, 0.7],
}

hr_course_params = {
    'colormap': utils.cmap_fau_blue('2_lp'),
    'line_styles': ['-', '--'],
    'markers': ['o', 'P'],
    'background.color': ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
    'background.alpha': [0.6, 0.7, 0.7],
    'x_offsets': [0, 0.05]
}


def cut_feedback_interval(
        dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Cuts heart rate data of each subject to equal length, i.e. to the minimal duration of each MIST phase
    (due to variable length of the Feedback interval).

    Parameters
    ----------
    dict_hr_subject : dict
        nested dict containing heart rate data per subject and MIST phase
        ('HR subject dict', see `utils.load_hr_excel_all_subjects`)

    Returns
    -------
    dict
        'HR subject dict' containing only relevant MIST phases (i.e., excluding Part1 and Part2)
        where each MIST phase is cut to the minimum duration of all subjects

    """
    # skip Part1 and Part2, extract only MIST Phases
    mist_phases = mist_params['phases']

    durations = np.array([[len(df) for phase, df in dict_hr.items() if phase not in ['Part1', 'Part2']] for dict_hr in
                          dict_hr_subject.values()])

    # minimal duration of each MIST Phase
    min_dur = {phase: dur for phase, dur in zip(mist_phases, np.min(durations, axis=0))}

    for subject_id, dict_hr in dict_hr_subject.items():
        dict_hr_cut = {}
        for phase in mist_phases:
            dict_hr_cut[phase] = dict_hr[phase][0:min_dur[phase]]
        dict_hr_subject[subject_id] = dict_hr_cut

    return dict_hr_subject


def concat_dicts(dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Rearranges the 'HR subject dict' (see `utils.load_hr_excel_all_subjects`) into 'MIST dict', i.e. a dictionary with
    one dataframe per MIST phase where each dataframe contains column-wise HR data for all subjects.

    The **output** format will be the following:

    { <"MIST_Phase"> : hr_dataframe, 1 subject per column }

    Parameters
    ----------
    dict_hr_subject : dict
        'HR subject dict', i.e. a nested dict with heart rate data per MIST phase and subject

    Returns
    -------
    dict
        'MIST dict', i.e. a dict with heart rate data of all subjects per MIST phase

    """
    mist_phases = mist_params['phases']

    dict_mist = {key: pd.DataFrame(columns=list(dict_hr_subject.keys())) for key in mist_phases}
    for subj in dict_hr_subject:
        dict_bl = dict_hr_subject[subj]
        for phase in mist_phases:
            dict_mist[phase][subj] = dict_bl[phase]['ECG_Rate']

    return dict_mist


def split_subphases(data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
                    mist_times: Optional[Sequence[Tuple[int, int]]] = None,
                    is_group_dict: Optional[bool] = False) \
        -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Splits a `MIST dict` (or a dict of such, in case of multiple groups, see `mist.mist_concat_dicts`)
    into a `MIST subphase dict` (see below for further explanation).

    The **input** is a `MIST dict`, i.e. a dictionary with heart rate data per MIST phase in the following format:

    { <"MIST_Phase"> : HR_dataframe, 1 subject per column }

    If multiple groups are present, then the expected input is nested, i.e. a dict of 'MIST dicts',
    with one entry per group.

    The **output** is a `MIST subphase dict`, i.e. a nested dictionary with heart rate data per MIST subphase in the
    following format:

    { <"MIST_Phase"> : { <"MIST_Subphase"> : HR_dataframe, 1 subject per column } }

    If multiple groups are present, then the output is nested, i.e. a dict of 'MIST subphase dicts',
    with one entry per group.


    Parameters
    ----------
    data : dict
        'MIST dict' or nested dict of 'MIST dict' if `is_group_dict` is ``True``
    mist_times : list, optional
        List with start and end times of each MIST subphase, or ``None`` to infer start and end times from data
        (with default MIST subphase durations)
    is_group_dict : bool, optional
        ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``

    Returns
    -------
    dict
        'MIST subphase dict' with course of HR data per MIST phase, subphase and subject, respectively or
        nested dict of 'MIST subphase dicts' if `is_group_dict` is ``True``

    """
    if is_group_dict:
        # recursively call this function for each group
        return {group: split_subphases(dict_group) for group, dict_group in data.items()}
    else:
        if not mist_times:
            mist_times = get_mist_times(data)
        mist_dict = {}
        mist_subph = mist_params['subphases']
        # split data into subphases for each MIST phase
        for phase, df in data.items():
            mist_dict[phase] = {subph: df[start:end] for subph, (start, end) in zip(mist_subph, mist_times)}
        return mist_dict


def split_groups(mist_dict: Dict[str, pd.DataFrame],
                 condition_dict: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Splits 'MIST dict' into group dict, i.e. one 'MIST dict' per group.

    Parameters
    ----------
    mist_dict : dict
        'MIST dict' to be split in groups. See `mist.mist_split_subphases` for further information
    condition_dict : dict
        dictionary of group membership. Keys are the different groups, values are lists of subject IDs that belong
        to the respective group

    Returns
    -------
    dict
        nested group dict with one 'MIST dict' per group

    """
    return {
        condition: {key: df[condition_dict[condition]] for key, df in mist_dict.items()} for condition
        in condition_dict.keys()
    }


def param_subphases(ecg_processor: Optional[ecg.EcgProcessor] = None,
                    dict_ecg: Optional[Dict[str, pd.DataFrame]] = None,
                    dict_rpeaks: Optional[Dict[str, pd.DataFrame]] = None,
                    param_types: Optional[Union[str, Sequence[str]]] = 'all',
                    sampling_rate: Optional[int] = 256, include_total: Optional[bool] = True,
                    subphases: Optional[Sequence[str]] = None,
                    subphase_durations: Optional[Sequence[int]] = None,
                    title: Optional[str] = None) -> pd.DataFrame:
    """
    Computes specified parameters (HRV / RSA / ...) over all MIST phases and subphases.

    To use this function, either simply pass an `EcgProcessor` object or two dictionaries `dict_ecg` and `dict_rpeaks`
    resulting from `EcgProcessor.ecg_process()`.`

    Parameters
    ----------
    ecg_processor : EcgProcessor, optional
        `EcgProcessor` object
    dict_ecg : dict, optional
        dict with dataframes of processed ECG signals. Output from `EcgProcessor.ecg_process()`.
    dict_rpeaks : dict, optional
        dict with dataframes of processed R peaks. Output from `EcgProcessor.ecg_process()`.
    param_types : list or str, optional
        list with parameter types to compute or 'all' to compute all available parameters. Choose from a subset of
        ['hrv', 'rsa'] to compute HRV and RSA parameters, respectively.
    sampling_rate : float, optional
        Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz
    include_total : bool, optional
        ``True`` to also compute parameters over the complete MIST phases (in addition to only over subphases),
        ``False`` to only compute parameters over the single MIST subphases. Default: ``True``
    subphases : list, optional
        list with subphase names or ``None`` to use default subphases. Default: ``None``
    subphase_durations : list, optional
        list with subphase durations or ``None`` to use default subphase durations. Default: ``None``
    title : str, optional
        Optional title of the processing progress bar. Default: ``None``

    Returns
    -------
    pd.DataFrame
        dataframe with computed parameters over the single MIST subphases
    """

    if ecg_processor is None and dict_rpeaks is None and dict_ecg is None:
        raise ValueError("Either `ecg_processor` or `dict_rpeaks` and `dict_ecg` must be passed as arguments!")

    # get all desired parameter types
    possible_param_types = {'hrv': EcgProcessor.hrv_process, 'rsp': EcgProcessor.rsp_rsa_process}
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

    if not subphases:
        subphases = mist_params['subphases']
    if not subphase_durations:
        subphase_durations = mist_params['subphases.duration']

    index_name = "Subphase"
    # dict to store results. one entry per parameter and a list of dataframes per MIST phase
    # that will later be concated to one large dataframes
    dict_df_subphases = {param: list() for param in param_types}
    # iterate through all phases in the data
    for (phase, rpeaks), (ecg_phase, ecg) in tqdm(zip(dict_rpeaks.items(), dict_ecg.items()), desc=title):
        rpeaks = rpeaks.copy()
        ecg = ecg.copy()

        # dict to store intermediate results of subphases. one entry per parameter with a
        # list of dataframes per subphase that will later be concated to one dataframe per MIST phase
        dict_subphases = {param: list() for param in param_types}
        if include_total:
            # compute HRV, RSP over complete phase
            for param_type, param_func in param_types.items():
                dict_subphases[param_type].append(
                    param_func(ecg_signal=ecg, rpeaks=rpeaks, index="Total", index_name=index_name,
                               sampling_rate=sampling_rate))

        if phase not in ["Part1", "Part2"]:
            # skip Part1, Part2 for subphase parameter analysis (parameters in total are computed above)
            for subph, dur in zip(subphases, subphase_durations):
                # get the first xx seconds of data (i.e., get only the current subphase)
                # TODO change to mist.mist_get_times?
                df_subph_rpeaks = rpeaks.first('{}S'.format(dur))
                # ECG does not need to be sliced because rpeaks are already sliced and
                # will select only the relevant ECG signal parts anyways
                df_subph_ecg = ecg
                for param_type, param_func in param_types.items():
                    # compute HRV, RSP over subphases
                    dict_subphases[param_type].append(
                        param_func(ecg_signal=df_subph_ecg, rpeaks=df_subph_rpeaks, index=subph, index_name=index_name,
                                   sampling_rate=sampling_rate))

                # remove the currently analyzed subphase of data
                # (so that the next subphase is first in the next iteration)
                rpeaks = rpeaks[~rpeaks.index.isin(df_subph_rpeaks.index)]

            if len(subphase_durations) < len(subphases):
                # add Feedback Interval (= remaining time) if present
                for param_type, param_func in param_types.items():
                    dict_subphases[param_type].append(
                        param_func(ecg_signal=ecg, rpeaks=rpeaks, index=subphases[-1], index_name=index_name,
                                   sampling_rate=sampling_rate))

        for param in dict_subphases:
            # concat dataframe of all subphases to one dataframe per MIST phase and add to parameter dict
            dict_df_subphases[param].append(pd.concat(dict_subphases[param]))

    # concat all dataframes together to one big result dataframes
    return pd.concat(
        [pd.concat(dict_df, keys=dict_rpeaks.keys(), names=["Phase"]) for dict_df in dict_df_subphases.values()],
        axis=1)


def hr_course(data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
              subphases: Optional[Sequence[str]] = None,
              is_group_dict: Optional[bool] = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Computes the heart rate mean and standard error per MIST subphase over all subjects.

    As input either 1. a 'MIST subphase dict' (for only one group) or 2. a dict of 'MIST subphase dict', one dict per
    group (for multiple groups, if `is_group_dict` is ``True``) can be passed
    (see `mist.mist_split_subphases` for more explanation). Both dictionaries are outputs from
    `mist.mist_split_subphases`.

    The output is a 'mse dataframe' (or a dict of such, in case of multiple groups), a pandas dataframe with:
        * columns: ['mean', 'se'] for mean and standard error
        * rows: MultiIndex with level 0 = MIST phases and level 1 = MIST subphases.

    The dict structure should like the following:
        (a) { "<MIST_Phase>" : { "<MIST_Subphase>" : heart rate dataframe, 1 subject per column } }
        (b) { "<Group>" : { <"MIST_Phase"> : { "<MIST_Subphase>" : heart rate dataframe, 1 subject per column } } }

    Parameters
    ----------
    data : dict
        nested dictionary containing heart rate data.
    subphases : list, optional
        list of subphase names or ``None`` to use default subphase names. Default: ``None``
    is_group_dict : boolean, optional
        ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
        Default: ``False``

    Returns
    -------
    dict or pd.DataFrame
        'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.
    """

    if not subphases:
        subphases = mist_params['subphases']

    if is_group_dict:
        return {group: hr_course(dict_group, subphases) for group, dict_group in data.items()}
    else:
        mean_hr = {phase: pd.DataFrame({subph: df[subph].mean() for subph in subphases}) for phase, df in data.items()}
        df_mist = pd.concat(mean_hr.values(), axis=1, keys=mean_hr.keys())
        return pd.concat([df_mist.mean(), df_mist.std() / np.sqrt(df_mist.shape[0])], axis=1, keys=["mean", "se"])


def get_mist_times(mist_dur: Union[Sequence[int], Dict[str, pd.DataFrame]],
                   subph_dur: Optional[Sequence[int]] = None) -> Sequence[Tuple[int, int]]:
    """
    Computes the start and end times of each MIST subphase. It is assumed that all MIST subphases,
    except Feedback, have equal length for all MIST phases. The length of the Feedback subphase is computed as
    the maximum length of all MIST phases.

    To compute the MIST subphase durations either pass a list with total MIST durations per phase or a
    'MIST dict' (see `mist.mist_concat_dicts` for further explanation).
    The length of the dataframe then corresponds to the total MIST duration per phase.

    Parameters
    ----------
    mist_dur : list or dict
        if a list is passed then each entry of the list is the total duration of one MIST phase. If a dict is passed
        then if is assumed that it is a 'MIST dict'. The length of the dataframe then corresponds to the total
        duration of the MIST phase
    subph_dur : list, optional
        durations of MIST subphases, i.e., Baseline, Arithmetic Tasks, etc. or ``None`` to use default durations.
        Default: ``None`

    Returns
    -------
    list
        a list with tuples of MIST subphase start and end times (in seconds)
    """

    if subph_dur:
        subph_dur = np.array(subph_dur)
    else:
        subph_dur = np.array(mist_params['subphases.duration'])

    if isinstance(mist_dur, dict):
        mist_dur = np.array([len(v) for v in mist_dur.values()])
    else:
        mist_dur = np.array(mist_dur)

    # compute the duration to the beginning of Feedback subphase
    dur_to_fb = sum(subph_dur)
    # (variable) duration of the feedback intervals: total MIST duration - duration to end of AT
    dur_fb = mist_dur - dur_to_fb

    # add maximum FB duration to subphase duration list
    subph_dur = np.append(subph_dur, max(dur_fb))
    # cumulative times
    times_cum = np.cumsum(np.array(subph_dur))
    # compute start/end times per subphase
    return [(start, end) for start, end in zip(np.append([0], times_cum[:-1]), times_cum)]


# TODO add kw_args
def hr_ensemble_plot(data: Dict[str, pd.DataFrame], plot_params: Optional[Dict] = None,
                     ylims: Optional[Sequence[float]] = None, fontsize: Optional[int] = 14,
                     ax: Optional[plt.Axes] = None,
                     figsize: Optional[Tuple[float, float]] = None) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    """
    Plots the course of heart rate during each MIST phase continuously as ensemble plot (mean ± standard error).
    Simply pass a 'MIST dict' dictionary with one pandas heart rate dataframe per MIST phase
    (see `mist.mist_concat_dicts` for further explanation), i.e. heart rate data with one column per subject.

    Parameters
    ----------
    data : dict
        dict with heart rate data to plot
    plot_params : dict, optional
        dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
        For an overview of parameters and their default values, see `mist.hr_ensemble_params`
    ylims : list, optional
        y axis limits or ``None`` to infer y axis limits from data. Default: ``None``
    fontsize : int, optional. Default: ``None``
        font size. Default: 14
    ax : plt.Axes, optional
        Axes to plot on, otherwise create a new one. Default: ``None``

    Returns
    -------
    tuple or none
        Tuple of Figure and Axes or None if Axes object was passed
    """
    import matplotlib.ticker as mticks

    fig: Union[plt.Figure, None] = None
    if ax is None:
        if figsize is None:
            figsize = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots()

    if plot_params:
        hr_ensemble_params.update(plot_params)

    # sns.despine()
    sns.set_palette(hr_ensemble_params['colormap'])

    line_styles = hr_ensemble_params['line_styles']
    subphases = np.array(mist_params['subphases'])

    mist_dur = [len(v) for v in data.values()]
    start_end = get_mist_times(mist_dur)

    for i, key in enumerate(data):
        hr_mist = data[key]
        x = hr_mist.index
        hr_mean = hr_mist.mean(axis=1)
        hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
        ax.plot(x, hr_mean, zorder=2, label="MIST Phase {}".format(i + 1), linestyle=line_styles[i])
        ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=0.4)
        ax.vlines(x=mist_dur[i], ymin=-20, ymax=40, linestyles='dashed', colors="#bdbdbd", zorder=3)
        ax.text(x=mist_dur[i] - 5, y=10 + 5 * i, s="End Phase {}".format(i + 1), fontsize=fontsize - 4,
                horizontalalignment='right', bbox=dict(facecolor='#e0e0e0', alpha=0.7, boxstyle='round'), zorder=3)

    ax.set_ylabel(r'$\Delta$HR [%]', fontsize=fontsize)
    ax.set_xlabel(r'Time [s]', fontsize=fontsize)
    ax.set_xticks([start for (start, end) in start_end])
    ax.xaxis.set_minor_locator(mticks.MultipleLocator(60))
    ax.tick_params(axis="x", which='both', bottom=True)
    ax.tick_params(axis="y", which='major', left=True)

    for (start, end), subphase in zip(start_end, subphases):
        ax.text(x=start + 0.5 * (end - start), y=35, s=subphase, horizontalalignment='center', fontsize=fontsize)

    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85), prop={'size': fontsize})
    if ylims:
        ax.set_ylim(ylims)
    ax._xmargin = 0

    for (start, end), color, alpha in zip(start_end, hr_ensemble_params['background.color'],
                                          hr_ensemble_params['background.alpha']):
        ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)

    if fig:
        fig.tight_layout()
        return fig, ax


# TODO add support for groups in one dataframe (indicated by group column)
# TODO add kw_args
def hr_course_plot(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                   groups: Optional[Sequence[str]] = None, group_col: Optional[str] = None,
                   plot_params: Optional[Dict] = None, ylims: Optional[Sequence[float]] = None,
                   fontsize: Optional[int] = 14, ax: Optional[plt.Axes] = None,
                   figsize: Optional[Tuple[float, float]] = None) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Plots the course of heart rate during the complete MIST (mean ± standard error per subphase).

    In case of only one group a pandas dataframe can be passed.

    In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe belongs
    to one group, or one dataframe with a column indicating group membership (parameter `group_col`).

    Regardless of the kind of input the dataframes need to be in the format of a 'mse dataframe', as returned
    by `mist.mist_hr_course` (see `mist.mist_hr_course` for further information).


    Parameters
    ----------
    data : dataframe or dict
        Heart rate data to plot. Can either be one dataframe (in case of only one group or in case of multiple groups,
        together with `group_col`) or a dictionary of dataframes, where one dataframe belongs to one group
    groups : list, optional:
         list of group names. If ``None`` is passed, it is inferred from the dictionary keys or from the unique
         values in `group_col`. Default: ``None``
    group_col : str, optional
        Name of group column in the dataframe in case of multiple groups and one dataframe
    plot_params : dict, optional
        dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
        For an overview of parameters and their default values, see `mist.hr_course_params`
    ylims : list, optional
        y axis limits or ``None`` to infer y axis limits from data. Default: ``None``
    fontsize : int, optional. Default: ``None``
        font size. Default: 14
    ax : plt.Axes, optional
        Axes to plot on, otherwise create a new one. Default: ``None``

    Returns
    -------
    tuple or none
        Tuple of Figure and Axes or None if Axes object was passed
    """

    fig: Union[plt.Figure, None] = None
    if ax is None:
        if figsize is None:
            figsize = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots(figsize=figsize)

    # update default parameter if plot parameter were passe
    if plot_params:
        hr_course_params.update(plot_params)

    # get all plot parameter
    sns.set_palette(hr_course_params['colormap'])
    line_styles = hr_course_params['line_styles']
    markers = hr_course_params['markers']
    bg_colors = hr_course_params['background.color']
    bg_alphas = hr_course_params['background.alpha']
    x_offsets = hr_course_params['x_offsets']

    mist_phases = mist_params['phases']
    subphases = mist_params['subphases']

    if isinstance(data, pd.DataFrame):
        # get subphase labels from data
        subphase_labels = data.index.get_level_values(1)
        if not ylims:
            ylims = [1.1 * (data['mean'] - data['se']).min(), 1.5 * (data['mean'] + data['se']).max()]
        if group_col and not groups:
            # get group names from data if groups were not supplied
            groups = list(data[group_col].unique())
    else:
        # get subphase labels from data
        subphase_labels = list(data.values())[0].index.get_level_values(1)
        if not ylims:
            ylims = [1.1 * min([(d['mean'] - d['se']).min() for d in data.values()]),
                     1.5 * max([(d['mean'] + d['se']).max() for d in data.values()])]
        if not groups:
            # get group names from dict if groups were not supplied
            groups = list(data.keys())

    num_subph = len(subphases)
    # build x axis, axis limits and limits for MIST phase spans
    x = np.arange(len(subphase_labels))
    xlims = np.append(x, x[-1] + 1)
    xlims = xlims[::num_subph] + 0.5 * (xlims[::num_subph] - xlims[::num_subph] - 1)
    span_lims = [(x_l, x_u) for x_l, x_u in zip(xlims, xlims[1::])]

    # plot data as errorbar with mean and se
    if groups:
        for df, group, x_off, marker, ls in zip(data.values(), groups, x_offsets, markers, line_styles):
            ax.errorbar(x=x + x_off, y=df['mean'], label=group, yerr=df['se'], capsize=3, marker=marker, linestyle=ls)
    else:
        ax.errorbar(x=x, y=data['mean'], yerr=data['se'], capsize=3, marker=markers[0], linestyle=line_styles[0])

    # add decorators: spans and MIST Phase labels
    for (i, name), (x_l, x_u), color, alpha in zip(enumerate(mist_phases), span_lims, bg_colors, bg_alphas):
        ax.axvspan(x_l, x_u, color=color, alpha=alpha, zorder=0, lw=0)
        ax.text(x=x_l + 0.5 * (x_u - x_l), y=0.9 * ylims[-1], s='MIST Phase {}'.format(i + 1),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize)

    # customize x axis
    ax.tick_params(axis='x', bottom=True)
    ax.set_xticks(x)
    ax.set_xticklabels(subphase_labels)
    ax.set_xlim([span_lims[0][0], span_lims[-1][-1]])

    # customize y axis
    ax.set_ylabel("$\Delta$HR [%]", fontsize=fontsize)
    ax.tick_params(axis="y", which='major', left=True)
    ax.set_ylim(ylims)

    # customize legend
    if groups:
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.85), numpoints=1, prop={"size": fontsize})

    if fig:
        fig.tight_layout()
        return fig, ax
