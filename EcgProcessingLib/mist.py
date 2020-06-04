from typing import Dict, Sequence, List, Tuple

from EcgProcessingLib.plotting import *

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
    'linestyles': ['-', '--', ':'],
    'background.color': ['#e0e0e0', '#9e9e9e', '#757575'],
    'background.alpha': [0.6, 0.7, 0.7],
}

hr_course_params = {
    'colormap': utils.cmap_fau_blue('2_lp'),
    'linestyles': ['-', '--'],
    'markers': ['o', 'P'],
    'background.color': ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
    'background.alpha': [0.6, 0.7, 0.7],
    'x_offsets': [0, 0.05]
}


def mist_split_subphases(data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
                         mist_times: Optional[Sequence[Tuple[int, int]]] = None,
                         is_group_dict: Optional[bool] = False) -> Dict[str, Dict[str, pd.DataFrame]]:
    if is_group_dict:
        return {group: mist_split_subphases(dict_group) for group, dict_group in data.items()}
    else:
        if not mist_times:
            mist_times = mist_get_times(data)
        mist_dict = {}
        mist_subph = mist_params['subphases']
        for phase, df in data.items():
            mist_dict[phase] = {subph: df[start:end] for subph, (start, end) in zip(mist_subph, mist_times)}
        return mist_dict


def mist_split_groups(condition_dict: Dict[str, Sequence[str]],
                      mist_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    return {condition: {key: df[condition_dict[condition]] for key, df in mist_dict.items()} for condition
            in condition_dict.keys()}


def mist_hr_course(data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
                   subphases: Optional[Sequence[str]] = None,
                   is_group_dict: Optional[bool] = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if not subphases:
        subphases = mist_params['subphases']

    if is_group_dict:
        return {group: mist_hr_course(dict_group, subphases) for group, dict_group in data.items()}
    else:
        mean_hr = {phase: pd.DataFrame({subph: df[subph].mean() for subph in subphases}) for phase, df in data.items()}
        df_mist = pd.concat(mean_hr.values(), axis=1, keys=mean_hr.keys())
        return pd.concat([df_mist.mean(), df_mist.std() / np.sqrt(df_mist.shape[0])], axis=1, keys=["mean", "se"])


def mist_get_times(mist_dur: Union[Sequence[int], Dict[str, pd.DataFrame]], subph_dur: Optional[Sequence[int]] = None):
    if subph_dur:
        subph_dur = np.array(subph_dur)
    else:
        subph_dur = np.array(mist_params['subphases.duration'])

    if isinstance(mist_dur, dict):
        mist_dur = np.array([len(v) for v in mist_dur.values()])
    else:
        mist_dur = np.array(mist_dur)

    dur_to_at = sum(subph_dur)
    # length of the feedback intervals
    len_fb = mist_dur - dur_to_at

    # add maximum FB duration to subphase duration list
    subph_dur = np.append(subph_dur, max(len_fb))
    # cumulative times
    times_cum = np.cumsum(np.array(subph_dur))
    # compute start/end times per subphase
    start_end = [(start, end) for start, end in zip(np.append([0], times_cum[:-1]), times_cum)]

    return start_end


def mist_hr_ensemble_plot(ax: plt.Axes, data: Dict[str, pd.DataFrame], plot_params: Optional[Dict] = None,
                          ylims: Optional[Sequence[float]] = None, fontsize: Optional[int] = 14) -> plt.Axes:
    if plot_params:
        hr_ensemble_params.update(plot_params)

    # sns.despine()
    sns.set_palette(hr_ensemble_params['colormap'])

    linestyles = hr_ensemble_params['linestyles']
    subphases = np.array(mist_params['subphases'])

    mist_dur = [len(v) for v in data.values()]
    start_end = mist_get_times(mist_dur)

    for i, key in enumerate(data):
        hr_mist = data[key]
        x = hr_mist.index
        hr_mean = hr_mist.mean(axis=1)
        hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
        ax.plot(x, hr_mean, zorder=2, label="MIST Phase {}".format(i + 1), linestyle=linestyles[i])
        ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=0.4)
        ax.vlines(x=mist_dur[i], ymin=-20, ymax=40, linestyles='dashed', colors="#bdbdbd", zorder=3)
        ax.text(x=mist_dur[i] - 5, y=10 + 5 * i, s="End Phase {}".format(i + 1), fontsize=fontsize - 4,
                horizontalalignment='right', bbox=dict(facecolor='#e0e0e0', alpha=0.7, boxstyle='round'), zorder=3)

    ax.set_ylabel(r'$\Delta$HR [%]', fontsize=fontsize)
    ax.set_xlabel(r'Time [s]', fontsize=fontsize)
    ax.set_xticks([start for (start, end) in start_end])
    for (start, end), subphase in zip(start_end, subphases):
        ax.text(x=start + 0.5 * (end - start), y=35, s=subphase, horizontalalignment='center', fontsize=fontsize)

    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85), prop={'size': fontsize})
    if ylims:
        ax.set_ylim(ylims)
    ax._xmargin = 0

    for (start, end), color, alpha in zip(start_end, hr_ensemble_params['background.color'],
                                          hr_ensemble_params['background.alpha']):
        ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)

    return ax


# TODO add support for groups in one dataframe (indicated by group column)
def mist_hr_course_plot(ax: plt.Axes, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                        groups: Optional[str] = None, group_col: Optional[str] = None,
                        plot_params: Optional[Dict] = None, ylims: Optional[Sequence[float]] = None,
                        fontsize: Optional[int] = 14) -> plt.Axes:
    if plot_params:
        hr_course_params.update(plot_params)

    sns.set_palette(hr_course_params['colormap'])
    line_styles = hr_course_params['linestyles']
    markers = hr_course_params['markers']
    bg_colors = hr_course_params['background.color']
    bg_alphas = hr_course_params['background.alpha']
    x_offsets = hr_course_params['x_offsets']

    mist_phases = mist_params['phases']
    subphases = mist_params['subphases']

    if isinstance(data, pd.DataFrame):
        subphase_labels = data.index.get_level_values(1)
        if not ylims:
            ylims = [1.1 * (data['mean'] - data['se']).min(), 1.5 * (data['mean'] + data['se']).max()]
        if group_col and not groups:
            groups = list(data[group_col].unique())
    else:
        subphase_labels = list(data.values())[0].index.get_level_values(1)
        if not ylims:
            ylims = [1.1 * min([(d['mean'] - d['se']).min() for d in data.values()]),
                     1.5 * max([(d['mean'] + d['se']).max() for d in data.values()])]
        if not groups:
            groups = list(data.keys())

    num_subph = len(subphases)

    x = np.arange(len(subphase_labels))
    xlims = np.append(x, x[-1] + 1)
    xlims = xlims[::num_subph] + 0.5 * (xlims[::num_subph] - xlims[::num_subph] - 1)

    span_lims = [(x_l, x_u) for x_l, x_u in zip(xlims, xlims[1::])]

    if groups:
        for df, group, x_off, marker, ls in zip(data.values(), groups, x_offsets, markers, line_styles):
            ax.errorbar(x=x + x_off, y=df['mean'], label=group, yerr=df['se'], capsize=3, marker=marker, linestyle=ls)
    else:
        ax.errorbar(x=x, y=data['mean'], yerr=data['se'], capsize=3, marker=markers[0], linestyle=line_styles[0])

    for (i, name), (x_l, x_u), color, alpha in zip(enumerate(mist_phases), span_lims, bg_colors, bg_alphas):
        ax.axvspan(x_l, x_u, color=color, alpha=alpha, zorder=0, lw=0)
        ax.text(x=x_l + 0.5 * (x_u - x_l), y=0.9 * ylims[-1], s='MIST Phase {}'.format(i + 1),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize)

    ax.set_ylabel("$\Delta$HR [%]", fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(subphase_labels)
    ax.set_xlim([span_lims[0][0], span_lims[-1][-1]])
    ax.set_ylim(ylims)

    if groups:
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.9), numpoints=1, prop={"size": fontsize})

    return ax
