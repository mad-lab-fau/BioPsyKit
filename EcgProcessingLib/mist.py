from typing import Dict, Sequence, List

from EcgProcessingLib.plotting import *

hr_ensemble_params = {
    'colormap': utils.cmap_fau_blue('3'),
    # MIST Subphases
    'subphases': ['BL', 'AT', 'FB'],
    # duration of subphases BL, AT in seconds
    'subphases.duration': [60, 240],
    'marker': ['-', '--', ':'],
    'background.color': ['#e0e0e0', '#9e9e9e', '#757575'],
    'background.alpha': [0.6, 0.7, 0.7],
}


def mist_hr_ensemble_plot(ax: plt.Axes, data: Dict, plot_params: Optional[Dict] = None,
                          ylims: Optional[Sequence[float]] = None, fontsize: Optional[int] = 14) -> plt.Axes:
    if plot_params:
        hr_ensemble_params.update(plot_params)

    sns.despine()
    sns.set_palette(hr_ensemble_params['colormap'])

    subphases = hr_ensemble_params['subphases']
    subph_dur: List[int] = hr_ensemble_params['subphases.duration']
    dur_to_at = sum(subph_dur)
    # length of the feedback intervals
    len_fb = [len(v) - dur_to_at for v in data.values()]
    marker = hr_ensemble_params['marker']
    # add maximum FB duration to subphase duration list (for plotting)
    subph_dur = subph_dur + [max(len_fb)]

    times_cum = np.cumsum(np.array(subph_dur)).tolist()
    # compute start/end times per subphase
    start_end = [(start, end) for start, end in zip([0] + times_cum[:-1], times_cum)]

    for i, key in enumerate(data):
        hr_mist = data[key]
        x = hr_mist.index
        hr_mean = hr_mist.mean(axis=1)
        hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
        ax.plot(x, hr_mean, zorder=2, label="MIST Phase {}".format(i + 1), linestyle=marker[i])
        ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=0.4)
        ax.vlines(x=dur_to_at + len_fb[i], ymin=-20, ymax=40, linestyles='dashed', colors="#bdbdbd", zorder=3)
        ax.text(x=dur_to_at + len_fb[i] - 5, y=10 + 5 * i, s="End Phase {}".format(i + 1), fontsize=fontsize - 4,
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
