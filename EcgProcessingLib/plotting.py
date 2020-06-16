from typing import Optional, Union, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns

import EcgProcessingLib.utils as utils

sns.set(context="paper", style="white")

plt_fontsize = 14
mpl_rc_params = {
    'xtick.labelsize': plt_fontsize,
    'ytick.labelsize': plt_fontsize,
    'axes.labelsize': plt_fontsize,
    'axes.titlesize': plt_fontsize,
    'legend.title_fontsize': plt_fontsize,
    'legend.fontsize': plt_fontsize,
    'mathtext.default': 'regular'
}
plt.rcParams.update(mpl_rc_params)


# TODO add kwargs
def ecg_plot(ecg_signals: pd.DataFrame, heart_rate: pd.DataFrame, sampling_rate: Optional[int] = 256,
             name: Optional[str] = None, plot_distribution: Optional[bool] = True,
             plot_individual_beats: Optional[bool] = True) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    import matplotlib.gridspec as gs
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticks

    sns.set_palette(utils.cmap_fau)
    plt.rcParams['timezone'] = ecg_signals.index.tz.zone

    outlier = np.where(ecg_signals["ECG_R_Peaks_Outlier"] == 1)[0]
    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]
    peaks = np.setdiff1d(peaks, outlier)
    # Prepare figure and set axes.
    x_axis = ecg_signals.index

    fig = plt.figure(figsize=(15, 5), constrained_layout=False)

    if plot_individual_beats or plot_distribution:
        spec = gs.GridSpec(2, 2, width_ratios=[3, 1])
        axs = {
            'ecg': fig.add_subplot(spec[0, :-1]),
            'hr': fig.add_subplot(spec[1, :-1])
        }
        if plot_distribution and plot_individual_beats:
            axs['beats'] = fig.add_subplot(spec[0, -1])
            axs['dist'] = fig.add_subplot(spec[1, -1])
        elif plot_individual_beats:
            axs['beats'] = fig.add_subplot(spec[:, -1])
        elif plot_distribution:
            axs['dist'] = fig.add_subplot(spec[:, -1])
    else:
        axs = {
            'ecg': fig.add_subplot(2, 1, 1),
            'hr': fig.add_subplot(2, 1, 2)
        }

    axs['ecg'].get_shared_x_axes().join(axs['ecg'], axs['hr'])

    if name:
        fig.suptitle("Electrocardiogram (ECG) – {}".format(name), fontweight="bold")
    else:
        fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    ecg_clean = nk.rescale(ecg_signals["ECG_Clean"],
                           to=[0, 1])
    quality = ecg_signals["ECG_Quality"]
    minimum_line = np.full(len(x_axis), quality.min())

    # Plot quality area first
    axs['ecg'].fill_between(x_axis, minimum_line, quality, alpha=0.2, zorder=2,
                            interpolate=True, facecolor=utils.fau_color('med'), label='Quality')

    # Plot signals
    # axs['ecg'].plot(ecg_signals["ECG_Raw"], color=utils.fau_color('tech'), label='Raw', zorder=1, alpha=0.8)
    axs['ecg'].plot(ecg_clean, color=utils.fau_color('fau'), label="Cleaned", zorder=1,
                    linewidth=1.5)
    axs['ecg'].scatter(x_axis[peaks], ecg_clean[peaks], color=utils.fau_color('nat'),
                       label="R Peaks", zorder=2)
    axs['ecg'].scatter(x_axis[outlier], ecg_clean[outlier], color=utils.fau_color('phil'),
                       label="Outlier", zorder=2)
    axs['ecg'].set_ylabel("ECG Quality")

    # Optimize legend
    handles, labels = axs['ecg'].get_legend_handles_labels()
    # order = [2, 0, 1, 3]
    order = [0, 1, 2, 3]
    axs['ecg'].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right")

    # Plot heart rate
    hr_plot(heart_rate, axs['hr'])

    # Plot individual heart beats
    if plot_individual_beats:
        individual_beats_plot(ecg_signals, peaks, sampling_rate, axs['beats'])

    # Plot heart rate distribution
    if plot_distribution:
        ecg_distribution_plot(heart_rate, axs['dist'])

    axs['ecg'].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    axs['ecg'].tick_params(axis='y', which='major', left=True)

    axs['ecg'].xaxis.set_major_locator(mdates.MinuteLocator())
    axs['ecg'].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs['ecg'].xaxis.set_minor_locator(mticks.AutoMinorLocator(5))

    fig.tight_layout()
    return fig, list(axs.values())


def hr_plot(ecg_signals: pd.DataFrame, ax: Optional[plt.Axes] = None,
            show_mean: Optional[bool] = True, name: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticks

    sns.set_palette(utils.cmap_fau)

    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    if name:
        ax.set_title("Heart Rate {}".format(name))
    ax.set_ylabel("Heart Rate [bpm]")
    ax.plot(ecg_signals["ECG_Rate"], color=utils.fau_color('wiso'), label="Heart Rate", linewidth=1.5)
    if show_mean:
        rate_mean = ecg_signals["ECG_Rate"].mean()
        ax.axhline(y=rate_mean, label="Mean: {:.1f} bpm".format(rate_mean), linestyle="--",
                   color=utils.adjust_color('wiso'), linewidth=2)
        ax.margins(x=0)
        ax.legend(loc="upper right")

    ax.tick_params(axis='x', which='both', bottom=True)
    ax.xaxis.set_major_locator(mdates.MinuteLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(5))

    ax.tick_params(axis='y', which='major', left=True)
    ax.yaxis.set_major_locator(mticks.MaxNLocator(5, steps=[5, 10]))

    if fig:
        fig.tight_layout()
        fig.autofmt_xdate()
        return fig, ax


def hrv_plot(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256, plot_frequency: Optional[bool] = True) -> Tuple[
    plt.Figure, Sequence[plt.Axes]]:
    import matplotlib.gridspec as gs
    fig = plt.figure(constrained_layout=False, figsize=(14, 7))

    if plot_frequency:
        spec = gs.GridSpec(ncols=2, nrows=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = {
            'dist': fig.add_subplot(spec[0, :-1]),
            'freq': fig.add_subplot(spec[1, :-1]),
        }
    else:
        spec = gs.GridSpec(ncols=2, nrows=1, width_ratios=[1, 1])
        axs = {
            'dist': fig.add_subplot(spec[0, :-1])
        }

    spec_within = gs.GridSpecFromSubplotSpec(4, 4, subplot_spec=spec[:, -1], wspace=0.025, hspace=0.05)
    axs['poin'] = fig.add_subplot(spec_within[1:4, 0:3])
    axs['poin_x'] = fig.add_subplot(spec_within[0, 0:3])
    axs['poin_y'] = fig.add_subplot(spec_within[1:4, 3])
    axs['poin'].get_shared_x_axes().join(axs['poin'], axs['poin_x'])
    axs['poin'].get_shared_y_axes().join(axs['poin'], axs['poin_y'])

    hrv_distribution_plot(rpeaks, sampling_rate, axs['dist'])
    hrv_poincare_plot(rpeaks, sampling_rate, [axs['poin'], axs['poin_x'], axs['poin_y']])
    if plot_frequency:
        hrv_frequency_plot(rpeaks, sampling_rate, axs['freq'])

    fig.tight_layout()
    return fig, list(axs.values())


def individual_beats_plot(ecg_signals: pd.DataFrame, peaks: Optional[Sequence[int]] = None,
                          sampling_rate: Optional[int] = 256, ax: Optional[plt.Axes] = None) -> Tuple[
    plt.Figure, plt.Axes]:
    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    if peaks is None:
        peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    heartbeats = nk.ecg_segment(ecg_signals['ECG_Clean'], peaks, sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')

    ax.set_title("Individual Heart Beats")
    ax.margins(x=0)

    # Aesthetics of heart beats
    cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=int(heartbeats["Label"].nunique()))))

    for x, color in zip(heartbeats_pivoted, cmap):
        ax.plot(heartbeats_pivoted[x], color=color)

    ax.set_yticks([])
    ax.tick_params(axis='x', which='major', bottom=True, labelbottom=True)

    if fig:
        fig.tight_layout()
        return fig, ax


def ecg_distribution_plot(heart_rate: pd.DataFrame, ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    ax = sns.distplot(heart_rate, color=utils.fau_color('tech'), ax=ax)

    ax.set_title("Heart Rate Distribution")
    ax.set_xlabel("Heart Rate [bpm]")
    ax.set_xlim(heart_rate.min().min() - 1, heart_rate.max().max() + 1)
    ax.tick_params(axis='x', which='major', bottom=True, labelbottom=True)
    ax.set_yticks([])

    if fig:
        fig.tight_layout()
        return fig, ax


def ecg_plot_artifacts(ecg_signals: pd.DataFrame, sampling_rate: Optional[int] = 256):
    # Plot artifacts
    _, rpeaks = nk.ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=sampling_rate)
    _, _ = nk.ecg_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, show=True)


def hrv_distribution_plot(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256,
                          ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    rri = _get_rr_intervals(rpeaks, sampling_rate)

    sns.set_palette(utils.cmap_fau_blue('2'))
    sns.distplot(rri, ax=ax, bins=10, kde=False, rug=True,
                 rug_kws={'lw': 1.5, 'height': 0.05, 'zorder': 2},
                 hist_kws={'alpha': 0.5, 'zorder': 1})
    ax2 = ax.twinx()
    sns.kdeplot(rri, ax=ax2, lw=2.0, zorder=1)
    ax2.set_ylim(0)
    ax2.axis('off')

    ax.tick_params(axis='both', left=True, bottom=True)
    ax.set_title("Distribution of RR Intervals")
    ax.set_xlabel("RR Intervals [ms]")
    ax.set_ylabel("Count")

    ax2.boxplot(
        rri,
        vert=False,
        positions=[ax2.get_ylim()[-1] / 10 + 0.05 * ax2.get_ylim()[-1]],
        widths=ax2.get_ylim()[-1] / 10,
        manage_ticks=False,
        patch_artist=True,
        boxprops=dict(linewidth=2.0, color=utils.adjust_color('tech', 0.5), facecolor=utils.fau_color('tech')),
        medianprops=dict(linewidth=2.0, color=utils.adjust_color('tech', 0.5)),
        whiskerprops=dict(linewidth=2.0, color=utils.adjust_color('tech', 0.5)),
        capprops=dict(linewidth=2.0, color=utils.adjust_color('tech', 0.5)),
        zorder=4
    )

    ax.set_xlim(0.95 * np.min(rri), 1.05 * np.max(rri))

    if fig:
        fig.tight_layout()
        return fig, ax


def hrv_poincare_plot(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256,
                      axs: Optional[Sequence[plt.Axes]] = None) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    import matplotlib.ticker as mticks
    import matplotlib.gridspec as spec
    fig: Union[plt.Figure, None] = None
    if axs is None:
        fig = plt.figure(figsize=(8, 8))
        axs: Sequence[plt.Axes] = []
        # Prepare figure
        gs = spec.GridSpec(4, 4)
        axs.append(plt.subplot(gs[1:4, 0:3]))
        axs.append(plt.subplot(gs[0, 0:3]))
        axs.append(plt.subplot(gs[1:4, 3]))
        gs.update(wspace=0.025, hspace=0.05)  # Reduce spaces

    rri = _get_rr_intervals(rpeaks, sampling_rate)

    mean_rri = float(np.mean(rri))
    sd = np.ediff1d(rri)
    sdsd = np.std(sd, ddof=1)
    sd1 = sdsd / np.sqrt(2)
    sd2 = np.sqrt(2 * np.std(rri, ddof=1) ** 2 - sd1 ** 2)

    area = np.pi * sd1 * sd2

    sns.set_palette(utils.cmap_fau_blue('2'))
    sns.kdeplot(rri[:-1], rri[1:], ax=axs[0], n_levels=20, shade=True, shade_lowest=False, alpha=0.8)
    sns.scatterplot(rri[:-1], rri[1:], ax=axs[0], alpha=0.5, edgecolor=utils.fau_color('fau'))
    sns.distplot(rri[:-1], bins=int(len(rri) / 10), ax=axs[1], hist_kws=dict(edgecolor="none"))
    sns.distplot(rri[1:], bins=int(len(rri) / 10), ax=axs[2], vertical=True, hist_kws=dict(edgecolor="none"))

    ellipse = mpl.patches.Ellipse((mean_rri, mean_rri), width=2 * sd2, height=2 * sd1, angle=45,
                                  ec=utils.fau_color('fau'),
                                  fc=utils.adjust_color('fau', 1.5))
    axs[0].add_artist(ellipse)

    na = 4
    arr_sd1 = axs[0].arrow(mean_rri, mean_rri, -(sd1 - na) * np.cos(np.deg2rad(45)),
                           (sd1 - na) * np.cos(np.deg2rad(45)), head_width=na, head_length=na, linewidth=2.0,
                           ec=utils.fau_color('phil'), fc=utils.fau_color('phil'), zorder=4)
    arr_sd2 = axs[0].arrow(mean_rri, mean_rri, (sd2 - na) * np.cos(np.deg2rad(45)),
                           (sd2 - na) * np.sin(np.deg2rad(45)), head_width=na, head_length=na, linewidth=2.0,
                           ec=utils.fau_color('med'), fc=utils.fau_color('med'), zorder=4)
    axs[0].add_line(
        mpl.lines.Line2D((np.min(rri), np.max(rri)),
                         (np.min(rri), np.max(rri)),
                         c=utils.fau_color('med'), ls=':', lw=2.0, alpha=0.8))
    axs[0].add_line(
        mpl.lines.Line2D((mean_rri - sd1 * np.cos(np.deg2rad(45)) * na, mean_rri + sd1 * np.cos(np.deg2rad(45)) * na),
                         (mean_rri + sd1 * np.sin(np.deg2rad(45)) * na, mean_rri - sd1 * np.sin(np.deg2rad(45)) * na),
                         c=utils.fau_color('phil'), ls=':', lw=2.0, alpha=0.8))
    # for Area and SD1/SD2 in Legend
    a3 = mpl.patches.Patch(facecolor='white', alpha=0.0)
    a4 = mpl.patches.Patch(facecolor='white', alpha=0.0)

    axs[0].legend(
        [arr_sd1, arr_sd2, a3, a4],
        ['SD1: $%.3f ms$' % sd1, 'SD2: $%.3f ms$' % sd2, 'S: $%.3f ms^2$' % area, 'SD1/SD2: %.3f' % (sd1 / sd2)],
        framealpha=1, fontsize=plt_fontsize - 4)

    axs[0].set_xlabel(r"$RR_{i} [ms]$")
    axs[0].set_ylabel(r"$RR_{i+1} [ms]$")
    axs[0].xaxis.set_major_locator(mticks.MultipleLocator(50))
    axs[0].xaxis.set_minor_locator(mticks.MultipleLocator(10))
    axs[0].yaxis.set_major_locator(mticks.MultipleLocator(50))
    axs[0].yaxis.set_minor_locator(mticks.MultipleLocator(10))
    axs[0].tick_params(axis='both', which='both', left=True, bottom=True)
    axs[1].axis('off')
    axs[2].axis('off')

    if fig:
        fig.tight_layout()
        return fig, axs


def hrv_frequency_plot(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256,
                       ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    from neurokit2.hrv.hrv_frequency import _hrv_frequency_show
    from neurokit2.hrv.hrv_utils import _hrv_get_rri
    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    rri = _hrv_get_rri(rpeaks['R_Peak_Idx'], sampling_rate=sampling_rate, interpolate=True)[0]
    hrv = nk.hrv_frequency(rpeaks, sampling_rate)
    out_bands = hrv[["HRV_ULF", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF"]]
    out_bands.columns = [col.replace('HRV_', '') for col in out_bands.columns]
    _hrv_frequency_show(rri, out_bands, sampling_rate=256, ax=ax)

    ax.tick_params(axis='both', left=True, bottom=True)
    ax.margins(x=0)
    ax.set_ylim(0)

    if fig:
        fig.tight_layout()
        return fig, ax


def _get_rr_intervals(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256) -> np.array:
    rri = (np.ediff1d(rpeaks['R_Peak_Idx'], to_begin=0) / sampling_rate) * 1000
    rri = rri[1:]
    return rri
