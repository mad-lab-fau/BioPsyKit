from typing import Optional, Union, Sequence

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
             name: Optional[str] = None, plot_distribution: Optional[bool] = False,
             plot_individual_beats: Optional[bool] = False) -> plt.Figure:
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
            axs['dist'] = fig.add_subplot(spec[0, -1])
            axs['beats'] = fig.add_subplot(spec[1, -1])
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

    # Plot cleaned, raw ECG, R-peaks and signal quality
    # axs['ecg'].set_title("Raw and Cleaned Signal")

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
    axs['hr'] = hr_plot(heart_rate, axs['hr'])

    # Plot individual heart beats
    if plot_individual_beats:
        individual_beats_plot(ecg_signals, peaks, sampling_rate, axs['beats'])

    if plot_distribution:
        ecg_distribution_plot(heart_rate, axs['hist'])

    fig.tight_layout()
    fig.autofmt_xdate(rotation=0, ha='center')

    axs['ecg'].tick_params(axis='x', which='both', bottom=True)
    axs['ecg'].xaxis.set_major_locator(mdates.MinuteLocator())
    axs['ecg'].xaxis.set_minor_locator(mticks.AutoMinorLocator(5))

    axs['ecg'].tick_params(axis='y', which='major', left=True)

    if plot_individual_beats:
        axs['beats'].tick_params(axis='x', which='major', bottom=True, labelbottom=True)

    if plot_distribution:
        axs['dist'].tick_params(axis='x', which='major', bottom=True, labelbottom=True)
        axs['dist'].set_xlabel("Heart Rate [bpm]")

    return fig


def hr_plot(ecg_signals: pd.DataFrame, ax: Optional[plt.Axes] = None,
            show_mean: Optional[bool] = True, name: Optional[str] = None) -> plt.Axes:
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
    return ax


def hrv_plot(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256, plot_frequency: Optional[bool] = True):
    import matplotlib.gridspec as gs
    fig = plt.figure(constrained_layout=False, figsize=(12, 7))

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
    # axs['poin_x'].set_title("Poincaré Plot")
    axs['poin_y'] = fig.add_subplot(spec_within[1:4, 3])

    axs['dist'] = hrv_distribution_plot(rpeaks, sampling_rate, axs['dist'])

    fig.tight_layout()


def individual_beats_plot(ecg_signals: pd.DataFrame, peaks: Optional[Sequence[int]] = None,
                          sampling_rate: Optional[int] = 256, ax: Optional[plt.Axes] = None):
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

    if fig:
        fig.tight_layout()
        return ax


def ecg_distribution_plot(heart_rate: pd.DataFrame, ax: Optional[plt.Axes] = None):
    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    ax = sns.distplot(heart_rate, color=utils.fau_color('tech'), ax=ax)

    ax.set_title("Heart Rate Distribution")
    ax.set_xlabel("Heart Rate [bpm]")
    ax.set_yticks([])
    ax.set_xlim(heart_rate.min().min() - 1, heart_rate.max().max() + 1)
    # ax.tick_params(axis='x', which='major', bottom=True)

    if fig:
        fig.tight_layout()
        return ax


def ecg_plot_artifacts(ecg_signals: pd.DataFrame, sampling_rate: Optional[int] = 256):
    # Plot artifacts
    _, rpeaks = nk.ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=sampling_rate)
    _, _ = nk.ecg_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, show=True)


def hrv_distribution_plot(rpeaks: pd.DataFrame, sampling_rate: Optional[int] = 256,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    rri: np.ndarray = (np.ediff1d(rpeaks['R_Peak_Idx'], to_begin=0) / sampling_rate) * 1000
    rri = rri[1:]

    rug_height = 0.05

    sns.distplot(rri, color=utils.fau_color('med'), ax=ax, bins=10, kde=False, rug=True,
                 rug_kws={'color': utils.fau_color('fau'), 'lw': 1.5, 'height': rug_height, 'zorder': 2},
                 hist_kws={'alpha': 1, 'zorder': 1})
    ax2: plt.Axes = ax.twinx()
    sns.kdeplot(rri, color=utils.adjust_color('med', 0.75), ax=ax2, lw=2.0, zorder=1)
    ax2.tick_params(axis='y', right=False, labelright=False)
    ax2.set_ylim(0)

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
        boxprops=dict(linewidth=2.0, color=utils.fau_color('fau'), facecolor=utils.adjust_color('fau', 2.0)),
        medianprops=dict(linewidth=2.0, color=utils.fau_color('fau')),
        whiskerprops=dict(linewidth=2.0, color=utils.fau_color('fau')),
        capprops=dict(linewidth=2.0, color=utils.fau_color('fau')),
        zorder=4
    )

    ax.set_xlim(0.95 * np.min(rri), 1.05 * np.max(rri))

    return ax
