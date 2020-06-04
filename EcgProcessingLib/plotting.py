from typing import Optional, Union

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import seaborn as sns
from neurokit2 import rescale

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


def ecg_plot(ecg_signals: pd.DataFrame, sampling_rate: Optional[int] = 256, name: Optional[str] = None,
             plot_individual_beats: Optional[bool] = False) -> plt.Figure:
    import matplotlib.gridspec
    sns.set_palette(utils.cmap_fau)

    peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]
    # Prepare figure and set axes.
    x_axis = ecg_signals.index

    fig = plt.figure(figsize=(10, 5), constrained_layout=False)
    if plot_individual_beats:
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[2 / 3, 1 / 3])
        axs = {
            'ecg': fig.add_subplot(gs[0, :-1]),
            'hr': fig.add_subplot(gs[1, :-1]),
            'beats': fig.add_subplot(gs[:, -1])
        }
    else:
        axs = {
            'ecg': fig.add_subplot(2, 1, 1),
            'hr': fig.add_subplot(2, 1, 2)
        }
    axs['ecg'].get_shared_x_axes().join(axs['ecg'], axs['hr'])
    [ax.set_xlabel("Time [s]") for ax in axs.values()]

    if name:
        fig.suptitle("Electrocardiogram (ECG) – {}".format(name), fontweight="bold")
    else:
        fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # Plot cleaned, raw ECG, R-peaks and signal quality
    axs['ecg'].set_title("Raw and Cleaned Signal")

    quality = rescale(ecg_signals["ECG_Quality"],
                      to=[np.min(ecg_signals["ECG_Clean"]),
                          np.max(ecg_signals["ECG_Clean"])])
    minimum_line = np.full(len(x_axis), quality.min())

    # Plot quality area first
    axs['ecg'].fill_between(x_axis, minimum_line, quality, alpha=0.12, zorder=0,
                            interpolate=True, facecolor=utils.fau_color('med'), label='Quality')
    # Plot signals
    axs['ecg'].plot(ecg_signals["ECG_Raw"], color=utils.fau_color('tech'), label='Raw', zorder=1, alpha=0.8)
    axs['ecg'].plot(ecg_signals["ECG_Clean"], color=utils.fau_color('fau'), label="Cleaned", zorder=1,
                    linewidth=1.5)
    axs['ecg'].scatter(x_axis[peaks], ecg_signals["ECG_Clean"][peaks],
                       color=utils.fau_color('phil'), label="R-peaks", zorder=2)
    # Optimize legend
    handles, labels = axs['ecg'].get_legend_handles_labels()
    order = [2, 0, 1, 3]
    axs['ecg'].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right")
    # Plot heart rate
    axs['hr'] = hr_plot(ecg_signals, axs['hr'])

    # Plot individual heart beats
    if plot_individual_beats:
        heartbeats = nk.ecg_segment(ecg_signals["ECG_Clean"], peaks, sampling_rate)
        heartbeats = nk.epochs_to_df(heartbeats)
        heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')

        axs['beats'].set_title("Individual Heart Beats")
        axs['beats'].plot(heartbeats_pivoted)
        print(heartbeats["Label"].nunique())

        # Aesthetics of heart beats
        cmap = iter(plt.cm.YlOrRd(np.linspace(0, 1, num=int(heartbeats["Label"].nunique()))))

        lines = []
        for x, color in zip(heartbeats_pivoted, cmap):
            line, = axs['beats'].plot(heartbeats_pivoted[x], color=color)
            lines.append(line)

    fig.autofmt_xdate()
    return fig


def hr_plot(ecg_signals: pd.DataFrame, ax: Optional[plt.Axes] = None,
            show_mean: Optional[bool] = True, name: Optional[str] = None) -> plt.Axes:
    fig: Union[plt.Figure, None] = None
    if ax is None:
        fig, ax = plt.subplots()

    if name:
        ax.set_title("Heart Rate {}".format(name))
    ax.set_ylabel("Heart Rate [bpm]")
    ax.plot(ecg_signals["ECG_Rate"], color=utils.fau_color('wiso'), label="Heart Rate", linewidth=1.5)
    if show_mean:
        rate_mean = ecg_signals["ECG_Rate"].mean()
        ax.axhline(y=rate_mean, label="Mean: {:.1f} bpm".format(rate_mean), linestyle="--", color=utils.adjust_color('wiso'), linewidth=2)
        ax.legend(loc="upper right")

    if fig:
        fig.tight_layout()
        fig.autofmt_xdate()
    return ax


def ecg_plot_artifacts(ecg_signals: pd.DataFrame, sampling_rate: Optional[int] = 256):
    # Plot artifacts
    _, rpeaks = nk.ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=sampling_rate)
    _, _ = nk.ecg_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, show=True)
