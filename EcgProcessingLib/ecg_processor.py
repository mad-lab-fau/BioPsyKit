from typing import Optional, Dict, Sequence, Union

import pandas as pd
import numpy as np
import neurokit2 as nk
from neurokit2.stats import rescale

from NilsPodLib import Dataset

from EcgProcessingLib.utils import utc, tz


class EcgProcessor:

    def __init__(self, dataset: Optional[Dataset] = None, df: Optional[pd.DataFrame] = None,
                 sampling_rate: Optional[float] = 256.0):
        if dataset is None and df is None:
            raise ValueError("Either 'dataset' or 'df' must be specified as parameter!")
        self.df_full: pd.DataFrame = pd.DataFrame()
        self.df_result: pd.DataFrame = pd.DataFrame()
        self.df_r_peak_loc: pd.DataFrame = pd.DataFrame()
        self.sampling_rate: int = int(sampling_rate)

        if dataset:
            self.df_full = dataset.data_as_df("ecg", index="utc_datetime")
            self.sampling_rate = int(dataset.info.sampling_rate_hz)
        else:
            self.df_full = df

        self.df_full = pd.DataFrame(self.df_full.tz_localize(tz=utc).tz_convert(tz=tz))
        self.data_dict: Dict = {
            'Data': self.df_full
        }
        self.r_peak_loc_dict: Dict = {}

    @property
    def ecg_result(self) -> Dict[str, pd.DataFrame]:
        return self.data_dict

    @property
    def r_peak_loc(self) -> Dict[str, pd.DataFrame]:
        return self.r_peak_loc_dict

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        return {k: v['ECG_Clean'] for k, v in self.data_dict.items()}

    @property
    def heart_rate(self) -> Dict[str, pd.DataFrame]:
        return {k: v['ECG_Rate'] for k, v in self.data_dict.items()}

    def ecg_process(self, quality_thres: Optional[float] = 0.75):
        ecg_result, info = nk.ecg_process(self.df_full['ecg'].values, sampling_rate=self.sampling_rate)
        ecg_result.index = self.df_full.index
        self.df_result = ecg_result
        self.df_result['Quality_Mask'] = ecg_result['ECG_Quality'] < quality_thres
        rpeaks = info['ECG_R_Peaks']
        self.df_r_peak_loc = pd.DataFrame(rpeaks, index=self.df_full.index[rpeaks], columns=["R_Peaks"])
        self.r_peak_loc_dict['ECG_R_Peaks'] = self.df_r_peak_loc

    def split_data(self, phases: Dict[str, Sequence[str]]):
        self.data_dict.clear()
        self.r_peak_loc_dict.clear()
        for k, v in phases.items():
            self.data_dict[k] = self.df_result.between_time(*v)
            self.r_peak_loc_dict[k] = self.df_r_peak_loc.between_time(*v)

    def ecg_plot(self, ecg_signals: pd.DataFrame, name: Optional[str] = None,
                 plot_individual_beats: Optional[bool] = False) -> 'plt.Figure':
        import matplotlib.gridspec
        import matplotlib.pyplot as plt

        peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

        # Prepare figure and set axes.
        x_axis = ecg_signals.index

        fig = plt.figure(figsize=(10, 5), constrained_layout=False)
        axs: Dict[str, plt.Axes] = {}

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
        # axs = []
        axs['ecg'].get_shared_x_axes().join(axs['ecg'], axs['hr'])

        [ax.set_xlabel("Time (seconds)") for ax in axs.values()]

        if name:
            fig.suptitle("Electrocardiogram (ECG) – {}".format(name), fontweight="bold")
        else:
            fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # Plot cleaned, raw ECG, R-peaks and signal quality.
        axs['ecg'].set_title("Raw and Cleaned Signal")

        quality = rescale(ecg_signals["ECG_Quality"],
                          to=[np.min(ecg_signals["ECG_Clean"]),
                              np.max(ecg_signals["ECG_Clean"])])
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        axs['ecg'].fill_between(x_axis, minimum_line, quality, alpha=0.12, zorder=0,
                                interpolate=True, facecolor="#4CAF50", label='Quality')

        # Plot signals
        axs['ecg'].plot(ecg_signals["ECG_Raw"], color='#B0BEC5', label='Raw', zorder=1)
        axs['ecg'].plot(ecg_signals["ECG_Clean"], color='#E91E63', label="Cleaned", zorder=1, linewidth=1.5)
        axs['ecg'].scatter(x_axis[peaks], ecg_signals["ECG_Clean"][peaks],
                           color="#FFC107", label="R-peaks", zorder=2)

        # Optimize legend
        handles, labels = axs['ecg'].get_legend_handles_labels()
        order = [2, 0, 1, 3]
        axs['ecg'].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper right")

        # Plot heart rate
        axs['hr'].set_title("Heart Rate")
        axs['hr'].set_ylabel("Heart Rate (bpm)")
        axs['hr'].plot(ecg_signals["ECG_Rate"], color="#FF5722", label="Heart Rate", linewidth=1.5)
        rate_mean = ecg_signals["ECG_Rate"].mean()
        axs['hr'].axhline(y=rate_mean, label="Mean", linestyle="--", color="#FF9800")
        axs['hr'].legend(loc="upper right")

        # Plot individual heart beats.
        if plot_individual_beats:
            axs['beats'].set_title("Individual Heart Beats")

            heartbeats = nk.ecg_segment(ecg_signals["ECG_Clean"], peaks, self.sampling_rate)
            heartbeats = nk.epochs_to_df(heartbeats)

            heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')

            axs['beats'].plot(heartbeats_pivoted)

            cmap = iter(
                plt.cm.YlOrRd(np.linspace(0, 1, num=int(heartbeats["Label"].nunique()))))  # Aesthetics of heart beats

            lines = []
            for x, color in zip(heartbeats_pivoted, cmap):
                line, = axs['beats'].plot(heartbeats_pivoted[x], color=color)
                lines.append(line)

        fig.autofmt_xdate()
        return fig

    def plot_ecg_artifacts(self, ecg_signals: pd.DataFrame):
        # Plot artifacts
        _, rpeaks = nk.ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=self.sampling_rate)
        print(rpeaks)
        _, _ = nk.ecg_fixpeaks(rpeaks, sampling_rate=self.sampling_rate, iterative=True, show=True)
