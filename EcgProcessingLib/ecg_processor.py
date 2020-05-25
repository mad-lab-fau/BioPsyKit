from typing import Optional, Dict, Sequence, Union
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import neurokit2 as nk
from neurokit2.stats import rescale

import matplotlib.pyplot as plt
import seaborn as sns

from NilsPodLib import Dataset
import EcgProcessingLib.utils as utils


class EcgProcessor:

    @classmethod
    def split_data(cls, time_info: Union[pd.Series, Dict[str, Sequence[str]]], dataset: Optional[Dataset] = None,
                   df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        data_dict: Dict[str, pd.DataFrame] = {}
        if dataset is None and df is None:
            raise ValueError("Either 'dataset' or 'df' must be specified as parameter!")
        if dataset:
            df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utils.utc).tz_convert(tz=utils.tz)
        if isinstance(time_info, pd.Series):
            for name, start, end in zip(time_info.index, np.pad(time_info, (0, 1)), time_info[1:]):
                data_dict[name] = df.between_time(start, end)
        else:
            data_dict = {name: df.between_time(*start_end) for name, start_end in time_info.items()}
        return data_dict

    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, dataset: Optional[Dataset] = None,
                 df: Optional[pd.DataFrame] = None,
                 sampling_rate: Optional[float] = 256.0):
        if all([i is None for i in [dataset, df, data_dict]]):
            raise ValueError("Either 'dataset', 'df', or 'data_dict' must be specified as parameter!")

        self.df: pd.DataFrame = pd.DataFrame()
        self.sampling_rate: int = int(sampling_rate)

        if data_dict:
            self.data_dict = data_dict
        elif dataset:
            self.df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utils.utc).tz_convert(tz=utils.tz)
            self.sampling_rate = int(dataset.info.sampling_rate_hz)
            self.data_dict: Dict = {
                'Data': self.df
            }
        else:
            self.df = df.tz_localize(tz=utils.utc).tz_convert(tz=utils.tz)
            self.data_dict: Dict = {
                'Data': self.df
            }

        self.ecg_result: Dict = {}

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        return {k: pd.DataFrame(v['ECG_Clean']) for k, v in self.ecg_result.items()}

    @property
    def heart_rate(self) -> Dict[str, pd.DataFrame]:
        return {k: df[df['ECG_R_Peaks'] == 1.0][['ECG_Rate']] for k, df in self.ecg_result.items()}

    def ecg_process(self, quality_thres: Optional[float] = 0.75, title: Optional[str] = None) -> None:
        for name, df in tqdm(self.data_dict.items(), desc=title):
            ecg_result, info = nk.ecg_process(df['ecg'].values, sampling_rate=self.sampling_rate)
            ecg_result.index = df.index
            ecg_result['Quality_Mask'] = ecg_result['ECG_Quality'] < quality_thres
            self.ecg_result[name] = ecg_result

    def ecg_plot(self, ecg_signals: pd.DataFrame, name: Optional[str] = None,
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
        axs['hr'] = self.hr_plot(ecg_signals, axs['hr'])

        # Plot individual heart beats
        if plot_individual_beats:
            heartbeats = nk.ecg_segment(ecg_signals["ECG_Clean"], peaks, self.sampling_rate)
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

    @classmethod
    def hr_plot(cls, ecg_signals: pd.DataFrame, ax: Optional[plt.Axes] = None, show_mean: Optional[bool] = True,
                name: Optional[str] = None) -> plt.Axes:
        fig: Union[plt.Figure, None] = None
        if ax is None:
            fig, ax = plt.subplots()

        if name:
            ax.set_title("Heart Rate {}".format(name))
        ax.set_ylabel("Heart Rate [bpm]")
        ax.plot(ecg_signals["ECG_Rate"], color=utils.fau_color('wiso'), label="Heart Rate", linewidth=1.5)
        if show_mean:
            rate_mean = ecg_signals["ECG_Rate"].mean()
            ax.axhline(y=rate_mean, label="Mean", linestyle="--", color=utils.adjust_color('wiso'), linewidth=2)
            ax.legend(loc="upper right")

        if fig:
            fig.tight_layout()
            fig.autofmt_xdate()
        return ax

    def ecg_plot_artifacts(self, ecg_signals: pd.DataFrame):
        # Plot artifacts
        _, rpeaks = nk.ecg_peaks(ecg_signals["ECG_Clean"], sampling_rate=self.sampling_rate)
        print(rpeaks)
        _, _ = nk.ecg_fixpeaks(rpeaks, sampling_rate=self.sampling_rate, iterative=True, show=True)
