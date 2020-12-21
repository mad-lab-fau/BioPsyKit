import warnings
from typing import Optional, Tuple, Dict, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import biopsykit.colors as colors


class CFT:
    """
    Class representing the Cold Face Test.
    """

    def __init__(
            self,
            name: Optional[str] = None, cft_start: Optional[int] = None,
            cft_duration: Optional[int] = None
    ):
        self.name: str
        """
        Name of the Study
        """
        self.cft_start: int
        """
        Start of the Cold Face Exposure in seconds. It is assumed that the time between the beginning of the
        recording and the start of the Cold Face Exposure is the Baseline. Default: 60 seconds
        """

        self.cft_duration: int
        """
        Duration of the Cold Face Exposure in seconds. Default: 120 seconds
        """
        self._set_cft_params(name, cft_start, cft_duration)

        self.cft_plot_params = {
            'background.color': ['#e0e0e0', '#9e9e9e', '#757575'],
            'background.alpha': [0.5, 0.5, 0.5],
            'phase.names': ["Baseline", "Cold Face Test", "Recovery"]
        }

    def _set_cft_params(self, name: str, cft_start: int, cft_duration: int):
        if name is None:
            self.name = "CFT"
        else:
            self.name = name
        if cft_start is None:
            self.cft_start = 60
        else:
            if cft_start < 0:
                raise ValueError("`cft_start` must be positive or 0!")
            self.cft_start = cft_start

        if cft_duration is None:
            self.cft_duration = 120
        else:
            if cft_duration <= 0:
                raise ValueError("`cft_duration` must be positive!")
            self.cft_duration = cft_duration

    def compute_cft_parameter(self,
                              data: pd.DataFrame,
                              index: Optional[str] = None,
                              return_dict: Optional[bool] = False) -> Union[Dict, pd.DataFrame]:

        if index:
            index = [index]

        dict_cft = {}
        parameters = ['onset', 'peak_bradycardia', 'mean_bradycardia', 'poly_fit']

        df_cft = self.extract_cft_interval(data)
        hr_bl = self.baseline_hr(data)
        dict_cft['baseline_hr'] = hr_bl
        for parameter in parameters:
            dict_cft.update(
                getattr(self, parameter)(data=df_cft, is_cft_interval=True, compute_baseline=False, hr_bl=hr_bl))

        if return_dict:
            return dict_cft
        return pd.DataFrame([dict_cft], index=index)

    def baseline_hr(self, data: pd.DataFrame) -> float:
        """
        Computes the mean heart rate during baseline.

        Parameters
        ----------
        data

        Returns
        -------

        Warnings
        ------
            If ``cft_start`` is set to 0, i.e. no baseline is present, the first HR value in the
            dataframe if used as baseline
        """

        # start of baseline = start of recording
        bl_start = data.index[0]
        if self.cft_start == 0:
            warnings.warn(
                "cft_start is 0, no baseline can be extracted! Using the first HR value in the dataframe as baseline."
            )
        # end of baseline = start of CFT
        bl_end = bl_start + pd.Timedelta(minutes=int(self.cft_start / 60))
        # heart rate during Baseline
        hr_bl = data.between_time(bl_start.time(), bl_end.time())
        return float(hr_bl.mean())

    def extract_cft_interval(self, data: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        cft_start = data.index[0] + pd.Timedelta(minutes=int(self.cft_start / 60))
        cft_end = data.index[0] + pd.Timedelta(minutes=int((self.cft_start + self.cft_duration) / 60))
        return data.between_time(cft_start.time(), cft_end.time())

    def onset(self,
              data: pd.DataFrame,
              is_cft_interval: Optional[bool] = False,
              compute_baseline: Optional[bool] = True,
              hr_bl: Optional[float] = None
              ) -> Dict:

        df_hr_cft, hr_bl = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_bl)

        # bradycardia mask (True where heart rate is below baseline, False otherwise)
        hr_brady = df_hr_cft < hr_bl
        # bradycardia borders (1 where we have a change between lower and higher heart rate)
        brady_border = np.abs(np.ediff1d(hr_brady.astype(int), to_begin=0))
        # filter out the phases where we have at least 3 heart rate values lower than baseline
        brady_phases = hr_brady.groupby([np.cumsum(brady_border)]).filter(lambda df: df.sum() >= 3)
        # CFT onset is the third beat
        cft_onset = brady_phases.index[2]
        cft_onset_latency = (cft_onset - df_hr_cft.index[0]).total_seconds()
        # heart rate at onset point
        hr_onset = np.squeeze(df_hr_cft.loc[cft_onset])
        return {
            'onset': cft_onset,
            'onset_latency': cft_onset_latency,
            'onset_hr': hr_onset,
            'onset_hr_percent': (1 - hr_onset / hr_bl) * 100,
            'onset_slope': (hr_onset - hr_bl) / cft_onset_latency
        }

    def peak_bradycardia(self,
                         data: pd.DataFrame,
                         is_cft_interval: Optional[bool] = False,
                         compute_baseline: Optional[bool] = True,
                         hr_bl: Optional[float] = None) -> Dict:

        df_hr_cft, hr_bl = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_bl)

        peak_brady = np.squeeze(df_hr_cft.idxmin())
        peak_brady_seconds = (peak_brady - df_hr_cft.index[0]).total_seconds()
        hr_brady = np.squeeze(df_hr_cft.loc[peak_brady])
        return {
            'peak_brady': peak_brady,
            'peak_brady_latency': peak_brady_seconds,
            'peak_brady_bpm': hr_brady - hr_bl,
            'peak_brady_percent': (hr_brady / hr_bl - 1) * 100,
            'peak_brady_slope': (hr_brady - hr_bl) / peak_brady_seconds
        }

    def mean_bradycardia(self,
                         data: pd.DataFrame,
                         is_cft_interval: Optional[bool] = False,
                         compute_baseline: Optional[bool] = True,
                         hr_bl: Optional[float] = None) -> Dict:

        df_hr_cft, hr_bl = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_bl)

        hr_mean = np.squeeze(df_hr_cft.mean())

        return {
            'mean_hr_bpm': hr_mean,
            'mean_brady_bpm': hr_mean - hr_bl,
            'mean_brady_percent': (hr_mean / hr_bl - 1) * 100
        }

    def poly_fit(self,
                 data: pd.DataFrame,
                 is_cft_interval: Optional[bool] = False,
                 compute_baseline: Optional[bool] = True,
                 hr_bl: Optional[float] = None) -> Dict:

        df_hr_cft, hr_bl = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_bl)

        # get time points in seconds
        idx_s = df_hr_cft.index.astype(int) / 1e9
        idx_s = idx_s - idx_s[0]

        # apply a 2nd degree polynomial fit
        poly = np.polyfit(idx_s, np.squeeze(df_hr_cft.values), deg=2)
        return {'poly_fit_a{}'.format(i): p for i, p in enumerate(poly)}

    def cft_plot(
            self,
            data: pd.DataFrame,
            time_before: Optional[int] = None,
            time_after: Optional[int] = None,
            ax: Optional[plt.Axes] = None,
            figsize: Optional[Tuple[int, int]] = None,
            plot_baseline: Optional[bool] = True,
            plot_mean: Optional[bool] = True,
            plot_onset: Optional[bool] = True,
            plot_peak_brady: Optional[bool] = True,
            plot_poly_fit: Optional[bool] = True,
    ) -> Union[Tuple[plt.Figure, plt.Axes], plt.Axes]:
        """

        Parameters
        ----------
        data
        time_before
        time_after
        ax
        figsize
        plot_poly_fit
        plot_peak_brady
        plot_onset
        plot_mean
        plot_baseline

        Returns
        -------

        """

        from biopsykit.signals.ecg.plotting import hr_plot

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        if time_before is None:
            time_before = self.cft_start
        if time_after is None:
            time_after = self.cft_start

        bg_colors = self.cft_plot_params['background.color']
        bg_alphas = self.cft_plot_params['background.alpha']
        names = self.cft_plot_params['phase.names']

        cft_params = self.compute_cft_parameter(data, return_dict=True)

        cft_start = data.index[0] + pd.Timedelta(minutes=int(self.cft_start / 60))
        plot_start = cft_start - pd.Timedelta(minutes=int(time_before / 60))
        cft_end = cft_start + pd.Timedelta(minutes=int(self.cft_duration / 60))
        plot_end = cft_end + pd.Timedelta(minutes=int(time_after / 60))

        df_plot = data.between_time(plot_start.time(), plot_end.time())
        df_cft = self.extract_cft_interval(data)
        hr_bl = self.baseline_hr(data)

        times_dict = {'plot_start': plot_start, 'cft_start': cft_start, 'cft_end': cft_end, 'plot_end': plot_end}
        times = [(start, end) for start, end in zip(list(times_dict.values()), list(times_dict.values())[1:])]

        hr_plot(heart_rate=df_plot, ax=ax, plot_mean=False)

        # TODO change hardcoded plot parameter
        ylims = [0.9 * float(df_plot.min()), 1.1 * float(df_plot.max())]
        ax.set_ylim(ylims)

        bbox = dict(
            fc=(1, 1, 1, plt.rcParams['legend.framealpha']),
            ec=plt.rcParams['legend.edgecolor'],
            boxstyle="round",
        )

        for (start, end), bg_color, bg_alpha, name in zip(times, bg_colors, bg_alphas, names):
            ax.axvspan(xmin=start, xmax=end, color=bg_color, alpha=bg_alpha, lw=0)
            ax.text(x=start + 0.5 * (end - start), y=0.95 * ylims[-1], s=name,
                    # bbox=bbox,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=14)
        ax.axhspan(ymin=0.93, ymax=1.0, transform=ax.get_xaxis_transform(),color='white', alpha=0.4, zorder=3, lw=0)

        if plot_baseline:
            self._add_baseline_plot(data, cft_params, times_dict, ax, bbox)

        if plot_mean:
            self._add_mean_bradycardia_plot(data, cft_params, times_dict, ax, bbox)

        if plot_onset:
            self._add_onset_plot(data, cft_params, times_dict, ax, bbox)

        if plot_peak_brady:
            self._add_peak_bradycardia_plot(data, cft_params, times_dict, ax, bbox)

        if plot_poly_fit:
            self._add_poly_fit_plot(data, cft_params, times_dict, ax, bbox)

        ax._xmargin = 0

        if fig:
            ax.set_xlabel("Time")
            fig.tight_layout()
            fig.autofmt_xdate(rotation=0, ha='center')
            return fig, ax

    def _sanitize_cft_input(self, data: pd.DataFrame,
                            is_cft_interval: Optional[bool] = False,
                            compute_baseline: Optional[bool] = True,
                            hr_bl: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
        if is_cft_interval:
            df_hr_cft = data
        else:
            # extract CFT interval
            df_hr_cft = self.extract_cft_interval(data)

        if compute_baseline:
            if is_cft_interval:
                raise ValueError("`compute_baseline` must be set to False and `baseline_hr` be supplied "
                                 "when only CFT data is passed (`is_cft_interval` is `True`)")
            hr_bl = self.baseline_hr(data)
        else:
            if hr_bl is None:
                raise ValueError("`baseline_hr` must be supplied as parameter when `compute_baseline` is set to False!")

        return df_hr_cft, hr_bl

    def _add_peak_bradycardia_plot(self,
                                   data: pd.DataFrame,
                                   cft_params: Dict,
                                   cft_times: Dict,
                                   ax: plt.Axes,
                                   bbox: Dict) -> None:

        color_key = 'fau'
        brady_time = cft_params['peak_brady']
        hr_bl = cft_params['baseline_hr']
        max_hr_cft = float(self.extract_cft_interval(data).max())
        cft_start = cft_times['cft_start']

        color = colors.fau_color(color_key)
        color_adjust = colors.adjust_color(color_key, 1.5)

        # Peak Bradycardia vline
        ax.axvline(
            x=brady_time,
            ls="--",
            lw=2,
            alpha=0.6,
            color=color
        )

        # Peak Bradycardia marker
        ax.plot(
            brady_time,
            data.loc[brady_time],
            color=color,
            marker="o",
            markersize=7,
        )

        # Peak Bradycardia hline
        ax.hlines(
            y=data.loc[brady_time],
            xmin=brady_time,
            xmax=brady_time + pd.Timedelta(seconds=20),
            ls="--",
            lw=2,
            color=color_adjust,
            alpha=0.6,
        )

        # Peak Bradycardia arrow
        ax.annotate(
            "",
            xy=(
                brady_time + pd.Timedelta(seconds=10),
                float(data.loc[brady_time])),
            xytext=(brady_time + pd.Timedelta(seconds=10), hr_bl),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=color_adjust,
                shrinkA=0.0, shrinkB=0.0,
            )
        )

        # Peak Bradycardia Text
        ax.annotate(
            "$Peak_{CFT}$: " + "{:.1f} %".format(cft_params['peak_brady_percent']),
            xy=(brady_time + pd.Timedelta(seconds=10), float(data.loc[brady_time])),
            xytext=(7.5, -5),
            textcoords="offset points",
            size=14, bbox=bbox, ha='left', va='top'
        )

        # Peak Bradycardia Latency arrow
        ax.annotate(
            "",
            xy=(cft_start, max_hr_cft),
            xytext=(brady_time, max_hr_cft),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=color,
                shrinkA=0.0, shrinkB=0.0,
            )
        )

        # Peak Bradycardia Latency Text
        ax.annotate(
            "$Latency_{CFT}$: " + "{:.1f} s".format(cft_params['peak_brady_latency']),
            xy=(brady_time, max_hr_cft),
            xytext=(-7.5, 10),
            textcoords="offset points",
            size=14, bbox=bbox, ha='right', va='bottom'
        )

    def _add_baseline_plot(self,
                           data: pd.DataFrame,
                           cft_params: Dict,
                           cft_times: Dict,
                           ax: plt.Axes,
                           bbox: Dict) -> None:
        color_key = 'tech'

        # Baseline HR
        ax.hlines(
            y=cft_params['baseline_hr'],
            xmin=cft_times['plot_start'],
            xmax=cft_times['cft_end'],
            ls="--",
            lw=2,
            color=colors.fau_color(color_key),
            alpha=0.6
        )

    def _add_mean_bradycardia_plot(self,
                                   data: pd.DataFrame,
                                   cft_params: Dict,
                                   cft_times: Dict,
                                   ax: plt.Axes,
                                   bbox: Dict) -> None:

        color_key = 'wiso'
        mean_hr = cft_params['mean_hr_bpm']
        cft_start = cft_times['cft_start']
        cft_end = cft_times['cft_end']
        # Mean HR during CFT
        ax.hlines(
            y=mean_hr,
            xmin=cft_start,
            xmax=cft_end,
            ls="--",
            lw=2,
            color=colors.adjust_color(color_key),
            alpha=0.6
        )

        # Mean Bradycardia arrow
        ax.annotate(
            "",
            xy=(cft_end - pd.Timedelta(seconds=5), mean_hr),
            xytext=(cft_end - pd.Timedelta(seconds=5), cft_params['baseline_hr']),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=colors.adjust_color(color_key, 1.5),
                shrinkA=0.0, shrinkB=0.0,
            )
        )

        # Peak Bradycardia Text
        ax.annotate(
            "$Mean_{CFT}$: " + "{:.1f} %".format(cft_params['mean_brady_percent']),
            xy=(cft_end, mean_hr),
            xytext=(0, 0),
            textcoords="offset points",
            size=14, bbox=bbox, ha='left', va='bottom'
        )

    def _add_onset_plot(self,
                        data: pd.DataFrame,
                        cft_params: Dict,
                        cft_times: Dict,
                        ax: plt.Axes,
                        bbox: Dict) -> None:

        color_key = 'med'
        onset_time = cft_params['onset']
        onset_y = float(data.loc[onset_time])
        color = colors.fau_color(color_key)

        # CFT Onset vline
        ax.axvline(
            onset_time,
            ls="--",
            lw=2,
            alpha=0.6,
            color=color
        )

        # CFT Onset marker
        ax.plot(
            onset_time,
            onset_y,
            color=color,
            marker="o",
            markersize=7,
        )

        # CFT Onset arrow
        ax.annotate(
            "",
            xy=(onset_time, onset_y),
            xytext=(cft_times['cft_start'], onset_y),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=color,
                shrinkA=0.0, shrinkB=0.0,
            )
        )

        # CFT Onset Text
        ax.annotate(
            "$Onset_{CFT}$: " + "{:.1f} s".format(cft_params['onset_latency']),
            xy=(onset_time, onset_y),
            xytext=(-7.5, -10),
            textcoords="offset points",
            size=14, bbox=bbox, ha='right', va='top'
        )

    def _add_poly_fit_plot(self,
                           data: pd.DataFrame,
                           cft_params: Dict,
                           cft_times: Dict,
                           ax: plt.Axes,
                           bbox: Dict) -> None:

        color_key = 'phil'
        df_cft = self.extract_cft_interval(data)
        x_poly = df_cft.index.astype(int) / 1e9
        x_poly = x_poly - x_poly[0]
        y_poly = cft_params['poly_fit_a0'] * x_poly ** 2 + cft_params['poly_fit_a1'] * x_poly + cft_params[
            'poly_fit_a2']

        ax.plot(
            df_cft.index,
            y_poly,
            lw=2,
            color=colors.fau_color(color_key),
            alpha=0.6,
            zorder=2
        )
