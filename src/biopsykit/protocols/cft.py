"""Module representing the Cold Face Test (CFT) protocol."""
import datetime
import warnings
from typing import Optional, Tuple, Dict, Union, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

import biopsykit.colors as colors
from biopsykit.protocols import BaseProtocol
from biopsykit.signals.ecg.plotting import hr_plot
from biopsykit.utils.exceptions import FeatureComputationError


class CFT(BaseProtocol):
    """Class representing the Cold Face Test (CFT) and data collected while conducting the CFT."""

    def __init__(
        self,
        name: Optional[str] = None,
        structure: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """Class representing the Cold Face Test (CFT) and data collected while conducting the CFT.

        The typical structure of the CFT consists of three phases:

          * "Baseline": Time at rest before applying cold face stimulus
          * "CFT": Application of cold face stimulus
          * "Recovery": Time at rest after applying cold face stimulus


        Parameters
        ----------
        name : str
            name of protocol or ``None`` to use "CFT" as default name. Default: ``None``
        structure : dict, optional
            nested dictionary specifying the structure of the CFT.

            The typical structure of the CFT consists of three phases:

            * "Baseline": Time at rest before applying cold face stimulus
            * "CFT": Application of cold face stimulus
            * "Recovery": Time at rest after applying cold face stimulus

            The duration of each phase is specified in seconds.
            Typical durations are: 60s for *Baseline*, 120s for *CFT*, 60s for *Recovery*

            The start and duration of the CFT Exposure (``cft_start`` and ``cft_duration``) will be automatically
            extracted from the structure dictionary.
        **kwargs :
            additional parameters to be passed to ``CFT`` and its superclass, ``BaseProcessor``, such as:

            * ``cft_plot_params``: dictionary with parameters to style
              :meth:`~biopsykit.protocols.CFT.cft_plot`


        Examples
        --------
        >>> from biopsykit.protocols import CFT
        >>> # Example: CFT procedure consisting of three parts.
        >>>
        >>> structure = {
        >>>     "Baseline": 60,
        >>>     "CFT": 120,
        >>>     "Recovery": 60
        >>> }
        >>> CFT(name="CFT", structure=structure)

        """
        if name is None:
            name = "CFT"

        if structure is None:
            structure = {"Baseline": 60, "CFT": 120, "Recovery": 60}

        cft_start = structure.setdefault("Baseline", 0)
        if cft_start < 0:
            raise ValueError("'Baseline' duration must be non-negative!")

        cft_duration = structure.setdefault("CFT", 120)
        if cft_duration <= 0:
            raise ValueError("'CFT' duration must be positive!")

        self.cft_start: int = cft_start
        """Start of Cold Face Exposure in seconds. It is assumed that the time between the *beginning* of the
        recording and the *start* of the Cold Face Exposure is the Baseline. Default: 60 seconds"""

        self.cft_duration: int = cft_duration
        """Duration of the Cold Face Exposure in seconds. Default: 120 seconds"""

        cft_plot_params = {
            "background_color": ["#e0e0e0", "#9e9e9e", "#757575"],
            "background_alpha": [0.5, 0.5, 0.5],
            "phase_names": ["Baseline", "Cold Face Test", "Recovery"],
        }
        cft_plot_params.update(kwargs.get("cft_plot_params", {}))
        self.cft_plot_params: Dict[str, Any] = cft_plot_params

        super().__init__(name, structure, **kwargs)

    def compute_cft_parameter(
        self,
        data: pd.DataFrame,
        index: Optional[str] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Dict, pd.DataFrame]:
        """Compute CFT parameter.

        This function computes the following CFT parameter and returns the result in a dataframe
        (or, optionally, as dictionary):

        * Baseline Heart Rate (see :meth:`~biopsykit.protocols.CFT.baseline_hr` for further information)
        * CFT Onset (see :meth:`~biopsykit.protocols.CFT.onset` for further information)
        * Peak Bradycardia (see :meth:`~biopsykit.protocols.CFT.peak_bradycardia` for further information)
        * Mean Bradycardia (see :meth:`~biopsykit.protocols.CFT.mean_bradycardia` for further information)
        * Polynomial Fit on CFT Reaction (see :meth:`~biopsykit.protocols.CFT.poly_fit` for further information)


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        index : str, optional
            index value of resulting dataframe. Not needed if dictionary should be returned.
        return_dict : bool, optional
            ``True`` to return a dictionary with CFT parameters, ``False`` to return a dataframe. Default: ``False``


        Returns
        -------
        :class:`~pandas.DataFrame` or dict
            dataframe or dictionary with CFT parameter

        """
        if isinstance(index, str):
            index = [index]

        dict_cft = {}
        cft_parameter = ["onset", "peak_bradycardia", "mean_bradycardia", "poly_fit"]

        df_cft = self.extract_cft_interval(data)

        hr_baseline = self.baseline_hr(data)
        dict_cft["baseline_hr"] = hr_baseline
        dict_cft["cft_start_idx"] = data.index.get_loc(df_cft.index[0])
        for parameter in cft_parameter:
            dict_cft.update(
                getattr(self, parameter)(
                    data=df_cft,
                    is_cft_interval=True,
                    compute_baseline=False,
                    hr_baseline=hr_baseline,
                )
            )

        if return_dict:
            return dict_cft
        return pd.DataFrame([dict_cft], index=index)

    def baseline_hr(self, data: pd.DataFrame) -> float:
        """Compute mean heart rate during Baseline Interval.

        The Baseline Interval is data in the interval [``0``, ``cft_start``].

        .. warning::
            If ``cft_start`` is 0, it is assumed that no Baseline is present and the first heart rate value
            in the dataframe is used as CFT Baseline.


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data


        Returns
        -------
        float
            mean heart rate during Baseline Interval


        Raises
        ------
        :ex:`~biopsykit.utils.exceptions.FeatureComputationError`
            if data is shorter than the expected duration of the Baseline interval

        """
        # start of baseline = start of recording
        bl_start = data.index[0]
        if self.cft_start == 0:
            warnings.warn(
                "cft_start is 0, no baseline can be extracted! "
                "Using the first heart rate value in the dataframe as baseline."
            )
        # end of baseline = start of CFT
        if isinstance(data.index, pd.DatetimeIndex):
            bl_end = bl_start + pd.Timedelta(seconds=self.cft_start)
            if bl_end > data.index[-1]:
                raise FeatureComputationError(
                    "Error computing Baseline heart rate! "
                    "The provided data is shorter than the expected Baseline interval."
                )
            # heart rate during Baseline
            hr_baseline = data.between_time(bl_start.time(), bl_end.time())
        else:
            bl_end = bl_start + self.cft_start
            if bl_end > data.index[-1]:
                raise FeatureComputationError(
                    "Error computing Baseline heart rate! The provided data is shorter than the Baseline interval."
                )
            hr_baseline = data.loc[bl_start:bl_end]
        return float(hr_baseline.mean())

    def extract_cft_interval(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract interval during which CFT was applied.

        This function extracts only the part of the data during the "actual" Cold Face Test, i.e.,
        the time during which the cold face stimulus was applied.


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data


        Returns
        -------
        :class:`~pandas.DataFrame`
            data during application of cold face stimulus

        """
        if isinstance(data.index, pd.DatetimeIndex):
            cft_start = data.index[0] + pd.Timedelta(seconds=self.cft_start)
            cft_end = data.index[0] + pd.Timedelta(seconds=self.cft_start + self.cft_duration)
            return data.between_time(cft_start.time(), cft_end.time())
        return data.loc[self.cft_start : self.cft_start + self.cft_duration]

    def onset(
        self,
        data: pd.DataFrame,
        is_cft_interval: Optional[bool] = False,
        compute_baseline: Optional[bool] = True,
        hr_baseline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute **CFT onset**.

        The CFT onset is defined as the time point after beginning of the CFT Interval where three consecutive
        heart beats are lower than the Baseline heart rate (typically the Interval directly before the CFT).

        This function computes the following CFT onset parameter:

        * ``onset``: location of CFT onset. This value is the same datatype as the index of ``data``
          (i.e., either a absolute datetime timestamp or a relative timestamp in time since recording).
        * ``onset_latency``: CFT onset latency, i.e., the duration between beginning of the CFT Interval and
          CFT onset in seconds.
        * ``onset_idx``: location of CFT onset as array index
        * ``onset_hr``: heart rate at CFT onset in bpm
        * ``onset_hr_percent``: relative change of CFT onset heart rate compared to Baseline heart rate in percent.
        * ``onset_slope``: Slope between Baseline heart rate and CFT onset heart rate, computed as:
          `onset_slope = (onset_hr - baseline_hr) / onset_latency`


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        is_cft_interval : bool, optional
            ``True`` if the heart rate data passed via ``data`` contains only the CFT Interval,
            ``False`` if it contains the data during the whole CFT procedure. Default: ``False``
        compute_baseline : bool, optional
            ``True`` if Baseline Interval is included in data passed via ``data`` and Baseline heart rate
            should be computed or ``False`` if Baseline heart rate is passed separately via ``hr_baseline``.
            Default: ``True``
        hr_baseline : float, optional
            mean heart rate during Baseline Interval or ``None`` if Baseline interval is present in ``data`` and
            Baseline heart rate is computed from there. Default: ``None``


        Returns
        -------
        dict
            dictionary with CFT onset parameter

        """
        df_hr_cft, hr_baseline = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_baseline)

        # bradycardia mask (True where heart rate is below baseline, False otherwise)
        hr_brady = df_hr_cft < hr_baseline
        # bradycardia borders (1 where we have a change between lower and higher heart rate)
        brady_border = np.abs(np.ediff1d(hr_brady.astype(int), to_begin=0))
        # filter out the phases where we have at least 3 heart rate values lower than baseline
        brady_phases = hr_brady.groupby([np.cumsum(brady_border)]).filter(lambda df: df.sum() >= 3)
        # CFT onset is the third beat
        onset = brady_phases.index[2]
        # TODO check index handling again...
        onset_latency = (onset - df_hr_cft.index[0]).total_seconds()
        onset_idx = df_hr_cft.index.get_loc(onset)

        # heart rate at onset point
        hr_onset = np.squeeze(df_hr_cft.loc[onset])
        return {
            "onset": onset,
            "onset_latency": onset_latency,
            "onset_idx": onset_idx,
            "onset_hr": hr_onset,
            "onset_hr_percent": (1 - hr_onset / hr_baseline) * 100,
            "onset_slope": (hr_onset - hr_baseline) / onset_latency,
        }

    def peak_bradycardia(
        self,
        data: pd.DataFrame,
        is_cft_interval: Optional[bool] = False,
        compute_baseline: Optional[bool] = True,
        hr_baseline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute **CFT peak bradycardia**.

        The CFT peak bradycardia is defined as the maximum bradycardia (i.e., the minimum heart rate)
        during the CFT Interval.

        This function computes the following CFT peak bradycardia parameter:

        * ``peak_brady``: location of CFT peak bradycardia. This value is the same datatype as the index of ``data``
          (i.e., either a absolute datetime timestamp or a relative timestamp in time since recording).
        * ``peak_brady_latency``: CFT peak bradycardia latency, i.e., the duration between beginning of the
          CFT Interval and CFT peak bradycardia in seconds.
        * ``peak_brady_idx``: location of CFT peak bradycardia as array index
        * ``peak_brady_bpm``: CFT peak bradycardia in bpm
        * ``peak_brady_percent``: Relative change of CFT peak bradycardia heart rate compared to Baseline
          heart rate in percent.
        * ``peak_brady_slope``: Slope between Baseline heart rate and CFT peak bradycardia heart rate, computed as:
          ``peak_brady_slope = (peak_brady_bpm - baseline_hr) / peak_brady_latency``


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        is_cft_interval : bool, optional
            ``True`` if the heart rate data passed via ``data`` contains only the CFT Interval,
            ``False`` if it contains the data during the whole CFT procedure. Default: ``False``
        compute_baseline : bool, optional
            ``True`` if Baseline Interval is included in data passed via ``data`` and Baseline heart rate
            should be computed or ``False`` if Baseline heart rate is passed separately via ``hr_baseline``.
            Default: ``True``
        hr_baseline : float, optional
            mean heart rate during Baseline Interval or ``None`` if Baseline interval is present in ``data`` and
            Baseline heart rate is computed from there. Default: ``None``


        Returns
        -------
        dict
            dictionary with CFT peak bradycardia parameter

        """
        df_hr_cft, hr_baseline = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_baseline)

        peak_brady = np.squeeze(df_hr_cft.idxmin())
        peak_brady_latency = (peak_brady - df_hr_cft.index[0]).total_seconds()
        peak_brady_idx = df_hr_cft.index.get_loc(peak_brady)
        hr_brady = np.squeeze(df_hr_cft.loc[peak_brady])
        return {
            "peak_brady": peak_brady,
            "peak_brady_latency": peak_brady_latency,
            "peak_brady_idx": peak_brady_idx,
            "peak_brady_bpm": hr_brady - hr_baseline,
            "peak_brady_percent": (hr_brady / hr_baseline - 1) * 100,
            "peak_brady_slope": (hr_brady - hr_baseline) / peak_brady_latency,
        }

    def mean_bradycardia(
        self,
        data: pd.DataFrame,
        is_cft_interval: Optional[bool] = False,
        compute_baseline: Optional[bool] = True,
        hr_baseline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute **CFT mean bradycardia**.

        The CFT mean bradycardia is defined as the mean bradycardia (i.e., the mean decrease of heart rate)
        during the CFT Interval.

        This function computes the following CFT mean bradycardia parameter:

        * ``mean_hr_bpm``: average heart rate during CFT Interval in bpm
        * ``mean_brady_bpm``: average bradycardia during CFT Interval, computed as:
          ``mean_brady_bpm = mean_hr_bpm - hr_baseline``
        * ``mean_brady_percent``: relative change of CFT mean bradycardia heart rate compared to Baseline
          heart rate in percent


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        is_cft_interval : bool, optional
            ``True`` if the heart rate data passed via ``data`` contains only the CFT Interval,
            ``False`` if it contains the data during the whole CFT procedure. Default: ``False``
        compute_baseline : bool, optional
            ``True`` if Baseline Interval is included in data passed via ``data`` and Baseline heart rate
            should be computed or ``False`` if Baseline heart rate is passed separately via ``hr_baseline``.
            Default: ``True``
        hr_baseline : float, optional
            mean heart rate during Baseline Interval or ``None`` if Baseline interval is present in ``data`` and
            Baseline heart rate is computed from there. Default: ``None``


        Returns
        -------
        dict
            dictionary with CFT mean bradycardia parameter

        """
        df_hr_cft, hr_baseline = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_baseline)

        hr_mean = np.squeeze(df_hr_cft.mean())

        return {
            "mean_hr_bpm": hr_mean,
            "mean_brady_bpm": hr_mean - hr_baseline,
            "mean_brady_percent": (hr_mean / hr_baseline - 1) * 100,
        }

    def poly_fit(
        self,
        data: pd.DataFrame,
        is_cft_interval: Optional[bool] = False,
        compute_baseline: Optional[bool] = True,
        hr_baseline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute **CFT polynomial fit**.

        The CFT polynomial fit is computed by applying a 2nd order least-squares polynomial fit to the heart rate
        during the CFT Interval because the CFT-induced bradycardia and the following recovery is assumed to follow
        a polynomial function.

        This function computes the following CFT polynomial fit parameter:

        * ``poly_fit_a{0-2}``: constants of the polynomial ``p(x) = p[2] * x**deg + p[1]* x + p[0]``


        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        is_cft_interval : bool, optional
            ``True`` if the heart rate data passed via ``data`` contains only the CFT Interval,
            ``False`` if it contains the data during the whole CFT procedure. Default: ``False``
        compute_baseline : bool, optional
            ``True`` if Baseline Interval is included in data passed via ``data`` and Baseline heart rate
            should be computed or ``False`` if Baseline heart rate is passed separately via ``hr_baseline``.
            Default: ``True``
        hr_baseline : float, optional
            mean heart rate during Baseline Interval or ``None`` if Baseline interval is present in ``data`` and
            Baseline heart rate is computed from there. Default: ``None``


        Returns
        -------
        dict
            dictionary with CFT polynomial fit parameter

        """
        df_hr_cft, hr_baseline = self._sanitize_cft_input(data, is_cft_interval, compute_baseline, hr_baseline)

        # get time points in seconds
        # TODO check index type
        idx_s = df_hr_cft.index.astype(int) / 1e9
        idx_s = idx_s - idx_s[0]

        # apply a 2nd degree polynomial fit
        poly = np.polyfit(idx_s, np.squeeze(df_hr_cft.values), deg=2)
        return {"poly_fit_a{}".format(i): p for i, p in enumerate(reversed(poly))}

    def _sanitize_cft_input(
        self,
        data: pd.DataFrame,
        is_cft_interval: Optional[bool] = False,
        compute_baseline: Optional[bool] = True,
        hr_baseline: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, float]:
        """Sanitize CFT input.

        Most functions for computing CFT parameter expect multiple possible combinations of input parameter:

        * Either data over the whole duration of the CFT procedure (Baseline, CFT, Recovery):
          Then, the CFT Interval will be extracted and Baseline heart rate will be computed based on the
          ``cft_start`` and ``cft_duration`` parameters of the ``CFT`` object
          (``compute_baseline`` must then be set to ``True`` – the default)
        * Or only data during the CFT interval (``is_cft_interval`` must be set to ``True``).
          Then, the Baseline heart rate muse be explicitly provided via ``hr_baseline`` parameter and
          ``compute_baseline`` must be set to ``False``

        This function sanitizes the input and, independent from the input, always returns a tuple with the data cut
        to the CFT Interval and the mean heart rate during the Baseline Interval.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        is_cft_interval : bool, optional
            ``True`` if the heart rate data passed via ``data`` contains only the CFT Interval,
            ``False`` if it contains the data during the whole CFT procedure. Default: ``False``
        compute_baseline : bool, optional
            ``True`` if Baseline Interval is included in data passed via ``data`` and Baseline heart rate
            should be computed or ``False`` if Baseline heart rate is passed separately via ``hr_baseline``.
            Default: ``True``
        hr_baseline : float, optional
            mean heart rate during Baseline Interval or ``None`` if Baseline interval is present in ``data`` and
            Baseline heart rate is computed from there. Default: ``None``

        Returns
        -------
        data_cft : :class:`~pandas.DataFrame`
            heart rate data during CFT Interval

        """
        if is_cft_interval:
            data_cft = data
        else:
            # extract CFT interval
            data_cft = self.extract_cft_interval(data)

        if compute_baseline:
            if is_cft_interval:
                raise ValueError(
                    "`compute_baseline` must be set to False and `baseline_hr` be supplied "
                    "when only CFT data is passed (`is_cft_interval` is `True`)"
                )
            hr_baseline = self.baseline_hr(data)
        else:
            if hr_baseline is None:
                raise ValueError("`baseline_hr` must be supplied as parameter when `compute_baseline` is set to False!")

        return data_cft, hr_baseline

    def cft_plot(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Draw Cold Face Test (CFT) plot.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data
        **kwargs: dict, optional
            optional parameters to be passed to the plot, such as:

            * ``time_baseline`` : duration of Baseline Interval to include in plot or ``None`` to include the
              whole Baseline Interval in the plot.
            * ``time_recovery`` : duration of Recovery Interval to include in plot or ``None`` to include the
              whole Recovery Interval in the plot.
            * ``plot_datetime_index`` : ``True`` to plot x axis with absolute time (:class:`~pandas.DatetimeIndex`),
              or ``False`` to plot data with relative time (starting from second 0). Default: ``False``
            * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created and
              returned.
            * ``figsize``: tuple specifying figure dimensions
            * ``ylims``: list to manually specify y axis limits, float to specify y axis margin
              (see :meth:`matplotlib.axes.Axes.margins` for further information), or ``None`` to automatically
              infer y axis limits.
            * ``plot_onset``: whether to plot CFT onset annotations or not: Default: ``True``
            * ``plot_peak_brady``: whether to plot CFT peak bradycardia annotations or not: Default: ``True``
            * ``plot_mean``: whether to plot CFT mean bradycardia annotations or not. Default: ``True``
            * ``plot_baseline``: whether to plot heart rate baseline annotations or not. Default: ``True``
            * ``plot_poly_fit``: whether to plot CFT polynomial fit annotations or not. Default: ``True``

        Returns
        -------
        fig : :class:`~matplotlib.figure.Figure`
            figure object
        ax : :class:`~matplotlib.axes.Axes`
            axes object

        """
        ax: plt.Axes = kwargs.pop("ax", None)
        figsize = kwargs.get("figsize", (12, 5))
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        time_baseline = kwargs.get("time_baseline", self.structure["Baseline"])
        time_recovery = kwargs.get("time_recovery", self.structure["Recovery"])

        data = data.copy()
        cft_params = self.compute_cft_parameter(data, return_dict=True)

        if not kwargs.get("plot_datetime_index", False):
            data.index = (data.index - data.index[0]).astype(int) / 1e9

        times_dict = self._cft_plot_get_cft_times(data, time_baseline, time_recovery)
        df_plot = self._cft_plot_extract_plot_interval(data, times_dict)

        bbox = dict(
            fc=(1, 1, 1, plt.rcParams["legend.framealpha"]),
            ec=plt.rcParams["legend.edgecolor"],
            boxstyle="round",
        )

        hr_plot(heart_rate=df_plot, ax=ax, plot_mean=False, show_legend=False)
        self._cft_plot_add_phase_annotations(ax, times_dict, **kwargs)
        self._cft_plot_add_param_annotations(data, cft_params, times_dict, ax, bbox, **kwargs)

        self._cft_plot_style_axis(data, ax, **kwargs)

        fig.tight_layout()
        fig.autofmt_xdate(rotation=0, ha="center")
        return fig, ax

    @staticmethod
    def _cft_plot_style_axis(data: pd.DataFrame, ax: plt.Axes, **kwargs):
        ylims = kwargs.get("ylims", None)
        if isinstance(ylims, (tuple, list)):
            ax.set_ylim(ylims)
        else:
            ymargin = 0.1
            if isinstance(ylims, float):
                ymargin = ylims
            ax.margins(x=0, y=ymargin)

        if isinstance(data.index, pd.DatetimeIndex):
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("Time [s]")

    def _cft_plot_get_cft_times(
        self, data: pd.DataFrame, time_baseline: int, time_recovery: int
    ) -> Dict[str, Union[int, datetime.datetime]]:
        if isinstance(data.index, pd.DatetimeIndex):
            cft_start = data.index[0] + pd.Timedelta(seconds=self.cft_start)
            plot_start = cft_start - pd.Timedelta(seconds=time_baseline)
            cft_end = cft_start + pd.Timedelta(seconds=self.cft_duration)
            plot_end = cft_end + pd.Timedelta(seconds=time_recovery)
        else:
            cft_start = data.index[0] + self.cft_start
            plot_start = cft_start - time_baseline
            cft_end = cft_start + self.cft_duration
            plot_end = cft_end + time_recovery

        cft_times = {
            "plot_start": plot_start,
            "cft_start": cft_start,
            "cft_end": cft_end,
            "plot_end": plot_end,
        }
        return cft_times

    def _cft_plot_extract_plot_interval(  # pylint:disable=no-self-use
        self, data: pd.DataFrame, times_dict: Dict[str, Union[int, datetime.datetime]]
    ) -> pd.DataFrame:
        plot_start = times_dict["plot_start"]
        plot_end = times_dict["plot_end"]
        if isinstance(data.index, pd.DatetimeIndex):
            df_plot = data.between_time(plot_start.time(), plot_end.time())
        else:
            df_plot = data.loc[plot_start:plot_end]

        return df_plot

    def _cft_plot_add_phase_annotations(
        self, ax: plt.Axes, times_dict: Dict[str, Union[int, datetime.datetime]], **kwargs
    ):
        times = list(zip(list(times_dict.values()), list(times_dict.values())[1:]))
        bg_colors = kwargs.get("background_color", self.cft_plot_params["background_color"])
        bg_alphas = kwargs.get("background_alpha", self.cft_plot_params["background_alpha"])
        names = kwargs.get("phase_names", self.cft_plot_params["phase_names"])

        for (start, end), bg_color, bg_alpha, name in zip(times, bg_colors, bg_alphas, names):
            ax.axvspan(xmin=start, xmax=end, color=bg_color, alpha=bg_alpha, lw=0)
            ax.text(
                x=start + 0.5 * (end - start),
                y=0.95,
                transform=ax.get_xaxis_transform(),
                s=name,
                # bbox=bbox,
                ha="center",
                va="center",
            )
        rect = mpatch.Rectangle(
            xy=(0, 0.9),
            width=1,
            height=0.1,
            color="white",
            alpha=0.4,
            zorder=3,
            lw=0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

    def _cft_plot_add_param_annotations(self, data, cft_params, times_dict, ax, bbox, **kwargs):
        if kwargs.get("plot_baseline", True):
            self._cft_plot_add_baseline(cft_params, times_dict, ax)
        if kwargs.get("plot_mean", True):
            self._cft_plot_add_mean_bradycardia(data, cft_params, times_dict, ax, bbox)
        if kwargs.get("plot_onset", True):
            self._cft_plot_add_onset(data, cft_params, times_dict, ax, bbox)
        if kwargs.get("plot_peak_brady", True):
            self._cft_plot_add_peak_bradycardia(data, cft_params, times_dict, ax, bbox)
        if kwargs.get("plot_poly_fit", True):
            self._cft_plot_add_poly_fit(data, cft_params, ax)

    def _cft_plot_add_peak_bradycardia(
        self,
        data: pd.DataFrame,
        cft_params: Dict,
        cft_times: Dict,
        ax: plt.Axes,
        bbox: Dict,
    ) -> None:

        color_key = "fau"
        if isinstance(data.index, pd.DatetimeIndex):
            brady_loc = cft_params["peak_brady"]
            brady_x = brady_loc
            brady_y = float(data.loc[brady_loc])
        else:
            brady_loc = cft_params["cft_start_idx"] + cft_params["peak_brady_idx"]
            brady_x = data.index[brady_loc]
            brady_y = float(data.iloc[brady_loc])

        hr_baseline = cft_params["baseline_hr"]
        max_hr_cft = float(self.extract_cft_interval(data).max())
        cft_start = cft_times["cft_start"]

        color = colors.fau_color(color_key)
        color_adjust = colors.adjust_color(color_key, 1.5)

        # Peak Bradycardia vline
        ax.axvline(x=brady_x, ls="--", lw=2, alpha=0.6, color=color)

        # Peak Bradycardia marker
        ax.plot(
            brady_x,
            brady_y,
            color=color,
            marker="o",
            markersize=7,
        )

        # Peak Bradycardia hline
        if isinstance(data.index, pd.DatetimeIndex):
            xmax = brady_x + pd.Timedelta(seconds=20)
        else:
            xmax = brady_x + 20

        ax.hlines(
            y=brady_y,
            xmin=brady_x,
            xmax=xmax,
            ls="--",
            lw=2,
            color=color_adjust,
            alpha=0.6,
        )

        # Peak Bradycardia arrow
        if isinstance(data.index, pd.DatetimeIndex):
            brady_x_offset = brady_x + pd.Timedelta(seconds=10)
        else:
            brady_x_offset = brady_x + 10

        ax.annotate(
            "",
            xy=(brady_x_offset, brady_y),
            xytext=(brady_x_offset, hr_baseline),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=color_adjust,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
        )

        # Peak Bradycardia Text
        ax.annotate(
            "$Peak_{CFT}$: " + "{:.1f} %".format(cft_params["peak_brady_percent"]),
            xy=(brady_x_offset, brady_y),
            xytext=(10, -5),
            textcoords="offset points",
            bbox=bbox,
            ha="left",
            va="top",
        )

        # Peak Bradycardia Latency arrow
        ax.annotate(
            "",
            xy=(cft_start, max_hr_cft),
            xytext=(brady_x, max_hr_cft),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=color,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
        )

        # Peak Bradycardia Latency Text
        ax.annotate(
            "$Latency_{CFT}$: " + "{:.1f} s".format(cft_params["peak_brady_latency"]),
            xy=(brady_x, max_hr_cft),
            xytext=(-7.5, 10),
            textcoords="offset points",
            bbox=bbox,
            ha="right",
            va="bottom",
        )

    def _cft_plot_add_baseline(  # pylint:disable=no-self-use
        self,
        cft_params: Dict,
        cft_times: Dict,
        ax: plt.Axes,
    ) -> None:
        color_key = "tech"

        # Baseline HR
        ax.hlines(
            y=cft_params["baseline_hr"],
            xmin=cft_times["plot_start"],
            xmax=cft_times["cft_end"],
            ls="--",
            lw=2,
            color=colors.fau_color(color_key),
            alpha=0.6,
        )

    def _cft_plot_add_mean_bradycardia(  # pylint:disable=no-self-use
        self,
        data: pd.DataFrame,
        cft_params: Dict,
        cft_times: Dict,
        ax: plt.Axes,
        bbox: Dict,
    ) -> None:

        color_key = "wiso"
        mean_hr = cft_params["mean_hr_bpm"]
        cft_start = cft_times["cft_start"]
        cft_end = cft_times["cft_end"]

        # Mean HR during CFT
        ax.hlines(
            y=mean_hr,
            xmin=cft_start,
            xmax=cft_end,
            ls="--",
            lw=2,
            color=colors.adjust_color(color_key),
            alpha=0.6,
        )

        if isinstance(data.index, pd.DatetimeIndex):
            x_offset = cft_end - pd.Timedelta(seconds=5)
        else:
            x_offset = cft_end - 5

        # Mean Bradycardia arrow
        ax.annotate(
            "",
            xy=(x_offset, mean_hr),
            xytext=(x_offset, cft_params["baseline_hr"]),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=colors.adjust_color(color_key, 1.5),
                shrinkA=0.0,
                shrinkB=0.0,
            ),
        )

        # Mean Bradycardia Text
        ax.annotate(
            "$Mean_{CFT}$: " + "{:.1f} %".format(cft_params["mean_brady_percent"]),
            xy=(x_offset, mean_hr),
            xytext=(10, -5),
            textcoords="offset points",
            bbox=bbox,
            ha="left",
            va="top",
        )

    def _cft_plot_add_onset(  # pylint:disable=no-self-use
        self,
        data: pd.DataFrame,
        cft_params: Dict,
        cft_times: Dict,
        ax: plt.Axes,
        bbox: Dict,
    ) -> None:

        color_key = "med"
        color = colors.fau_color(color_key)

        if isinstance(data.index, pd.DatetimeIndex):
            onset_idx = cft_params["onset"]
            onset_x = onset_idx
            onset_y = float(data.loc[onset_idx])
        else:
            onset_idx = cft_params["cft_start_idx"] + cft_params["onset_idx"]
            onset_y = float(data.iloc[onset_idx])
            onset_x = data.index[onset_idx]

        # CFT Onset vline
        ax.axvline(onset_x, ls="--", lw=2, alpha=0.6, color=color)

        # CFT Onset marker
        ax.plot(
            onset_x,
            onset_y,
            color=color,
            marker="o",
            markersize=7,
        )

        # CFT Onset arrow
        ax.annotate(
            "",
            xy=(onset_x, onset_y),
            xytext=(cft_times["cft_start"], onset_y),
            arrowprops=dict(
                arrowstyle="<->",
                lw=2,
                color=color,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
        )

        # CFT Onset Text
        ax.annotate(
            "$Onset_{CFT}$: " + "{:.1f} s".format(cft_params["onset_latency"]),
            xy=(onset_x, onset_y),
            xytext=(-10, -10),
            textcoords="offset points",
            bbox=bbox,
            ha="right",
            va="top",
        )

    def _cft_plot_add_poly_fit(
        self,
        data: pd.DataFrame,
        cft_params: Dict,
        ax: plt.Axes,
    ) -> None:

        color_key = "phil"
        df_cft = self.extract_cft_interval(data)
        if isinstance(df_cft.index, pd.DatetimeIndex):
            x_poly = df_cft.index.astype(int) / 1e9
        else:
            x_poly = df_cft.index

        x_poly = x_poly - x_poly[0]
        y_poly = (
            cft_params["poly_fit_a2"] * x_poly ** 2 + cft_params["poly_fit_a1"] * x_poly + cft_params["poly_fit_a0"]
        )

        ax.plot(
            df_cft.index,
            y_poly,
            lw=2,
            color=colors.fau_color(color_key),
            alpha=0.6,
            zorder=2,
        )
