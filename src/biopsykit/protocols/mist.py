from typing import Dict, Tuple, Union, Optional, Sequence

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks

import biopsykit.signals.ecg as ecg
import biopsykit.colors as colors
import biopsykit.protocols.base as base
import biopsykit.protocols.plotting as plot

from biopsykit.utils.array_handling import interpolate_and_cut
from biopsykit.utils.data_processing import param_subphases


class MIST(base.BaseProtocol):
    """
    Class representing the Montreal Imaging Stress Task (MIST).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        phases: Optional[Sequence[str]] = None,
        subphases: Optional[Sequence[str]] = None,
        subphase_durations: Optional[Sequence[int]] = None,
    ):
        if name is None:
            name = "MIST"
        super().__init__(name)

        self.mist_times: Sequence[int] = [0, 30]

        self.phases: Sequence[str] = ["MIST1", "MIST2", "MIST3"]
        """
        MIST Phases
        
        Names of MIST phases
        """

        self.subphases: Sequence[str] = ["BL", "AT", "FB"]
        """
        MIST Subphases
        
        Names of MIST subphases
        """

        self.subphase_durations: Sequence[int] = [60, 240, 0]
        """
        MIST Subphase Durations
        
        Total duration of subphases in seconds
        """

        self.hr_ensemble_plot_params = {
            "colormap": colors.cmap_fau_blue("3_ens"),
            "line_styles": ["-", "--", ":"],
            "ensemble_alpha": 0.4,
            "background_color": ["#e0e0e0", "#9e9e9e", "#757575"],
            "background_alpha": [0.5, 0.5, 0.5],
            "fontsize": 14,
            "xaxis_label": r"Time [s]",
            "xaxis_minor_ticks": mticks.MultipleLocator(60),
            "yaxis_label": r"$\Delta$HR [%]",
            "legend_loc": "lower right",
            "legend_bbox_to_anchor": (0.99, 0.01),
            "phase_text": "MIST Phase {}",
            "end_phase_text": "End Phase {}",
            "end_phase_line_color": "#e0e0e0",
            "end_phase_line_style": "dashed",
            "end_phase_line_width": 2.0,
        }

        self.hr_mean_plot_params = {
            "colormap": colors.cmap_fau_blue("2_lp"),
            "line_styles": ["-", "--"],
            "markers": ["o", "P"],
            "background_color": ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
            "background_alpha": [0.5, 0.5, 0.5],
            "x_offsets": [0, 0.05],
            "fontsize": 14,
            "xaxis_label": "MIST Subphases",
            "yaxis_label": r"$\Delta$HR [%]",
            "phase_text": "MIST Phase {}",
        }

        self.saliva_params = {
            "test_text": "MIST",
            "xaxis_label": "Time relative to MIST start [min]",
        }

        self._update_mist_params(phases, subphases, subphase_durations)

    def __str__(self) -> str:
        return """{}
        Phases: {}
        Subphases: {}
        Subphase Durations: {}
        """.format(
            self.name, self.phases, self.subphases, self.subphase_durations
        )

    @property
    def mist_times(self):
        return self.test_times

    @mist_times.setter
    def mist_times(self, mist_times):
        self.test_times = mist_times

    def _update_mist_params(
        self,
        phases: Sequence[str],
        subphases: Sequence[str],
        subphase_durations: Sequence[int],
    ):
        if phases:
            self.phases = phases
        if subphases:
            self.subphases = subphases
        if subphase_durations:
            self.subphase_durations = subphase_durations

    def interpolate_and_cut_feedback_interval(
        self, dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Interpolates heart rate input to a frequency of 1 Hz and then cuts heart rate data of
        each subject to equal length, i.e. to the minimal duration of each MIST phase
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
        dict_hr_subject = dict_hr_subject.copy()
        # skip Part1 and Part2, extract only MIST Phases
        for subject_id, dict_subject in dict_hr_subject.items():
            for phase in ["Part1", "Part2"]:
                if phase in dict_subject:
                    dict_subject.pop(phase)

        return interpolate_and_cut(dict_hr_subject)

    def concat_phase_dict(
        self, dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]], **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Rearranges the 'HR subject dict' (see `util s.load_hr_excel_all_subjects`) into 'MIST Phase dict'.
        See ``bp.protocols.utils.concat_phase_dict`` for further information.

        Parameters
        ----------
        dict_hr_subject : dict
            'HR subject dict', i.e. a nested dict with heart rate data per MIST phase and subject
        **kwargs

        Returns
        -------
        dict
            'MIST dict', i.e. a dict with heart rate data of all subjects per MIST phase

        """
        if "phases" in kwargs:
            return super().concat_phase_dict(dict_hr_subject, kwargs["phases"])
        else:
            return super().concat_phase_dict(dict_hr_subject, self.phases)

    def split_subphases(
        self,
        phase_dict: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
        is_group_dict: Optional[bool] = False,
        **kwargs
    ) -> Union[
        Dict[str, Dict[str, pd.DataFrame]],
        Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    ]:
        """
        Splits a `MIST Phase dict` (or a dict of such, in case of multiple groups,
        see ``bp.protocols.utils.concat_dict``)
        into a `MIST Subphase dict`. See ``bp.protocols.utils.split_subphases`` for further information.

        Parameters
        ----------
        phase_dict : dict
            'Phase dict' or nested dict of 'Phase dicts' if `is_group_dict` is ``True``
        is_group_dict : bool, optional
            ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``
        **kwargs

        Returns
        -------
        dict
            'Subphase dict' with course of HR data per Stress Test phase, subphase and subject, respectively or
            nested dict of 'Subphase dicts' if `is_group_dict` is ``True``

        """
        if "subphase_times" in kwargs and "subphases" in kwargs:
            subphase_times = kwargs["subphase_times"]
            subphase_names = kwargs["subphases"]
        else:
            subphase_times = self.get_mist_times(
                phase_dict=phase_dict, is_group_dict=is_group_dict
            )
            subphase_names = self.subphases
        return super().split_subphases(
            data=phase_dict,
            subphase_names=subphase_names,
            subphase_times=subphase_times,
            is_group_dict=is_group_dict,
        )

    @classmethod
    def split_groups(
        cls,
        phase_dict: Dict[str, pd.DataFrame],
        condition_dict: Dict[str, Sequence[str]],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Splits 'MIST Phase dict' into group dict, i.e. one 'MIST Phase dict' per group.

        Parameters
        ----------
        phase_dict : dict
            'MIST Phase dict' to be split in groups. See ``bp.protocols.utils.concat_phase_dict``
            for further information
        condition_dict : dict
            dictionary of group membership. Keys are the different groups, values are lists of subject IDs that
            belong to the respective group

        Returns
        -------
        dict
            nested group dict with one 'MIST Phase dict' per group
        """

        return super().split_groups(phase_dict, condition_dict)

    def get_mist_times(
        self,
        mist_dur: Optional[Sequence[int]] = None,
        phase_dict: Optional[
            Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]
        ] = None,
        is_group_dict: Optional[bool] = False,
    ) -> Sequence[Tuple[int, int]]:
        """
        Computes the start and end times of each MIST subphase. It is assumed that all MIST subphases,
        except Feedback, have equal length for all MIST phases. The length of the Feedback subphase is computed as
        the maximum length of all MIST phases.

        To compute the MIST subphase durations either pass a list with total MIST durations per phase or a
        'MIST Phase dict' (see ``MIST.concat_mist_dict`` for further information) or 'grouped MIST dict'
        (if ``is_grouped_dict`` is set to ``True``, see ``MIST.split_mist_groups`` for further information).
        The length of the dataframe then corresponds to the total MIST duration per phase.

        Parameters
        ----------
        mist_dur : list, optional
            a list where each entry of the list is the total duration of one MIST phase.
        phase_dict : dict, optional
            'MIST Phase dict' or grouped 'MIST Phase dict' (if ``is_group_dict`` is set to ``True``).
            The length of the dataframes (= dict entries) correspond to the total duration of the MIST phases
        is_group_dict : bool, optional
            ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``

        Returns
        -------
        list
            a list with tuples of MIST subphase start and end times (in seconds)

        Raises
        ------
        ValueError
            if neither ``mist_dur`` nor ``phase_dict`` are passed as arguments
        """

        if mist_dur is None and phase_dict is None:
            raise ValueError(
                "Either `mist_dur` or `phase_dict` must be supplied as parameter!"
            )

        if mist_dur:
            # ensure numpy
            mist_dur = np.array(mist_dur)
        else:
            if is_group_dict:
                # Grouped MIST Phase dict
                mist_dur = np.array(
                    [[len(v) for v in d.values()] for d in phase_dict.values()]
                )
                if not (mist_dur == mist_dur[0]).all():
                    # ensure that durations of all groups are equal
                    raise ValueError(
                        "All groups are expected to have the same durations for the single phases!"
                    )
                mist_dur = mist_dur[0]
            else:
                # MIST Phase dict
                mist_dur = np.array([len(v) for v in phase_dict.values()])

        # compute the duration to the beginning of Feedback subphase
        dur_to_fb = sum(self.subphase_durations)
        # (variable) duration of the feedback intervals: total MIST duration - duration to end of AT
        dur_fb = mist_dur - dur_to_fb

        # set FB duration
        subph_dur = np.array(self.subphase_durations)
        subph_dur[-1] = max(dur_fb)
        # cumulative times
        times_cum = np.cumsum(np.array(subph_dur))
        # compute start/end times per subphase
        return [
            (start, end)
            for start, end in zip(np.append([0], times_cum[:-1]), times_cum)
        ]

    def param_subphases(
        self,
        ecg_processor: Optional[ecg.EcgProcessor] = None,
        dict_ecg: Optional[Dict[str, pd.DataFrame]] = None,
        dict_rpeaks: Optional[Dict[str, pd.DataFrame]] = None,
        param_types: Optional[Union[str, Sequence[str]]] = "all",
        sampling_rate: Optional[int] = 256,
        include_total: Optional[bool] = True,
        title: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Computes specified parameters (HRV / RSA / ...) over all MIST phases and subphases.

        To use this function, either simply pass an ``EcgProcessor`` object or two dictionaries
        ``dict_ecg`` and ``dict_rpeaks`` resulting from ``EcgProcessor.ecg_process()``.

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
        title : str, optional
            Optional title of the processing progress bar. Default: ``None``

        Returns
        -------
        pd.DataFrame
            dataframe with computed parameters over the single MIST subphases
        """

        return param_subphases(
            ecg_processor=ecg_processor,
            dict_ecg=dict_ecg,
            dict_rpeaks=dict_rpeaks,
            subphases=self.subphases,
            subphase_durations=self.subphase_durations,
            include_total=include_total,
            param_types=param_types,
            sampling_rate=sampling_rate,
            title=title,
        )

    def hr_mean_se_subphases(
        self,
        data: Union[
            Dict[str, Dict[str, pd.DataFrame]],
            Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        ],
        is_group_dict: Optional[bool] = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Computes the heart rate mean and standard error per MIST subphase over all subjects.
        See ``bp.protocols.utils.hr_course`` for further information.

        Parameters
        ----------
        data : dict
            nested dictionary containing heart rate data.
        is_group_dict : boolean, optional
            ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
            Default: ``False``

        Returns
        -------
        dict or pd.DataFrame
            'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.
        """

        return super()._mean_se_subphases(
            data, subphases=self.subphases, is_group_dict=is_group_dict
        )

    def hr_ensemble_plot(
        self,
        data: Dict[str, pd.DataFrame],
        plot_params: Optional[Dict] = None,
        ylims: Optional[Sequence[float]] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> Union[Tuple[plt.Figure, plt.Axes], None]:
        """
        Plots the course of heart rate during each MIST phase continuously as ensemble plot (mean ± standard error).
        Simply pass a 'MIST dict' dictionary with one pandas heart rate dataframe per MIST phase
        (see ``MIST.concat_mist_dicts`` for further explanation), i.e. heart rate data with one column per subject.

        Parameters
        ----------
        data : dict
            dict with heart rate data to plot
        plot_params : dict, optional
            dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
            For an overview of parameters and their default values, see `mist.hr_ensemble_params`
        ylims : list, optional
            y axis limits or ``None`` to infer y axis limits from data. Default: ``None``
        ax : plt.Axes, optional
            Axes to plot on, otherwise create a new one. Default: ``None``

        Returns
        -------
        tuple or none
            Tuple of Figure and Axes or None if Axes object was passed
        """
        # TODO add option to apply moving average filter before plotting ensemble plot

        import matplotlib.patches as mpatch

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if "figsize" in kwargs:
                figsize = kwargs["figsize"]
            else:
                figsize = plt.rcParams["figure.figsize"]
            fig, ax = plt.subplots(figsize=figsize)

        if plot_params:
            self.hr_ensemble_plot_params.update(plot_params)

        # sns.despine()
        sns.set_palette(self.hr_ensemble_plot_params["colormap"])
        line_styles = self.hr_ensemble_plot_params["line_styles"]
        fontsize = self.hr_ensemble_plot_params["fontsize"]
        xaxis_label = self.hr_ensemble_plot_params["xaxis_label"]
        yaxis_label = self.hr_ensemble_plot_params["yaxis_label"]
        xaxis_minor_ticks = self.hr_ensemble_plot_params["xaxis_minor_ticks"]
        ensemble_alpha = self.hr_ensemble_plot_params["ensemble_alpha"]
        bg_color = self.hr_ensemble_plot_params["background_color"]
        bg_alpha = self.hr_ensemble_plot_params["background_alpha"]
        phase_text = self.hr_ensemble_plot_params["phase_text"]
        end_phase_text = self.hr_ensemble_plot_params["end_phase_text"]
        end_phase_color = self.hr_ensemble_plot_params["end_phase_line_color"]
        end_phase_line_style = self.hr_ensemble_plot_params["end_phase_line_style"]
        end_phase_line_width = self.hr_ensemble_plot_params["end_phase_line_width"]
        legend_loc = self.hr_ensemble_plot_params["legend_loc"]
        legend_bbox_to_anchor = self.hr_ensemble_plot_params["legend_bbox_to_anchor"]

        subphases = np.array(self.subphases)
        mist_dur = [len(v) for v in data.values()]
        start_end = self.get_mist_times(mist_dur)

        for i, key in enumerate(data):
            hr_mist = data[key]
            x = hr_mist.index
            hr_mean = hr_mist.mean(axis=1)
            hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
            ax.plot(
                x,
                hr_mean,
                zorder=2,
                label=phase_text.format(i + 1),
                linestyle=line_styles[i],
            )
            ax.fill_between(
                x,
                hr_mean - hr_stderr,
                hr_mean + hr_stderr,
                zorder=1,
                alpha=ensemble_alpha,
            )
            ax.vlines(
                x=mist_dur[i] - 0.5,
                ymin=0,
                ymax=1,
                transform=ax.get_xaxis_transform(),
                ls=end_phase_line_style,
                lw=end_phase_line_width,
                colors=end_phase_color,
                zorder=3,
            )
            ax.annotate(
                text=end_phase_text.format(i + 1),
                xy=(mist_dur[i], 0.85 - 0.05 * i),
                xytext=(-5, 0),
                xycoords=ax.get_xaxis_transform(),
                textcoords="offset points",
                ha="right",
                fontsize=fontsize - 4,
                bbox=dict(facecolor="#e0e0e0", alpha=0.7, boxstyle="round"),
                zorder=3,
            )

        for (start, end), subphase in zip(start_end, subphases):
            ax.text(
                x=start + 0.5 * (end - start),
                y=0.95,
                transform=ax.get_xaxis_transform(),
                s=subphase,
                ha="center",
                va="center",
                fontsize=fontsize,
            )
        p = mpatch.Rectangle(
            xy=(0, 0.9),
            width=1,
            height=0.1,
            transform=ax.transAxes,
            color="white",
            alpha=0.4,
            zorder=3,
            lw=0,
        )
        ax.add_patch(p)

        for (start, end), color, alpha in zip(start_end, bg_color, bg_alpha):
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)

        ax.set_xlabel(xaxis_label, fontsize=fontsize)
        ax.set_xticks([start for (start, end) in start_end])
        ax.xaxis.set_minor_locator(xaxis_minor_ticks)
        ax.tick_params(axis="x", which="both", bottom=True)

        ax.set_ylabel(yaxis_label, fontsize=fontsize)
        ax.tick_params(axis="y", which="major", left=True)

        if ylims:
            ax.margins(x=0)
            ax.set_ylim(ylims)
        else:
            ax.margins(0, 0.1)

        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            prop={"size": fontsize},
        )

        if fig:
            fig.tight_layout()
            return fig, ax

    # TODO add support for groups in one dataframe (indicated by group column)
    def hr_mean_plot(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        groups: Optional[Sequence[str]] = None,
        group_col: Optional[str] = None,
        plot_params: Optional[Dict] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the course of heart rate during the complete MIST (mean ± standard error per subphase).

        In case of only one group a pandas dataframe can be passed.

        In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe belongs
        to one group, or one dataframe with a column indicating group membership (parameter ``group_col``).

        Regardless of the kind of input the dataframes need to be in the format of a 'mse dataframe', as returned
        by ``MIST.hr_course_mist`` (see ``MIST.hr_course_mist`` for further information).


        Parameters
        ----------
        data : dataframe or dict
            Heart rate data to plot. Can either be one dataframe (in case of only one group or in case of
            multiple groups, together with `group_col`) or a dictionary of dataframes,
            where one dataframe belongs to one group
        groups : list, optional:
             List of group names. If ``None`` is passed, the groups and their order are inferred from the
             dictionary keys or from the unique values in `group_col`. If list is supplied the groups are
             plotted in that order.
             Default: ``None``
        group_col : str, optional
            Name of group column in the dataframe in case of multiple groups and one dataframe
        plot_params : dict, optional
            dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
            For an overview of parameters and their default values, see `mist.hr_course_params`
        ax : plt.Axes, optional
            Axes to plot on, otherwise create a new one. Default: ``None``
        kwargs: dict, optional
            optional parameters to be passed to the plot, such as:
                * figsize: tuple specifying figure dimensions
                * ylims: list to manually specify y-axis limits, float to specify y-axis margin (see ``Axes.margin()``
                for further information), None to automatically infer y-axis limits


        Returns
        -------
        tuple or none
            Tuple of Figure and Axes or None if Axes object was passed
        """

        if plot_params:
            self.hr_mean_plot_params.update(plot_params)
        return plot.hr_mean_plot(
            data=data,
            groups=groups,
            group_col=group_col,
            plot_params=self.hr_mean_plot_params,
            ax=ax,
            **kwargs
        )

    # TODO add methods to remove phases and subphases from MIST dict
