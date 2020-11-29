from typing import Optional, Sequence

from biopsykit.protocols import base


class TSST(base.BaseProtocol):
    """
    Class representing the Trier Social Stress Test (TSST).
    """

    def __init__(
            self, name: Optional[str] = None,
            phases: Optional[Sequence[str]] = None,
            phase_durations: Optional[Sequence[int]] = None
    ):
        if name is None:
            name = "TSST"
        super().__init__(name)

        self.tsst_times: Sequence[int] = [0, 20]

        self.phases: Sequence[str] = ["Prep", "Talk", "Arith"]
        """
        TSST Phases

        Names of TSST phases
        """

        self.phase_durations: Sequence[int] = [5 * 60, 5 * 60, 5 * 60]
        """
        TSST Phase Durations

        Total duration of phases in seconds
        """

        self.saliva_params = {
            'test.text': "TSST",
            'xaxis.label': "Time relative to TSST start [min]"
        }

        self._update_tsst_params(phases, phase_durations)

    def __str__(self) -> str:
        return """{}
        Phases: {}
        Phase Durations: {}
        """.format(self.name, self.phases, self.phase_durations)

    @property
    def mist_times(self):
        return self.test_times

    @mist_times.setter
    def mist_times(self, mist_times):
        self.test_times = mist_times

    def _update_tsst_params(self, phases: Sequence[str], phase_durations: Sequence[int]):
        if phases:
            self.phases = phases
        if phase_durations:
            self.phase_durations = phase_durations

    # def saliva_plot_tsst(
    #         self,
    #         data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    #         groups: Optional[Sequence[str]] = None,
    #         group_col: Optional[str] = None,
    #         plot_params: Optional[Dict] = None,
    #         ylims: Optional[Sequence[float]] = None,
    #         fontsize: Optional[int] = 14,
    #         ax: Optional[plt.Axes] = None,
    #         figsize: Optional[Tuple[float, float]] = None
    # ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    #     import matplotlib.ticker as mticks
    #
    #     fig: Union[plt.Figure, None] = None
    #     if ax is None:
    #         if figsize is None:
    #             figsize = plt.rcParams['figure.figsize']
    #         fig, ax = plt.subplots(figsize=figsize)
    #
    #     # update default parameter if plot parameter were passe
    #     if plot_params:
    #         self.saliva_params.update(plot_params)
    #
    #     # get all plot parameter
    #     sns.set_palette(self.saliva_params['colormap'])
    #     line_styles = self.saliva_params['line_styles']
    #     markers = self.saliva_params['markers']
    #     bg_color = self.saliva_params['background.color']
    #     bg_alpha = self.saliva_params['background.alpha']
    #     mist_color = self.saliva_params['mist.color']
    #     mist_alpha = self.saliva_params['mist.alpha']
    #
    #     saliva_times = np.array(data['cortisol'].index.get_level_values('time').unique())
    #     tsst_times = [0, 20]
    #     tsst_length = saliva_times[-1] - saliva_times[0]
    #     tsst_padding = 0.1 * tsst_length
    #
    #     x_offsets = self.saliva_params['x_offsets']
    #
    #     ax_amy = ax.twinx()
    #
    #     for group, x_off, marker, ls in zip(data['cortisol'].index.get_level_values("condition").unique(), x_offsets,
    #                                         markers,
    #                                         line_styles):
    #         df_cort_grp = data['cortisol'].xs(group, level="condition")
    #         ax.errorbar(x=saliva_times + x_off, y=df_cort_grp["mean"], label="Cortisol",
    #                     yerr=df_cort_grp["se"], capsize=3, marker=marker, ls=ls)
    #         df_amy_grp = data['amylase'].xs(group, level="condition")
    #         ax_amy.errorbar(x=saliva_times + 2, y=df_amy_grp["mean"], label="Amylase",
    #                         yerr=df_amy_grp["se"], capsize=3, marker='P', color='#68CAD0', ls='--')
    #
    #     ax.text(x=tsst_times[0] + 0.5 * (tsst_times[1] - tsst_times[0]), y=15.5, s='TSST',
    #             horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
    #     ax.axvspan(*tsst_times, color=mist_color, alpha=mist_alpha, zorder=1, lw=0)
    #     ax.axvspan(saliva_times[0] - tsst_padding, tsst_times[0], color=bg_color, alpha=bg_alpha, zorder=0, lw=0)
    #     ax.axvspan(tsst_times[1], saliva_times[-1] + tsst_padding, color=bg_color, alpha=bg_alpha, zorder=0, lw=0)
    #
    #     ax.xaxis.set_major_locator(mticks.MultipleLocator(20))
    #     ax.set_xlabel("Time relative to TSST start [min]", fontsize=fontsize)
    #     ax.set_xlim(saliva_times[0] - tsst_padding, saliva_times[-1] + tsst_padding)
    #
    #     ax.set_ylabel("Cortisol [nmol/l]", fontsize=fontsize)
    #     ax_amy.set_ylabel("Amylase [U/l]", fontsize=fontsize)
    #     ax.set_ylim(ylims)
    #     ax.tick_params(labelsize=fontsize)
    #     ax_amy.tick_params(labelsize=fontsize)
    #
    #     # get handles
    #     handles, labels = ax.get_legend_handles_labels()
    #     handles_amy, labels_amy = ax_amy.get_legend_handles_labels()
    #     handles = handles + handles_amy
    #     labels = labels + labels_amy
    #     # remove the errorbars
    #     handles = [h[0] for h in handles]
    #     # use them in the legend
    #     ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), numpoints=1,
    #               prop={"size": fontsize})
    #
    #     if fig:
    #         fig.tight_layout()
    #         return fig, ax
    #
    # def saliva_plot_tsst_il6(
    #         self,
    #         data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    #         groups: Optional[Sequence[str]] = None,
    #         group_col: Optional[str] = None,
    #         plot_params: Optional[Dict] = None,
    #         ylims: Optional[Sequence[float]] = None,
    #         fontsize: Optional[int] = 14,
    #         ax: Optional[plt.Axes] = None,
    #         figsize: Optional[Tuple[float, float]] = None
    # ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    #     import matplotlib.ticker as mticks
    #
    #     fig: Union[plt.Figure, None] = None
    #     if ax is None:
    #         if figsize is None:
    #             figsize = plt.rcParams['figure.figsize']
    #         fig, ax = plt.subplots(figsize=figsize)
    #
    #     # update default parameter if plot parameter were passe
    #     if plot_params:
    #         self.saliva_params.update(plot_params)
    #
    #     # get all plot parameter
    #     sns.set_palette(self.saliva_params['colormap'])
    #     line_styles = self.saliva_params['line_styles']
    #     markers = self.saliva_params['markers']
    #     bg_color = self.saliva_params['background.color']
    #     bg_alpha = self.saliva_params['background.alpha']
    #     mist_color = self.saliva_params['mist.color']
    #     mist_alpha = self.saliva_params['mist.alpha']
    #
    #     saliva_times = np.array(data['il6'].index.get_level_values('time').unique())
    #     tsst_times = [0, 20]
    #     tsst_length = saliva_times[-1] - saliva_times[0]
    #     tsst_padding = 0.1 * tsst_length
    #
    #     x_offsets = self.saliva_params['x_offsets']
    #
    #     for group, x_off, marker, ls in zip(data['il6'].index.get_level_values("condition").unique(), x_offsets,
    #                                         markers,
    #                                         line_styles):
    #         df_cort_grp = data['il6'].xs(group, level="condition")
    #         ax.errorbar(x=saliva_times + x_off, y=df_cort_grp["mean"], label="IL-6",
    #                     yerr=df_cort_grp["se"], capsize=3, marker=marker, ls=ls)
    #
    #     ax.text(x=tsst_times[0] + 0.5 * (tsst_times[1] - tsst_times[0]), y=4.5, s='TSST',
    #             horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
    #     ax.axvspan(*tsst_times, color=mist_color, alpha=mist_alpha, zorder=1, lw=0)
    #     ax.axvspan(saliva_times[0] - tsst_padding, tsst_times[0], color=bg_color, alpha=bg_alpha, zorder=0, lw=0)
    #     ax.axvspan(tsst_times[1], saliva_times[-1] + tsst_padding, color=bg_color, alpha=bg_alpha, zorder=0, lw=0)
    #
    #     ax.xaxis.set_major_locator(mticks.MultipleLocator(20))
    #     ax.set_xlabel("Time relative to TSST start [min]", fontsize=fontsize)
    #     ax.set_xlim(saliva_times[0] - tsst_padding, saliva_times[-1] + tsst_padding)
    #
    #     ax.set_ylabel("IL-6 [pg/ml]", fontsize=fontsize)
    #     ax.set_ylim(ylims)
    #     ax.tick_params(labelsize=fontsize)
    #
    #     # get handles
    #     handles, labels = ax.get_legend_handles_labels()
    #     # remove the errorbars
    #     handles = [h[0] for h in handles]
    #     # use them in the legend
    #     ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), numpoints=1,
    #               prop={"size": fontsize})
    #
    #     if fig:
    #         fig.tight_layout()
    #         return fig, ax
