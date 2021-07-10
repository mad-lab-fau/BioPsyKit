"""Module representing the Stroop Test protocol."""
# from typing import Dict, Tuple, Union, Optional, Sequence
from typing import Optional, Sequence

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticks
# import seaborn as sns
#
# import biopsykit.colors as colors
# import biopsykit.protocols.plotting as plot
from biopsykit.protocols import BaseProtocol


class Stroop(BaseProtocol):
    """Class representing the Stroop Test and data collected while conducting the Stroop test.

    # TODO add further documentation

    """

    def __init__(self, name: Optional[str] = None, structure: Optional[Sequence[str]] = None, **kwargs):

        if name is None:
            name = "Stroop"

        if structure is None:
            structure = {
                "Part1": None,
                "Stroop": {
                    "Stroop1": 180,
                    "Stroop2": 180,
                    "Stroop3": 180,
                },
                "Part2": None,
            }

        test_times = kwargs.pop("test_times", [0, 10])

        hr_mean_plot_params = {"xlabel": "Stroop Phases"}
        hr_mean_plot_params.update(kwargs.pop("hr_mean_plot_params", {}))

        saliva_plot_params = {"test_title": "Stroop", "xlabel": "Time relative to Stroop start [min]"}
        saliva_plot_params.update(kwargs.pop("saliva_plot_params", {}))

        kwargs.update({"hr_mean_plot_params": hr_mean_plot_params, "saliva_plot_params": saliva_plot_params})
        super().__init__(name=name, structure=structure, test_times=test_times, **kwargs)
        #
        # self.hr_ensemble_plot_params = {
        #     "colormap": colors.fau_palette_blue("ensemble_3"),
        #     "line_styles": ["-", "--", "-."],
        #     "ensemble_alpha": 0.4,
        #     "background_color": ["#e0e0e0", "#9e9e9e", "#757575"],
        #     "background_alpha": [0.5, 0.5, 0.5],
        #     "fontsize": 14,
        #     "xaxis_label": r"Time [s] ",
        #     "xaxis_minor_ticks": mticks.MultipleLocator(60),
        #     "yaxis_label": r"$\Delta$Mean HR [bpm]",
        #     "legend_loc": "upper right",
        #     "legend_bbox_to_anchor": (0.25, 0.90),
        #     "phase_text": "Stroop Phase {}",
        #     "end_phase_text": "End Phase {}",
        #     "end_phase_line_color": "#e0e0e0",
        #     "end_phase_line_style": "dashed",
        #     "end_phase_line_width": 2.0,
        # }
        # self.stroop_plot_params = {
        #     "colormap": colors.fau_palette_blue("ensemble_3"),
        #     "line_styles": ["-", "--", "-."],
        #     "background_color": ["#e0e0e0", "#9e9e9e", "#757575"],
        #     "background_alpha": [0.5, 0.5, 0.5],
        #     "fontsize": 14,
        #     "xaxis_label": r"Stroop phases",
        #     "xaxis_minor_ticks": mticks.MultipleLocator(60),
        #     "yaxis_label": r"$\Delta$Mean HR [bpm]",
        #     "legend_loc": "upper right",
        #     "legend_bbox_to_anchor": (1.00, 0.90),
        #     "phase_text": "Stroop Phase {}",
        # }
        #
        # self.hr_mean_plot_params = {
        #     "colormap": colors.fau_palette_blue("line_2"),
        #     "line_styles": ["-", "--"],
        #     "markers": ["o", "P"],
        #     "background_color": ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
        #     "background_alpha": [0.5, 0.5, 0.5],
        #     "x_offsets": [0, 0.05],
        #     "fontsize": 14,
        #     "xaxis_label": "Stroop Subphases",
        #     "yaxis_label": r"$\Delta$HR [%]",
        #     "mist_phase_text": "MIST Phase {}",
        # }

    # def hr_ensemble_plot(
    #     self,
    #     data: Dict[str, pd.DataFrame],
    #     plot_params: Optional[Dict] = None,
    #     ylims: Optional[Sequence[float]] = None,
    #     ax: Optional[plt.Axes] = None,
    #     is_group_dict: Optional[bool] = False,
    #     **kwargs,
    # ) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    #     """
    #     Plots the course of heart rate during each Stroop subphase continuously as ensemble plot
    #     (mean ± standard error).
    #     Simply pass a 'Stroop dict' dictionary with one pandas heart rate dataframe per Stroop subphase
    #     (see ``Stroop.concat_stroop_dicts`` for further explanation), i.e. heart rate data with one column
    #     per subject.
    #
    #     Parameters
    #     ----------
    #     data : dict
    #         dict with heart rate data to plot
    #     plot_params : dict, optional
    #         dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
    #         For an overview of parameters and their default values, see `mist.hr_ensemble_params`
    #     ylims : list, optional
    #         y axis limits or ``None`` to infer y axis limits from data. Default: ``None``
    #     ax : plt.Axes, optional
    #         Axes to plot on, otherwise create a new one. Default: ``None``
    #
    #     Returns
    #     -------
    #     tuple or none
    #         Tuple of Figure and Axes or None if Axes object was passed
    #     """
    #
    #     import matplotlib.patches as mpatch
    #
    #     fig: Union[plt.Figure, None] = None
    #     if ax is None:
    #         if "figsize" in kwargs:
    #             figsize = kwargs["figsize"]
    #         else:
    #             figsize = plt.rcParams["figure.figsize"]
    #         fig, ax = plt.subplots(figsize=figsize)
    #
    #     if plot_params:
    #         self.hr_ensemble_plot_params.update(plot_params)
    #
    #     # sns.despine()
    #     sns.set_palette(self.hr_ensemble_plot_params["colormap"])
    #     line_styles = self.hr_ensemble_plot_params["line_styles"]
    #     fontsize = self.hr_ensemble_plot_params["fontsize"]
    #     xaxis_label = self.hr_ensemble_plot_params["xaxis_label"]
    #     yaxis_label = self.hr_ensemble_plot_params["yaxis_label"]
    #     xaxis_minor_ticks = self.hr_ensemble_plot_params["xaxis_minor_ticks"]
    #     ensemble_alpha = self.hr_ensemble_plot_params["ensemble_alpha"]
    #     bg_color = self.hr_ensemble_plot_params["background_color"]
    #     bg_alpha = self.hr_ensemble_plot_params["background_alpha"]
    #     phase_text = self.hr_ensemble_plot_params["phase_text"]
    #     end_phase_text = self.hr_ensemble_plot_params["end_phase_text"]
    #     end_phase_color = self.hr_ensemble_plot_params["end_phase_line_color"]
    #     end_phase_line_style = self.hr_ensemble_plot_params["end_phase_line_style"]
    #     end_phase_line_width = self.hr_ensemble_plot_params["end_phase_line_width"]
    #     legend_loc = self.hr_ensemble_plot_params["legend_loc"]
    #     legend_bbox_to_anchor = self.hr_ensemble_plot_params["legend_bbox_to_anchor"]
    #
    #     subphases = np.array(self.subphases)
    #     # mist_dur = [len(v) for v in data.values()]
    #     start_end = [
    #         (0, self.subphase_durations[0]),
    #         (
    #             self.subphase_durations[0],
    #             self.subphase_durations[0] + self.subphase_durations[1],
    #         ),
    #     ]
    #
    #     if is_group_dict:
    #         for j, condition in enumerate(data):
    #             mist_dur = [len(v) for v in data[condition].values()]
    #             for i, key in enumerate(data[condition]):
    #                 pal = sns.color_palette()[j]
    #
    #                 hr_mist = data[condition][key]
    #                 x = hr_mist.index
    #                 hr_mean = hr_mist.mean(axis=1)
    #                 hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
    #                 ax.plot(
    #                     x,
    #                     hr_mean,
    #                     zorder=2,
    #                     label=phase_text.format(i + 1) + " - " + condition,
    #                     linestyle=line_styles[i],
    #                     color=pal,
    #                 )
    #                 ax.fill_between(
    #                     x,
    #                     hr_mean - hr_stderr,
    #                     hr_mean + hr_stderr,
    #                     zorder=1,
    #                     alpha=ensemble_alpha,
    #                 )
    #                 ax.vlines(
    #                     x=mist_dur[i] - 0.5,
    #                     ymin=0,
    #                     ymax=1,
    #                     transform=ax.get_xaxis_transform(),
    #                     ls=end_phase_line_style,
    #                     lw=end_phase_line_width,
    #                     colors=end_phase_color,
    #                     zorder=3,
    #                 )
    #                 ax.annotate(
    #                     text=end_phase_text.format(i + 1),
    #                     xy=(mist_dur[i], 0.85 - 0.05 * i),
    #                     xytext=(-5, 0),
    #                     xycoords=ax.get_xaxis_transform(),
    #                     textcoords="offset points",
    #                     ha="right",
    #                     fontsize=fontsize - 4,
    #                     bbox=dict(facecolor="#e0e0e0", alpha=0.7, boxstyle="round"),
    #                     zorder=3,
    #                 )
    #             ax.legend(loc=legend_loc, bbox_to_anchor=(0.20, 0.3), prop={"size": fontsize})
    #     else:
    #         mist_dur = [len(v) for v in data.values()]
    #         for i, key in enumerate(data):
    #             hr_mist = data[key]
    #             x = hr_mist.index
    #             hr_mean = hr_mist.mean(axis=1)
    #             hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
    #             ax.plot(
    #                 x,
    #                 hr_mean,
    #                 zorder=2,
    #                 label=phase_text.format(i + 1),
    #                 linestyle=line_styles[i],
    #             )
    #             ax.fill_between(
    #                 x,
    #                 hr_mean - hr_stderr,
    #                 hr_mean + hr_stderr,
    #                 zorder=1,
    #                 alpha=ensemble_alpha,
    #             )
    #             ax.vlines(
    #                 x=mist_dur[i] - 0.5,
    #                 ymin=0,
    #                 ymax=1,
    #                 transform=ax.get_xaxis_transform(),
    #                 ls=end_phase_line_style,
    #                 lw=end_phase_line_width,
    #                 colors=end_phase_color,
    #                 zorder=3,
    #             )
    #             ax.annotate(
    #                 text=end_phase_text.format(i + 1),
    #                 xy=(mist_dur[i], 0.85 - 0.05 * i),
    #                 xytext=(-5, 0),
    #                 xycoords=ax.get_xaxis_transform(),
    #                 textcoords="offset points",
    #                 ha="right",
    #                 fontsize=fontsize - 4,
    #                 bbox=dict(facecolor="#e0e0e0", alpha=0.7, boxstyle="round"),
    #                 zorder=3,
    #             )
    #         ax.legend(
    #             loc=legend_loc,
    #             bbox_to_anchor=legend_bbox_to_anchor,
    #             prop={"size": fontsize},
    #         )
    #
    #     for (start, end), subphase in zip(start_end, subphases):
    #         ax.text(
    #             x=start + 0.5 * (end - start),
    #             y=0.95,
    #             transform=ax.get_xaxis_transform(),
    #             s=subphase,
    #             ha="center",
    #             va="center",
    #             fontsize=fontsize,
    #         )
    #     p = mpatch.Rectangle(
    #         xy=(0, 0.9),
    #         width=1,
    #         height=0.1,
    #         transform=ax.transAxes,
    #         color="white",
    #         alpha=0.4,
    #         zorder=3,
    #         lw=0,
    #     )
    #     ax.add_patch(p)
    #
    #     for (start, end), color, alpha in zip(start_end, bg_color, bg_alpha):
    #         ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)
    #
    #     ax.set_xlabel(xaxis_label, fontsize=fontsize)
    #     ax.set_xticks([start for (start, end) in start_end])
    #     ax.xaxis.set_minor_locator(xaxis_minor_ticks)
    #     ax.tick_params(axis="x", which="both", bottom=True)
    #
    #     ax.set_ylabel(yaxis_label, fontsize=fontsize)
    #     ax.tick_params(axis="y", which="major", left=True)
    #
    #     if ylims:
    #         ax.margins(x=0)
    #         ax.set_ylim(ylims)
    #     else:
    #         ax.margins(0, 0.1)
    #
    #     if fig:
    #         fig.tight_layout()
    #         return fig, ax
    #
    # def hr_mean_subphases(
    #     self,
    #     data: Union[
    #         Dict[str, Dict[str, pd.DataFrame]],
    #         Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    #     ],
    #     is_group_dict: Optional[bool] = False,
    # ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    #     """
    #     Computes the heart rate mean and standard error per Stroop phase over all subjects.
    #     See ``bp.protocols.utils.hr_course`` for further information.
    #
    #     Parameters
    #     ----------
    #     data : dict
    #         nested dictionary containing heart rate data.
    #     is_group_dict : boolean, optional
    #         ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
    #         Default: ``False``
    #
    #     Returns
    #     -------
    #     dict or pd.DataFrame
    #         'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.
    #     """
    #
    #     return super().mean_se_subphases(data, subphases=self.subphases, is_group_dict=is_group_dict)
    #
    # def stroop_dict_to_dataframe(
    #     self,
    #     dict_stroop=Dict[str, Dict],
    #     columns: Optional[Sequence[str]] = None,
    #     is_group_dict: Optional[bool] = False,
    # ) -> pd.DataFrame:
    #     """
    #     Converts the dictionary into one dataframe with a MultiIndex (subject, phase). The structure needs to
    #     be the same as derived from load_stroop_inquisit_data.
    #
    #     The dictionary can also be a group dictionary. In this case, the MultiIndex is expanded with 'group'.
    #
    #     Parameters
    #     ----------
    #     dict_stroop : dict
    #         dictionary which should be converted into a dataframe. The structure should be as followed:
    #         {'subject': {'subphase' : data,...},..} or as group_dict
    #         {'group':{'subject': {'subphase' : data,...},..},..}
    #     columns : str
    #         column names which should be used.
    #         Default: ``None`` -> all existing columns are used.
    #     is_group_dict : bool
    #         ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
    #         Default: ``False``
    #     Returns
    #     -------
    #     dataframe :
    #         dataframe with the stroop test data ordered by (group), subject and subphase.
    #     """
    #     df_stroop = pd.DataFrame()
    #
    #     if is_group_dict:
    #         for group, dict_data in dict_stroop.items():
    #             for subject, data in dict_data.items():
    #                 for subphase, df in data.items():
    #                     df_stroop = pd.concat([df_stroop, df.set_index([[group], [subject], [subphase]])])
    #         df_stroop.index.names = ["group", "subject", "subphase"]
    #     else:
    #         for subject, data in dict_stroop.items():
    #             for subphase, df in data.items():
    #                 df_stroop = pd.concat([df_stroop, df.set_index([[subject], [subphase]])])
    #         df_stroop.index.names = ["subject", "subphase"]
    #
    #     if columns:
    #         df_stroop = df_stroop[columns]
    #
    #     return df_stroop
    #
    # def stroop_mean_se(self, data=pd.DataFrame, is_group_dict: Optional[bool] = False) -> pd.DataFrame:
    #     """
    #     Computes the mean and standard error of the stroop test data per Stroop subphase over all subjects.
    #
    #     Parameters
    #     ----------
    #     data : pd.Dataframe
    #         dataframe with data from the stroop test of which mean and standard error should be computed.
    #         It has to be one dataframe which is in the kind of format as returned by `stroop_dict_to_dataframe`
    #     is_group_dict : bool
    #         ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
    #         Default: ``False``
    #
    #     Returns
    #     -------
    #     dataframe:
    #         dataframe with mean and standard deviation values.
    #     """
    #     if is_group_dict:
    #         index = [("group", "subphase")]
    #     else:
    #         index = ["subphase"]
    #
    #     mean = data.mean(level=index).add_suffix("_mean")
    #     std = data.std(level=index).add_suffix("_std")
    #     df_mean_se = mean.join(std)
    #
    #     # scale correct answers to percent
    #     if ("correct_mean" and "correct_std") in df_mean_se.columns:
    #         df_mean_se[["correct_mean", "correct_std"]] = df_mean_se[["correct_mean", "correct_std"]] * 100
    #
    #     return df_mean_se
    #
    # def stroop_plot(
    #     self,
    #     data=pd.DataFrame,
    #     variable: Optional[str] = "meanRT",
    #     is_group_dict: Optional[bool] = False,
    #     group_col: Optional[str] = "condition",
    #     ylims: Optional[Sequence[float]] = None,
    #     ax: Optional[plt.Axes] = None,
    #     **kwargs,
    # ) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    #     """
    #     Plots the mean response time or correct answers during the different Stroop task
    #     (mean ± standard error per phase).
    #
    #     In case of only one group a pandas dataframe can be passed.
    #
    #     In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe
    #     belongs to one group, or one dataframe with a column indicating group membership (parameter ``group_col``).
    #
    #     Regardless of the kind of input the dataframes need to be in the format of a 'mean dataframe', as returned
    #     by ``stroop_mean`` (see ``Stroop.stroop_mean`` for further information).
    #
    #
    #     Parameters
    #     ----------
    #     data : dataframe or dict
    #         Mean response/Correct answers data to plot. It has to be one dataframe which is in the kind of format as
    #         returned by `stroop_mean_se`
    #     variable : str
    #          Determines if the mean response times (``meanRT``) or correct answers (``propcorrect``) of the stroop
    #          test should be plotted.
    #          Default: ``meanRT``
    #     is_group_dict : bool, optional:
    #          List of group names. If ``None`` is passed, the groups and their order are inferred from the
    #          dictionary keys or from the unique values in `group_col`. If list is supplied the groups are
    #          plotted in that order.
    #          Default: ``None``
    #     group_col : str, optional
    #         Name of group column in the dataframe in case of multiple groups and one dataframe
    #     ylims : Tuple(int,int)
    #         Integer to scale the y axes.
    #         Default: ``None``
    #     ax : plt.Axes, optional
    #         Axes to plot on, otherwise create a new one. Default: ``None``
    #     kwargs: dict, optional
    #         optional parameters to be passed to the plot, such as:
    #             * figsize: tuple specifying figure dimensions
    #             * ylims: list to manually specify y-axis limits, float to specify y-axis margin (see ``Axes.margin()``
    #             for further information), None to automatically infer y-axis limits
    #     """
    #
    #     fig: Union[plt.Figure, None] = None
    #     if ax is None:
    #         if "figsize" in kwargs:
    #             figsize = kwargs["figsize"]
    #         else:
    #             figsize = plt.rcParams["figure.figsize"]
    #         fig, ax = plt.subplots(figsize=figsize)
    #
    #     sns.set_palette(self.stroop_plot_params["colormap"])
    #     line_styles = self.stroop_plot_params["line_styles"]
    #     fontsize = self.stroop_plot_params["fontsize"]
    #     xaxis_label = self.stroop_plot_params["xaxis_label"]
    #     xaxis_minor_ticks = self.stroop_plot_params["xaxis_minor_ticks"]
    #     bg_color = self.stroop_plot_params["background_color"]
    #     bg_alpha = self.stroop_plot_params["background_alpha"]
    #     x_labels = self.phases
    #
    #     x = np.arange(len(x_labels))
    #     start_end = [(i - 0.5, i + 0.5) for i in x]
    #     if is_group_dict:
    #         conditions = list(set(data.index.get_level_values(group_col)))
    #         line1 = ax.errorbar(
    #             x,
    #             data.xs(conditions[0], level=group_col)[variable + "_mean"],
    #             yerr=data.xs(conditions[0], level=group_col)[variable + "_std"],
    #             color=sns.color_palette()[0],
    #             label=conditions[0],
    #             lw=2,
    #             errorevery=1,
    #             ls=line_styles[0],
    #             marker="D",
    #             capsize=3,
    #         )
    #         line2 = ax.errorbar(
    #             x,
    #             data.xs(conditions[1], level=group_col)[variable + "_mean"],
    #             yerr=data.xs(conditions[1], level=group_col)[variable + "_std"],
    #             color=sns.color_palette()[1],
    #             label=conditions[1],
    #             lw=2,
    #             errorevery=1,
    #             ls=line_styles[1],
    #             marker="D",
    #             capsize=3,
    #         )
    #         plt.legend(handles=[line1, line2], loc="upper right", prop={"size": fontsize})
    #     else:
    #         ax.errorbar(
    #             x,
    #             data[variable + "_mean"],
    #             yerr=data[variable + "_std"],
    #             color=sns.color_palette()[0],
    #             lw=2,
    #             errorevery=1,
    #             ls=line_styles[0],
    #             marker="D",
    #             capsize=3,
    #         )
    #
    #     for (start, end), color, alpha in zip(start_end, bg_color, bg_alpha):
    #         ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)
    #
    #     ax.set_xticklabels(x_labels, fontsize=fontsize)
    #     ax.set_xlabel(xaxis_label, fontsize=fontsize)
    #     ax.set_xticks([start + 0.5 for (start, end) in start_end])
    #     ax.xaxis.set_minor_locator(xaxis_minor_ticks)
    #     ax.tick_params(axis="x", which="both", bottom=True)
    #
    #     if variable == "correct":
    #         ax.set_ylim(0, 105)
    #         ax.set_ylabel(r"$\Delta$Correct answers [%]", fontsize=fontsize)
    #     elif variable == "latency":
    #         ax.set_ylabel(r"$\Delta$Response time [ms]", fontsize=fontsize)
    #
    #     ax.tick_params(axis="y", which="major", left=True, labelsize=fontsize)
    #
    #     if ylims:
    #         ax.margins(x=0)
    #         ax.set_ylim(ylims)
    #     else:
    #         ax.margins(0, 0.1)
    #
    #     if fig:
    #         fig.tight_layout()
    #         return fig, ax
    #
    # def concat_phase_dict(
    #     self, dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]], **kwargs
    # ) -> Dict[str, pd.DataFrame]:
    #     """
    #     Rearranges the 'HR subject dict' (see `util s.load_hr_excel_all_subjects`) into 'Stroop subphase dict'.
    #     See ``bp.protocols.utils.concat_phase_dict`` for further information.
    #
    #     Parameters
    #     ----------
    #     dict_hr_subject : dict
    #         'HR subject dict', i.e. a nested dict with heart rate data per Stroop subphase and subject
    #     **kwargs
    #
    #     Returns
    #     -------
    #     dict
    #         'Stroop dict', i.e. a dict with heart rate data of all subjects per Stroop subphase
    #
    #     """
    #     if "phases" in kwargs:
    #         return super().concat_phase_dict(dict_hr_subject, kwargs["phases"])
    #     else:
    #         return super().concat_phase_dict(dict_hr_subject, self.phases)
    #
    # def split_groups_stroop(
    #     self,
    #     dict_stroop=Dict[str, Dict[str, pd.DataFrame]],
    #     condition_dict=Dict[str, Sequence[str]],
    # ) -> Dict[str, Dict[str, pd.DataFrame]]:
    #     """
    #     Splits 'Stroop dict' into group dict, i.e. one 'Stroop dict' per group.
    #
    #     Parameters
    #     ----------
    #     phase_dict : dict
    #         'Dict stroop' to be split in groups. This is the outcome of 'stroop.load_stroop_test_data()'
    #     condition_dict : dict
    #         dictionary of group membership. Keys are the different groups, values are lists of subject IDs that
    #         belong to the respective group
    #
    #     Returns
    #     -------
    #     dict
    #         group dict with one 'Stroop dict' per group
    #
    #     """
    #     return {condition: {ID: dict_stroop[ID] for ID in IDs} for condition, IDs in condition_dict.items()}
    #
    # def split_groups(
    #     cls,
    #     phase_dict: Dict[str, pd.DataFrame],
    #     condition_list: Dict[str, Sequence[str]],
    # ) -> Dict[str, Dict[str, pd.DataFrame]]:
    #     """
    #     Splits 'Stroop Phase dict' into group dict, i.e. one 'Stroop Phase dict' per group.
    #
    #     Parameters
    #     ----------
    #     phase_dict : dict
    #         'Stroop Phase dict' to be split in groups. See ``bp.protocols.utils.concat_phase_dict``
    #         for further information
    #     condition_list : dict
    #         dictionary of group membership. Keys are the different groups, values are lists of subject IDs that
    #         belong to the respective group
    #
    #     Returns
    #     -------
    #     dict
    #         nested group dict with one 'Stroop Phase dict' per group
    #     """
    #
    #     return super().split_groups(phase_dict, condition_list)
    #
    # def hr_mean_plot(
    #     self,
    #     data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    #     groups: Optional[Sequence[str]] = None,
    #     group_col: Optional[str] = None,
    #     plot_params: Optional[Dict] = None,
    #     ax: Optional[plt.Axes] = None,
    #     **kwargs,
    # ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    #     """
    #     Plots the course of heart rate during the complete Stroop test (mean ± standard error per phase).
    #
    #     In case of only one group a pandas dataframe can be passed.
    #
    #     In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe
    #     belongs to one group, or one dataframe with a column indicating group membership (parameter ``group_col``).
    #
    #     Regardless of the kind of input the dataframes need to be in the format of a 'mse dataframe', as returned
    #     by ``stroop.hr_course_mist`` (see ``MIST.hr_course_mist`` for further information).
    #
    #
    #     Parameters
    #     ----------
    #     data : dataframe or dict
    #         Heart rate data to plot. Can either be one dataframe (in case of only one group or in case of
    #         multiple groups, together with `group_col`) or a dictionary of dataframes,
    #         where one dataframe belongs to one group
    #     groups : list, optional:
    #          List of group names. If ``None`` is passed, the groups and their order are inferred from the
    #          dictionary keys or from the unique values in `group_col`. If list is supplied the groups are
    #          plotted in that order.
    #          Default: ``None``
    #     group_col : str, optional
    #         Name of group column in the dataframe in case of multiple groups and one dataframe
    #     plot_params : dict, optional
    #         dict with adjustable parameters specific for this plot or ``None`` to keep default parameter values.
    #         For an overview of parameters and their default values, see `mist.hr_course_params`
    #     ax : plt.Axes, optional
    #         Axes to plot on, otherwise create a new one. Default: ``None``
    #     kwargs: dict, optional
    #         optional parameters to be passed to the plot, such as:
    #             * figsize: tuple specifying figure dimensions
    #             * ylims: list to manually specify y-axis limits, float to specify y-axis margin (see ``Axes.margin()``
    #             for further information), None to automatically infer y-axis limits
    #
    #
    #     Returns
    #     -------
    #     tuple or none
    #         Tuple of Figure and Axes or None if Axes object was passed
    #     """
    #
    #     if plot_params:
    #         self.hr_mean_plot_params.update(plot_params)
    #     return plot.hr_mean_plot(
    #         data=data, groups=groups, group_col=group_col, plot_params=self.hr_mean_plot_params, ax=ax, **kwargs
    #     )
    #
    # def hr_mean_se(
    #     self,
    #     data: Union[
    #         Dict[str, Dict[str, pd.DataFrame]],
    #         Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    #     ],
    #     is_group_dict: Optional[bool] = False,
    # ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    #     """
    #     Computes the heart rate mean and standard error per phase over all subjects.
    #     See ``bp.protocols.utils.hr_course`` for further information.
    #
    #     Parameters
    #     ----------
    #     data : dict
    #         nested dictionary containing heart rate data.
    #     is_group_dict : boolean, optional
    #         ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
    #         Default: ``False``
    #
    #     Returns
    #     -------
    #     dict or pd.DataFrame
    #         'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.
    #     """
    #
    #     return super().mean_se_subphases(data, is_group_dict=is_group_dict)
