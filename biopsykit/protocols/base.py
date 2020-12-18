from typing import Dict, Sequence, Union, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import biopsykit.colors as colors
import biopsykit.signals.utils as su


class BaseProtocol:

    def __init__(self, name: str):
        self.name: str = name
        """
        Study name
        """

        self._saliva_params = {}

        self.saliva_params: Dict = {
            'colormap': colors.cmap_fau_blue('2_lp'),
            'line_styles': ['-', '--'],
            'markers': ['o', 'P'],
            'background.color': "#e0e0e0",
            'background.alpha': 0.5,
            'test.color': "#9e9e9e",
            'test.alpha': 0.5,
            'x_offsets': [0, 0.5],
            'fontsize': 14,
            'multi.x_offset': 1,
            'multi.fontsize': 10,
            'multi.legend_offset': 0.3,
            'multi.colormap': colors.cmap_fau_phil('2_lp'),
            'xaxis.tick_locator': plt.MultipleLocator(20),
            'yaxis.label': {
                'cortisol': "Cortisol [nmol/l]",
                'amylase': "Amylase [U/l]",
                'il6': "IL-6 [pg/ml]",
            }
        }

        self.test_times: Sequence[int] = []

    def __repr__(self):
        return self.__str__()

    @property
    def saliva_params(self) -> Dict:
        return self._saliva_params

    @saliva_params.setter
    def saliva_params(self, saliva_params: Dict):
        self._saliva_params.update(saliva_params)

    def concat_phase_dict(self, dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
                          phases: Sequence[str]) -> Dict[str, pd.DataFrame]:
        """
        Rearranges the 'HR subject dict' (see `utils.load_hr_excel_all_subjects`) into 'Phase dict', i.e. a dictionary with
        one dataframe per Stress Test phase where each dataframe contains column-wise HR data for all subjects.

        The **output** format will be the following:

        { <"Stress_Phase"> : hr_dataframe, 1 subject per column }

        Parameters
        ----------
        dict_hr_subject : dict
            'HR subject dict', i.e. a nested dict with heart rate data per Stress Test phase and subject
        phases : list
            list of Stress Test phases. E.g. for MIST this would be the three MIST phases ['MIST1', 'MIST2', 'MIST3'],
            for TSST this would be ['Preparation', 'Speaking', 'ArithmeticTask']

        Returns
        -------
        dict
            'Phase dict', i.e. a dict with heart rate data of all subjects per Stress Test phase

        """

        return su.concat_phase_dict(dict_hr_subject, phases)

    def split_subphases(
            self,
            data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
            subphase_names: Sequence[str],
            subphase_times: Sequence[Tuple[int, int]],
            is_group_dict: Optional[bool] = False
    ) -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
        """
        Splits a `Phase dict` (or a dict of such, in case of multiple groups, see ``bp.signals.utils.concat_dicts``)
        into a `Subphase dict` (see below for further explanation).

        The **input** is a `Phase dict`, i.e. a dictionary with heart rate data per Stress Test phase
        in the following format:

        { <"Phase"> : HR_dataframe, 1 subject per column }

        If multiple groups are present, then the expected input is nested, i.e. a dict of 'Phase dicts',
        with one entry per group.

        The **output** is a `Subphase dict`, i.e. a nested dictionary with heart rate data per Subphase in the
        following format:

        { <"Phase"> : { <"Subphase"> : HR_dataframe, 1 subject per column } }

        If multiple groups are present, then the output is nested, i.e. a dict of 'Subphase dicts',
        with one entry per group.


        Parameters
        ----------
        data : dict
            'Phase dict' or nested dict of 'Phase dicts' if `is_group_dict` is ``True``
        subphase_names : list
            List with names of subphases
        subphase_times : list
            List with start and end times of each subphase in seconds
        is_group_dict : bool, optional
            ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``

        Returns
        -------
        dict
            'Subphase dict' with course of HR data per Stress Test phase, subphase and subject, respectively or
            nested dict of 'Subphase dicts' if `is_group_dict` is ``True``

        """
        return su.split_subphases(data=data, subphase_names=subphase_names, subphase_times=subphase_times,
                                  is_group_dict=is_group_dict)

    @classmethod
    def split_groups(cls, phase_dict: Dict[str, pd.DataFrame],
                     condition_dict: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Splits 'Phase dict' into group dict, i.e. one 'Phase dict' per group.

        Parameters
        ----------
        phase_dict : dict
            'Phase dict' to be split in groups. See ``utils.concat_phase_dict`` for further information
        condition_dict : dict
            dictionary of group membership. Keys are the different groups, values are lists of subject IDs that belong
            to the respective group

        Returns
        -------
        dict
            nested group dict with one 'Phase dict' per group

        """
        return su.split_groups(phase_dict=phase_dict, condition_dict=condition_dict)

    def _mean_se_subphases(
            self,
            data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
            subphases: Optional[Sequence[str]] = None,
            is_group_dict: Optional[bool] = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Computes the heart rate mean and standard error per subphase over all subjects.

        As input either 1. a 'Subphase dict' (for only one group) or 2. a dict of 'Subphase dict', one dict per
        group (for multiple groups, if `is_group_dict` is ``True``) can be passed
        (see ``utils.split_subphases`` for more explanation). Both dictionaries are outputs from
        `utils.split_subphases``.

        The output is a 'mse dataframe' (or a dict of such, in case of multiple groups), a pandas dataframe with:
            * columns: ['mean', 'se'] for mean and standard error
            * rows: MultiIndex with level 0 = Phases and level 1 = Subphases.

        The dict structure should like the following:
            (a) { "<Phase>" : { "<Subphase>" : heart rate dataframe, 1 subject per column } }
            (b) { "<Group>" : { <"Phase"> : { "<Subphase>" : heart rate dataframe, 1 subject per column } } }

        Parameters
        ----------
        data : dict
            nested dictionary containing heart rate data.
        subphases : list, optional
            list of subphase names or ``None`` to use default subphase names. Default: ``None``
        is_group_dict : boolean, optional
            ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
            Default: ``False``

        Returns
        -------
        dict or pd.DataFrame
            'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.
        """

        return su.mean_se_nested_dict(data, subphases=subphases, is_group_dict=is_group_dict)

    def saliva_plot(
            self,
            data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
            feature_name: Optional[str] = 'cortisol',
            saliva_times: Optional[Sequence[int]] = None,
            groups: Optional[Sequence[str]] = None,
            group_col: Optional[str] = None,
            plot_params: Optional[Dict] = None,
            ylims: Optional[Sequence[float]] = None,
            ax: Optional[plt.Axes] = None,
            figsize: Optional[Tuple[float, float]] = None
    ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
        """
        TODO: add documentation

        Parameters
        ----------
        data
        feature_name
        saliva_times
        groups
        group_col
        plot_params
        ylims
        ax
        figsize

        Returns
        -------

        """

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        # update default parameter if plot parameter were passe
        if plot_params:
            self.saliva_params.update(plot_params)

        bg_color = self.saliva_params['background.color']
        bg_alpha = self.saliva_params['background.alpha']
        test_text = self.saliva_params['test.text']
        test_color = self.saliva_params['test.color']
        test_alpha = self.saliva_params['test.alpha']
        fontsize = self.saliva_params['fontsize']
        xaxis_label = self.saliva_params['xaxis.label']
        xaxis_tick_locator = self.saliva_params['xaxis.tick_locator']

        ylim_padding = [0.9, 1.2]

        if isinstance(data, dict) and feature_name in data.keys():
            # multiple biomarkers were passed => get the selected biomarker and try to get the groups from the index
            data = data[feature_name]

        if saliva_times is None:
            if isinstance(data, pd.DataFrame):
                # DataFrame was passed
                if 'time' in data.index.names:
                    saliva_times = np.array(data.index.get_level_values('time').unique())
            else:
                # Dict was passed => multiple groups (where each entry is a dataframe per group) or multiple biomarker
                # (where each entry is one biomarker)
                if all(['time' in d.index.names for d in data.values()]):
                    saliva_times = np.array([d.index.get_level_values('time').unique() for d in data.values()],
                                            dtype=object)
                    if not all([len(saliva_time) == len(saliva_times[0]) for saliva_time in saliva_times]):
                        raise ValueError(
                            "Different saliva time lengths passed! Did you pass multiple biomarkers? "
                            "For plotting multiple biomarkers, call the `saliva_plot` function on the same axis "
                            "repeatedly for the different biomarkers!")
                    if (saliva_times == saliva_times[0]).all():
                        saliva_times = saliva_times[0]
                    else:
                        raise ValueError("Saliva times inconsistent for the different groups!")
                else:
                    raise ValueError("Not all dataframes contain a 'time' column for saliva times!")

        if not groups:
            # extract groups from data if they were not supplied
            if isinstance(data, pd.DataFrame):
                # get group names from index
                if "condition" in data.index.names:
                    groups = list(data.index.get_level_values("condition").unique())
                elif group_col:
                    if group_col in data:
                        groups = list(data[group_col].unique())
                    else:
                        raise ValueError(
                            "`{}`, specified as `group_col` not in columns of the dataframe!".format(group_col))
                else:
                    groups = ["Data"]
            else:
                # get group names from dict
                groups = list(data.keys())

        if not ylims:
            if isinstance(data, pd.DataFrame):
                ylims = [ylim_padding[0] * (data['mean'] - data['se']).min(),
                         ylim_padding[1] * (data['mean'] + data['se']).max()]
            else:
                ylims = [ylim_padding[0] * min([(d['mean'] - d['se']).min() for d in data.values()])]

        if saliva_times is None:
            raise ValueError("Must specify saliva times!")

        total_length = saliva_times[-1] - saliva_times[0]
        x_padding = 0.1 * total_length

        if len(ax.lines) == 0:
            line_colors = self.saliva_params['colormap']
            self._saliva_plot_helper(data, feature_name, groups, saliva_times, ylims, fontsize, ax,
                                     line_colors=line_colors)

            ax.text(x=self.test_times[0] + 0.5 * (self.test_times[1] - self.test_times[0]), y=0.95 * ylims[1],
                    s=test_text, horizontalalignment='center', verticalalignment='top', fontsize=fontsize)
            ax.axvspan(*self.test_times, color=test_color, alpha=test_alpha, zorder=1, lw=0)
            ax.axvspan(saliva_times[0] - x_padding, self.test_times[0], color=bg_color, alpha=bg_alpha, zorder=0, lw=0)
            ax.axvspan(self.test_times[1], saliva_times[-1] + x_padding, color=bg_color, alpha=bg_alpha, zorder=0, lw=0)

            ax.xaxis.set_major_locator(xaxis_tick_locator)
            ax.set_xlabel(xaxis_label, fontsize=fontsize)
            ax.set_xlim(saliva_times[0] - x_padding, saliva_times[-1] + x_padding)
        else:
            # the was already something drawn into the axis => we are using the same axis to add another feature
            ax_twin = ax.twinx()
            line_colors = self.saliva_params['multi.colormap']
            self._saliva_plot_helper(data, feature_name, groups, saliva_times, ylims, fontsize, ax_twin,
                                     x_offset_basis=self.saliva_params['multi.x_offset'],
                                     line_colors=line_colors)

        if len(groups) > 1:
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            # use them in the legend
            ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), numpoints=1,
                      prop={"size": fontsize})

        if fig:
            fig.tight_layout()
            return fig, ax

    def _saliva_plot_helper(self, data: pd.DataFrame, feature_name: str,
                            groups: Sequence[str], saliva_times: Sequence[int],
                            ylims: Sequence[float], fontsize: int,
                            ax: plt.Axes,
                            x_offset_basis: Optional[float] = 0,
                            line_colors: Optional[Sequence[Tuple]] = None) -> plt.Axes:
        # get all plot parameter
        line_styles = self.saliva_params['line_styles']
        markers = self.saliva_params['markers']
        x_offsets = list(np.array(self.saliva_params['x_offsets']) + x_offset_basis)
        yaxis_label = self.saliva_params['yaxis.label'][feature_name]
        if line_colors is None:
            line_colors = self.saliva_params['colormap']

        for group, x_off, line_color, marker, ls in zip(groups, x_offsets, line_colors, markers, line_styles):
            if group == 'Data':
                # no condition index
                df_grp = data
            else:
                df_grp = data.xs(group, level="condition")
            ax.errorbar(x=saliva_times + x_off, y=df_grp["mean"], label=group,
                        yerr=df_grp["se"], capsize=3, marker=marker, color=line_color, ls=ls)

        ax.set_ylabel(yaxis_label, fontsize=fontsize)
        ax.set_ylim(ylims)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        return ax

    def saliva_plot_combine_legend(self, figure: plt.Figure, ax: plt.Axes, biomarkers: Sequence[str],
                                   separate_legends: Optional[bool] = False):
        """
        TODO: add documentation

        Parameters
        ----------
        figure
        ax
        biomarkers
        separate_legends

        Returns
        -------

        """
        from matplotlib.legend_handler import HandlerTuple

        fontsize = self.saliva_params['multi.fontsize']
        legend_offset = self.saliva_params['multi.legend_offset']

        labels = [ax.get_legend_handles_labels()[1] for ax in figure.get_axes()]
        if all([len(l) == 1 for l in labels]):
            # only one group
            handles = [ax.get_legend_handles_labels()[0] for ax in figure.get_axes()]
            handles = [h[0] for handle in handles for h in handle]
            labels = biomarkers
            ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), prop={"size": fontsize})
        else:
            if separate_legends:
                for (i, a), biomarker in zip(enumerate(reversed(figure.get_axes())), reversed(biomarkers)):
                    handles, labels = a.get_legend_handles_labels()
                    l = ax.legend(handles, labels, title=biomarker, loc='upper right',
                                  bbox_to_anchor=(0.99 - legend_offset * i, 0.99), prop={"size": fontsize})
                    ax.add_artist(l)
            else:
                handles = [ax.get_legend_handles_labels()[0] for ax in figure.get_axes()]
                handles = [h[0] for handle in handles for h in handle]
                labels = [ax.get_legend_handles_labels()[1] for ax in figure.get_axes()]
                labels = ["{}:\n{}".format(b, " - ".join(l)) for b, l in zip(biomarkers, labels)]
                ax.legend(list(zip(handles[::2], handles[1::2])), labels, loc='upper right',
                          bbox_to_anchor=(0.99, 0.99), numpoints=1,
                          handler_map={tuple: HandlerTuple(ndivide=None)}, prop={"size": fontsize})
