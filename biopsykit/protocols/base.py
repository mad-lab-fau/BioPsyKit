from typing import Dict, Sequence, Union, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import biopsykit.colors as colors
import biopsykit.signals.utils as su
import biopsykit.protocols.plotting as plot


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
            # 'background.color': "#e0e0e0",
            # 'background.alpha': 0.5,
            # 'x_padding': 0.1,
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

        return su.mean_se_nested_dict(data=data, subphases=subphases, is_group_dict=is_group_dict)



    def saliva_plot(
            self,
            data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
            biomarker: Optional[str] = 'cortisol',
            saliva_times: Optional[Sequence[int]] = None,
            groups: Optional[Sequence[str]] = None,
            group_col: Optional[str] = None,
            ylims: Optional[Sequence[float]] = None,
            ax: Optional[plt.Axes] = None,
            figsize: Optional[Tuple[float, float]] = None
    ) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
        """
        TODO: add documentation

        Parameters
        ----------
        data
        biomarker
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

        return plot.saliva_plot(
            data=data, biomarker=biomarker, saliva_times=saliva_times, test_times=self.test_times,
            groups=groups, group_col=group_col, ylims=ylims, ax=ax, figsize=figsize, **self.saliva_params
        )

    def _saliva_plot_helper(self, data: pd.DataFrame, biomarker: str,
                            groups: Sequence[str], saliva_times: Sequence[int],
                            **kwargs) -> plt.Axes:
        return plot._saliva_plot_helper(
            data=data, biomarker=biomarker, groups=groups, saliva_times=saliva_times, **kwargs
        )

    def saliva_plot_combine_legend(self, fig: plt.Figure, ax: plt.Axes, biomarkers: Sequence[str],
                                   separate_legends: Optional[bool] = False):
        """
        TODO: add documentation

        Parameters
        ----------
        fig
        ax
        biomarkers
        separate_legends

        Returns
        -------

        """
        return plot.saliva_plot_combine_legend(fig=fig, ax=ax, biomarkers=biomarkers, separate_legends=separate_legends)
