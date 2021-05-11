"""Base class for representing psychological protocols."""
from typing import Dict, Sequence, Union, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

from biopsykit.colors import fau_palette_tech, fau_palette_blue
import biopsykit.protocols.plotting as plot
from biopsykit.protocols.utils import _check_sample_times_match, _get_sample_times
from biopsykit.utils.data_processing import (
    concat_phase_dict,
    split_subphases,
    split_groups,
    mean_se_nested_dict,
)
from biopsykit.utils.datatype_helper import (
    SalivaMeanSeDataFrame,
    is_saliva_mean_se_dataframe,
    SubjectConditionDict,
    SubjectConditionDataFrame,
)
from biopsykit.utils.exceptions import ValidationError


class BaseProtocol:
    def __init__(
        self,
        name: str,
    ):
        self.name: str = name
        """
        Study name
        """

        self.saliva_type: Sequence[str] = []
        self.test_times: Sequence[int] = [0, 0]
        self.sample_times: Dict[str, Sequence[int]] = {}
        self.saliva_data: Dict[str, SalivaMeanSeDataFrame] = {}

        self._saliva_params = {}

        self.saliva_params: Dict = {
            "colormap": fau_palette_blue("line_2"),
            "line_styles": ["-", "--"],
            "markers": ["o", "P"],
            # 'background.color': "#e0e0e0",
            # 'background.alpha': 0.5,
            # 'x_padding': 0.1,
            "test_color": "#9e9e9e",
            "test_alpha": 0.5,
            "x_offsets": [0, 0.5],
            "multi.x_offset": 1,
            "multi.legend_offset": 0.3,
            "multi.colormap": fau_palette_tech("line_2"),
            "xaxis.tick_locator": plt.MultipleLocator(20),
            "yaxis.label": {
                "cortisol": "Cortisol [nmol/l]",
                "amylase": "Amylase [U/l]",
                "il6": "IL-6 [pg/ml]",
            },
        }

    def __repr__(self):
        return self.__str__()

    @property
    def saliva_params(self) -> Dict:
        return self._saliva_params

    @saliva_params.setter
    def saliva_params(self, saliva_params: Dict):
        self._saliva_params.update(saliva_params)

    def add_saliva_data(
        self,
        saliva_type: Union[str, Sequence[str]],
        saliva_data: Union[SalivaMeanSeDataFrame, Dict[str, SalivaMeanSeDataFrame]],
        sample_times: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
        test_times: Optional[Sequence[int]] = None,
    ):
        if isinstance(saliva_type, str):
            saliva_type = [saliva_type]
        self.saliva_type = saliva_type

        if test_times is not None:
            self.test_times = test_times

        if saliva_data is not None:
            if not isinstance(sample_times, dict):
                sample_times = {key: sample_times for key in self.saliva_type}

            if not isinstance(saliva_data, dict):
                saliva_data = {key: saliva_data for key in self.saliva_type}

            self.sample_times = _get_sample_times(saliva_data, sample_times, self.test_times)
            self.saliva_data.update(self._add_saliva_data(saliva_data, self.saliva_type, self.sample_times))

    def _add_saliva_data(
        self,
        data: Union[SalivaMeanSeDataFrame, Dict[str, SalivaMeanSeDataFrame]],
        saliva_type: Union[str, Sequence[str]],
        sample_times: Union[Sequence[int], Dict[str, Sequence[int]]],
    ):
        saliva_data = {}
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    saliva_data[key] = self._add_saliva_data(value, key, sample_times[key])
            else:
                is_saliva_mean_se_dataframe(data)
                _check_sample_times_match(data, sample_times)
                saliva_data[saliva_type] = data
        except ValidationError as e:
            raise ValidationError(
                "'data' is not a 'SalivaMeanSeDataFrame' (or a dict of such). "
                "Before setting saliva data you need to compute mean and standard error of per sample using "
                "`biopsykit.saliva.mean_se`. The validation raised the following error:\n\n{}".format(str(e))
            ) from e
        return saliva_data

    def concat_phase_dict(
        self, dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]], phases: Sequence[str]
    ) -> Dict[str, pd.DataFrame]:
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

        return concat_phase_dict(dict_hr_subject, phases)

    def split_subphases(
        self,
        data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
        subphase_names: Sequence[str],
        subphase_times: Sequence[Tuple[int, int]],
        is_group_dict: Optional[bool] = False,
    ) -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]],]:
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
        return split_subphases(
            data=data,
            subphase_names=subphase_names,
            subphase_times=subphase_times,
            is_group_dict=is_group_dict,
        )

    @classmethod
    def split_groups(
        cls,
        phase_dict: Dict[str, pd.DataFrame],
        condition_list: Union[SubjectConditionDict, SubjectConditionDataFrame],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Splits 'Phase dict' into group dict, i.e. one 'Phase dict' per group.

        Parameters
        ----------
        phase_dict : dict
            'Phase dict' to be split in groups. See ``utils.concat_phase_dict`` for further information
        condition_list : dict
            dictionary of group membership. Keys are the different groups, values are lists of subject IDs that belong
            to the respective group

        Returns
        -------
        dict
            nested group dict with one 'Phase dict' per group

        """
        return split_groups(phase_dict=phase_dict, condition_list=condition_list)

    def mean_se_subphases(
        self,
        data: Union[
            Dict[str, Dict[str, pd.DataFrame]],
            Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        ],
        subphases: Optional[Sequence[str]] = None,
        is_group_dict: Optional[bool] = False,
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
        return mean_se_nested_dict(data, subphases=subphases, is_group_dict=is_group_dict)

    def saliva_plot(
        self,
        saliva_type: Optional[str] = "cortisol",
        groups: Optional[Sequence[str]] = None,
        group_col: Optional[str] = None,
        **kwargs,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        TODO: add documentation

        Parameters
        ----------
        saliva_type
        groups
        group_col
        **kwargs

        Returns
        -------

        """
        if len(self.saliva_type) == 0:
            raise ValueError("No saliva data to plot!")

        kwargs.update(self.saliva_params)
        return plot.saliva_plot(
            data=self.saliva_data[saliva_type],
            biomarker=saliva_type,
            saliva_times=self.sample_times[saliva_type],
            test_times=self.test_times,
            groups=groups,
            group_col=group_col,
            **kwargs,
        )

    def _saliva_plot_helper(
        self, data: pd.DataFrame, biomarker: str, groups: Sequence[str], saliva_times: Sequence[int], **kwargs
    ) -> plt.Axes:
        return plot._saliva_plot_helper(
            data=data, biomarker=biomarker, groups=groups, saliva_times=saliva_times, **kwargs
        )

    def saliva_plot_combine_legend(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        biomarkers: Sequence[str],
        separate_legends: Optional[bool] = False,
    ):
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
