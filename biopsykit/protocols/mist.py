from typing import Dict, Tuple, Union, Optional, Sequence

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import biopsykit.signals.ecg as ecg
from biopsykit.signals.ecg import EcgProcessor
import biopsykit.utils as utils
from biopsykit.protocols.utils import split_subphases, split_groups, concat_phase_dict, hr_course


class MIST:
    """
    Class representing the Montreal Imaging Stress Task (MIST).
    """

    def __init__(
            self,
            name: Optional[str] = None,
            phases: Optional[Sequence[str]] = None,
            subphases: Optional[Sequence[str]] = None,
            subphase_durations: Optional[Sequence[int]] = None
    ):
        self.name: str = name or "MIST"
        """
        Study name
        """

        self.phases: Sequence[str] = ["MIST1", "MIST2", "MIST3"]
        """
        MIST Phases
        
        Names of MIST phases
        """

        self.subphases: Sequence[str] = ['BL', 'AT', 'FB']
        """
        MIST Subphases
        
        Names of MIST subphases
        """

        self.subphase_durations: Sequence[int] = [60, 240, 0]
        """
        MIST Subphase Durations
        
        Total duration of subphases in seconds
        """

        self.hr_ensemble_params = {
            'colormap': utils.cmap_fau_blue('3'),
            'line_styles': ['-', '--', ':'],
            'background.color': ['#e0e0e0', '#9e9e9e', '#757575'],
            'background.alpha': [0.6, 0.7, 0.7],
        }

        self.hr_course_params = {
            'colormap': utils.cmap_fau_blue('2_lp'),
            'line_styles': ['-', '--'],
            'markers': ['o', 'P'],
            'background.color': ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
            'background.alpha': [0.6, 0.7, 0.7],
            'x_offsets': [0, 0.05]
        }

        self._update_mist_params(phases, subphases, subphase_durations)

    def __str__(self) -> str:
        return """{}
        Phases: {}
        Subphases: {}
        Subphase Durations: {}
        """.format(self.name, self.phases, self.subphases, self.subphase_durations)

    def __repr__(self):
        return self.__str__()

    def _update_mist_params(self, phases: Sequence[str], subphases: Sequence[str], subphase_durations: Sequence[int]):
        if phases:
            self.phases = phases
        if subphases:
            self.subphases = subphases
        if subphase_durations:
            self.subphase_durations = subphase_durations

    def cut_feedback_interval(
            self,
            dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Cuts heart rate data of each subject to equal length, i.e. to the minimal duration of each MIST phase
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
        # skip Part1 and Part2, extract only MIST Phases

        durations = np.array(
            [[len(df) for phase, df in dict_hr.items() if phase not in ['Part1', 'Part2']] for dict_hr in
             dict_hr_subject.values()])

        # minimal duration of each MIST Phase
        min_dur = {phase: dur for phase, dur in zip(self.phases, np.min(durations, axis=0))}

        for subject_id, dict_hr in dict_hr_subject.items():
            dict_hr_cut = {}
            for phase in self.phases:
                dict_hr_cut[phase] = dict_hr[phase][0:min_dur[phase]]
            dict_hr_subject[subject_id] = dict_hr_cut

        return dict_hr_subject

    def concat_mist_dict(self, dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Rearranges the 'HR subject dict' (see `utils.load_hr_excel_all_subjects`) into 'MIST Phase dict'.
        See ``bp.protocols.utils.concat_phase_dict`` for further information.

        Parameters
        ----------
        dict_hr_subject : dict
            'HR subject dict', i.e. a nested dict with heart rate data per MIST phase and subject

        Returns
        -------
        dict
            'MIST dict', i.e. a dict with heart rate data of all subjects per MIST phase

        """
        return concat_phase_dict(dict_hr_subject, self.phases)

    def split_mist_subphases(
            self,
            phase_dict: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
            is_group_dict: Optional[bool] = False
    ) -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
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

        Returns
        -------
        dict
            'Subphase dict' with course of HR data per Stress Test phase, subphase and subject, respectively or
            nested dict of 'Subphase dicts' if `is_group_dict` is ``True``

        """
        mist_subphase_times = self.get_mist_times(phase_dict=phase_dict, is_group_dict=is_group_dict)
        return split_subphases(data=phase_dict, subphase_names=self.subphases, subphase_times=mist_subphase_times,
                               is_group_dict=is_group_dict)

    @classmethod
    def split_mist_groups(cls, phase_dict: Dict[str, pd.DataFrame],
                          condition_dict: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
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
        return split_groups(phase_dict, condition_dict)

    def get_mist_times(
            self,
            mist_dur: Optional[Sequence[int]] = None,
            phase_dict: Optional[Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]] = None,
            is_group_dict: Optional[bool] = False
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
            raise ValueError("Either `mist_dur` or `phase_dict` must be supplied as parameter!")

        if mist_dur:
            # ensure numpy
            mist_dur = np.array(mist_dur)
        else:
            if is_group_dict:
                # Grouped MIST Phase dict
                mist_dur = np.array([[len(v) for v in d.values()] for d in phase_dict.values()])
                if not (mist_dur == mist_dur[0]).all():
                    # ensure that durations of all groups are equal
                    raise ValueError("All groups are expected to have the same durations for the single phases!")
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
        return [(start, end) for start, end in zip(np.append([0], times_cum[:-1]), times_cum)]

    def hr_course_mist(
            self,
            data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
            is_group_dict: Optional[bool] = False
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

        return hr_course(data, self.subphases, is_group_dict)

    # TODO add kw_args
    def hr_ensemble_plot(
            self,
            data: Dict[str, pd.DataFrame],
            plot_params: Optional[Dict] = None,
            ylims: Optional[Sequence[float]] = None,
            fontsize: Optional[int] = 14,
            ax: Optional[plt.Axes] = None,
            figsize: Optional[Tuple[float, float]] = None
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
        fontsize : int, optional. Default: ``None``
            font size. Default: 14
        ax : plt.Axes, optional
            Axes to plot on, otherwise create a new one. Default: ``None``
        figsize : tuple, optional
            figure size

        Returns
        -------
        tuple or none
            Tuple of Figure and Axes or None if Axes object was passed
        """
        import matplotlib.ticker as mticks

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        if plot_params:
            self.hr_ensemble_params.update(plot_params)

        # sns.despine()
        sns.set_palette(self.hr_ensemble_params['colormap'])

        line_styles = self.hr_ensemble_params['line_styles']
        subphases = np.array(self.subphases)

        mist_dur = [len(v) for v in data.values()]
        start_end = self.get_mist_times(mist_dur)

        for i, key in enumerate(data):
            hr_mist = data[key]
            x = hr_mist.index
            hr_mean = hr_mist.mean(axis=1)
            hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
            ax.plot(x, hr_mean, zorder=2, label="MIST Phase {}".format(i + 1), linestyle=line_styles[i])
            ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=0.4)
            ax.vlines(x=mist_dur[i], ymin=-20, ymax=40, linestyles='dashed', colors="#bdbdbd", zorder=3)
            ax.text(x=mist_dur[i] - 5, y=10 + 5 * i, s="End Phase {}".format(i + 1), fontsize=fontsize - 4,
                    horizontalalignment='right', bbox=dict(facecolor='#e0e0e0', alpha=0.7, boxstyle='round'), zorder=3)

        ax.set_ylabel(r'$\Delta$HR [%]', fontsize=fontsize)
        ax.set_xlabel(r'Time [s]', fontsize=fontsize)
        ax.set_xticks([start for (start, end) in start_end])
        ax.xaxis.set_minor_locator(mticks.MultipleLocator(60))
        ax.tick_params(axis="x", which='both', bottom=True)
        ax.tick_params(axis="y", which='major', left=True)

        for (start, end), subphase in zip(start_end, subphases):
            ax.text(x=start + 0.5 * (end - start), y=35, s=subphase, horizontalalignment='center', fontsize=fontsize)

        ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85), prop={'size': fontsize})
        if ylims:
            ax.set_ylim(ylims)
        ax._xmargin = 0

        for (start, end), color, alpha in zip(start_end, self.hr_ensemble_params['background.color'],
                                              self.hr_ensemble_params['background.alpha']):
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)

        if fig:
            fig.tight_layout()
            return fig, ax

    # TODO add support for groups in one dataframe (indicated by group column)
    # TODO add kw_args
    def hr_course_plot(
            self,
            data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
            groups: Optional[Sequence[str]] = None,
            group_col: Optional[str] = None,
            plot_params: Optional[Dict] = None,
            ylims: Optional[Sequence[float]] = None,
            fontsize: Optional[int] = 14,
            ax: Optional[plt.Axes] = None,
            figsize: Optional[Tuple[float, float]] = None
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
        ylims : list, optional
            y axis limits or ``None`` to infer y axis limits from data. Default: ``None``
        fontsize : int, optional. Default: ``None``
            font size. Default: 14
        ax : plt.Axes, optional
            Axes to plot on, otherwise create a new one. Default: ``None``
        figsize : tuple, optional
            figure size


        Returns
        -------
        tuple or none
            Tuple of Figure and Axes or None if Axes object was passed
        """

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        # update default parameter if plot parameter were passe
        if plot_params:
            self.hr_course_params.update(plot_params)

        # get all plot parameter
        sns.set_palette(self.hr_course_params['colormap'])
        line_styles = self.hr_course_params['line_styles']
        markers = self.hr_course_params['markers']
        bg_colors = self.hr_course_params['background.color']
        bg_alphas = self.hr_course_params['background.alpha']
        x_offsets = self.hr_course_params['x_offsets']

        if isinstance(data, pd.DataFrame):
            # get subphase labels from data
            subphase_labels = data.index.get_level_values(1)
            if not ylims:
                ylims = [1.1 * (data['mean'] - data['se']).min(), 1.5 * (data['mean'] + data['se']).max()]
            if group_col and not groups:
                # get group names from data if groups were not supplied
                groups = list(data[group_col].unique())
        else:
            # get subphase labels from data
            subphase_labels = list(data.values())[0].index.get_level_values(1)
            if not ylims:
                ylims = [1.1 * min([(d['mean'] - d['se']).min() for d in data.values()]),
                         1.5 * max([(d['mean'] + d['se']).max() for d in data.values()])]
            if not groups:
                # get group names from dict if groups were not supplied
                groups = list(data.keys())

        num_subph = len(self.subphases)
        # build x axis, axis limits and limits for MIST phase spans
        x = np.arange(len(subphase_labels))
        xlims = np.append(x, x[-1] + 1)
        xlims = xlims[::num_subph] + 0.5 * (xlims[::num_subph] - xlims[::num_subph] - 1)
        span_lims = [(x_l, x_u) for x_l, x_u in zip(xlims, xlims[1::])]

        # plot data as errorbar with mean and se
        if groups:
            for group, x_off, marker, ls in zip(groups, x_offsets, markers, line_styles):
                ax.errorbar(x=x + x_off, y=data[group]['mean'], label=group, yerr=data[group]['se'], capsize=3,
                            marker=marker, linestyle=ls)
        else:
            ax.errorbar(x=x, y=data['mean'], yerr=data['se'], capsize=3, marker=markers[0], linestyle=line_styles[0])

        # add decorators: spans and MIST Phase labels
        for (i, name), (x_l, x_u), color, alpha in zip(enumerate(self.phases), span_lims, bg_colors, bg_alphas):
            ax.axvspan(x_l, x_u, color=color, alpha=alpha, zorder=0, lw=0)
            ax.text(x=x_l + 0.5 * (x_u - x_l), y=0.9 * ylims[-1], s='MIST Phase {}'.format(i + 1),
                    horizontalalignment='center',
                    verticalalignment='top',
                    fontsize=fontsize)

        # customize x axis
        ax.tick_params(axis='x', bottom=True)
        ax.set_xticks(x)
        ax.set_xticklabels(subphase_labels)
        ax.set_xlim([span_lims[0][0], span_lims[-1][-1]])

        # customize y axis
        ax.set_ylabel("$\Delta$HR [%]", fontsize=fontsize)
        ax.tick_params(axis="y", which='major', left=True)
        ax.set_ylim(ylims)

        # customize legend
        if groups:
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            # use them in the legend
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.85), numpoints=1,
                      prop={"size": fontsize})

        if fig:
            fig.tight_layout()
            return fig, ax

    def param_subphases(
            self,
            ecg_processor: Optional[ecg.EcgProcessor] = None,
            dict_ecg: Optional[Dict[str, pd.DataFrame]] = None,
            dict_rpeaks: Optional[Dict[str, pd.DataFrame]] = None,
            param_types: Optional[Union[str, Sequence[str]]] = 'all',
            sampling_rate: Optional[int] = 256, include_total: Optional[bool] = True,
            title: Optional[str] = None
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

        if ecg_processor is None and dict_rpeaks is None and dict_ecg is None:
            raise ValueError("Either `ecg_processor` or `dict_rpeaks` and `dict_ecg` must be passed as arguments!")

        # get all desired parameter types
        possible_param_types = {'hrv': EcgProcessor.hrv_process, 'rsp': EcgProcessor.rsp_rsa_process}
        if param_types == 'all':
            param_types = possible_param_types

        if isinstance(param_types, str):
            param_types = {param_types: possible_param_types[param_types]}
        if not all([param in possible_param_types for param in param_types]):
            raise ValueError(
                "`param_types` must all be of {}, not {}".format(possible_param_types.keys(), param_types.keys()))

        param_types = {param: possible_param_types[param] for param in param_types}

        if ecg_processor:
            sampling_rate = ecg_processor.sampling_rate
            dict_rpeaks = ecg_processor.rpeaks
            dict_ecg = ecg_processor.ecg_result

        if 'rsp' in param_types and dict_ecg is None:
            raise ValueError("`dict_ecg` must be passed if param_type is {}!".format(param_types))

        index_name = "Subphase"
        # dict to store results. one entry per parameter and a list of dataframes per MIST phase
        # that will later be concated to one large dataframes
        dict_df_subphases = {param: list() for param in param_types}

        # iterate through all phases in the data
        for (phase, rpeaks), (ecg_phase, ecg_data) in tqdm(zip(dict_rpeaks.items(), dict_ecg.items()), desc=title):
            rpeaks = rpeaks.copy()
            ecg_data = ecg_data.copy()

            # dict to store intermediate results of subphases. one entry per parameter with a
            # list of dataframes per subphase that will later be concated to one dataframe per MIST phase
            dict_subphases = {param: list() for param in param_types}
            if include_total:
                # compute HRV, RSP over complete phase
                for param_type, param_func in param_types.items():
                    dict_subphases[param_type].append(
                        param_func(ecg_signal=ecg_data, rpeaks=rpeaks, index="Total", index_name=index_name,
                                   sampling_rate=sampling_rate))

            if phase not in ["Part1", "Part2"]:
                # skip Part1, Part2 for subphase parameter analysis (parameters in total are computed above)
                for subph, dur in zip(self.subphases, self.subphase_durations):
                    # get the first xx seconds of data (i.e., get only the current subphase)
                    # TODO change to mist.mist_get_times?
                    if dur > 0:
                        df_subph_rpeaks = rpeaks.first('{}S'.format(dur))
                    else:
                        # duration of 0 seconds = Feedback Interval, don't cut slice the beginning,
                        # use all remaining data
                        df_subph_rpeaks = rpeaks
                    # ECG does not need to be sliced because rpeaks are already sliced and
                    # will select only the relevant ECG signal parts anyways
                    df_subph_ecg = ecg_data

                    for param_type, param_func in param_types.items():
                        # compute HRV, RSP over subphases
                        dict_subphases[param_type].append(
                            param_func(ecg_signal=df_subph_ecg, rpeaks=df_subph_rpeaks, index=subph,
                                       index_name=index_name,
                                       sampling_rate=sampling_rate))

                    # remove the currently analyzed subphase of data
                    # (so that the next subphase is first in the next iteration)
                    rpeaks = rpeaks[~rpeaks.index.isin(df_subph_rpeaks.index)]

                # if len(self.subphase_durations) < len(self.subphases):
                #     # add Feedback Interval (= remaining time) if present
                #     for param_type, param_func in param_types.items():
                #         dict_subphases[param_type].append(
                #             param_func(ecg_signal=ecg, rpeaks=rpeaks, index=self.subphases[-1], index_name=index_name,
                #                        sampling_rate=sampling_rate))

            for param in dict_subphases:
                # concat dataframe of all subphases to one dataframe per MIST phase and add to parameter dict
                dict_df_subphases[param].append(pd.concat(dict_subphases[param]))

        # concat all dataframes together to one big result dataframes
        return pd.concat(
            [pd.concat(dict_df, keys=dict_rpeaks.keys(), names=["Phase"]) for dict_df in dict_df_subphases.values()],
            axis=1)
