from typing import Dict, Tuple, Union, Optional, Sequence
import csv
from biopsykit.protocols import base
import biopsykit.colors as colors
import pandas as pd
import os
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import biopsykit.protocols.plotting as plot
import matplotlib.ticker as mticks
import seaborn as sns
from matplotlib.lines import Line2D

class Stroop(base.BaseProtocol):
    """
    Class representing the Stroop test.
    """

    def __init__(
            self, name: Optional[str] = None,
            phases: Optional[Sequence[str]] = None,
            phase_durations: Optional[Sequence[int]] = None
    ):
        if name is None:
            name = "Stroop"
        super().__init__(name)

        self.stroop_times: Sequence[int] = [0, 10]

        self.phases: Sequence[str] = ["Stroop1", "Stroop2","Stroop3"]
        """
        Stroop Phases

        Names of Stroop phases
        """

        self.phase_durations: Sequence[int] = [3 * 60, 3 * 60, 3 * 60]
        """
        Stroop Phase Durations
        
        Total duration of phases in seconds
        """

        self.subphases: Sequence[str] = ['Stroop', 'Feedback']
        """
        Stroop Subphases

        Names of Stroop subphases
        """
        self.subphase_durations: Sequence[int] = [70, 60]
        """
        Stroop Subphase Durations

        Total duration of subphases in seconds
        """

        self.hr_ensemble_plot_params = {
            'colormap': colors.cmap_fau_blue('3_ens'),
            'line_styles': ['-', '--', '-.'],
            'ensemble_alpha': 0.4,
            'background_color': ['#e0e0e0', '#9e9e9e', '#757575'],
            'background_alpha': [0.5, 0.5, 0.5],
            'fontsize': 14,
            'xaxis_label': r"Time [s] ",
            'xaxis_minor_ticks': mticks.MultipleLocator(60),
            'yaxis_label': r"$\Delta$Mean HR [bpm]",
            'legend_loc': 'upper right',
            'legend_bbox_to_anchor': (0.25, 0.90),
            'phase_text': "Stroop Phase {}",
            'end_phase_text': "End Phase {}",
            'end_phase_line_color': "#e0e0e0",
            'end_phase_line_style': 'dashed',
            'end_phase_line_width': 2.0
        }
        self.stroop_plot_params = {
            'colormap': colors.cmap_fau_blue('3_ens'),
            'line_styles': ['-', '--', '-.'],
            'background_color': ['#e0e0e0', '#9e9e9e', '#757575'],
            'background_alpha': [0.5, 0.5, 0.5],
            'fontsize': 14,
            'xaxis_label': r"Stroop phases",
            'xaxis_minor_ticks': mticks.MultipleLocator(60),
            'yaxis_label': r"$\Delta$Mean HR [bpm]",
            'legend_loc': 'upper right',
            'legend_bbox_to_anchor': (1.00, 0.90),
            'phase_text': "Stroop Phase {}",
        }

        self.hr_mean_plot_params = {
            'colormap': colors.cmap_fau_blue('2_lp'),
            'line_styles': ['-', '--'],
            'markers': ['o', 'P'],
            'background_color': ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
            'background_alpha': [0.5, 0.5, 0.5],
            'x_offsets': [0, 0.05],
            'fontsize': 14,
            'xaxis_label': "Stroop Subphases",
            'yaxis_label': r"$\Delta$HR [%]",
            'mist_phase_text': "MIST Phase {}"
        }

        self.saliva_params = {
            'test_text': "Stroop",
            'xaxis_label': "Time relative to Stroop start [min]"
        }

        self._update_stroop_params(phases, phase_durations)

    def __str__(self) -> str:
        return """{}
        Phases: {}
        Phase Durations: {}
        """.format(self.name, self.phases, self.phase_durations)

    @property
    def stroop_times(self):
        return self.test_times

    @stroop_times.setter
    def stroop_times(self, stroop_times):
        self.test_times = stroop_times

    def _update_stroop_params(self, phases: Sequence[str], phase_durations: Sequence[int]):
        if phases:
            self.phases = phases
        if phase_durations:
            self.phase_durations = phase_durations

    def load_stroop_test_data(self, folder=str) -> Dict[str,pd.DataFrame]:
        #TODO: kommentieren
        dict_stroop_data = {}
        dataset = os.listdir(folder)
        tmp_ID = stroop_ID = ""
        dict_sub = {}
        first = True
        result = pd.DataFrame()
        #result_raw = pd.DataFrame()
        times = {}
        #iterate through data
        for data in dataset:

            #skip raw data --> will be handled later
            if ('raw' in data):
                continue

            #get data in .csv format
            if data.endswith('.csv'):
                dict_tmp = {'stroop_result': pd.read_csv(folder + data, sep=';')}
                df_tmp = pd.read_csv(folder + data.replace('summary', 'raw'), sep=';')

            elif data.endswith('.iqdat'):
                dict_tmp = self.get_stroop_test_results(data_stroop=folder + data)
                df_tmp = self.get_stroop_test_results(data_stroop=folder + data.replace('summary','raw'))

            #set ID, stroop phase and phase duration
            stroop_ID = dict_tmp['stroop_result']['subjectid'][0]
            stroop_n = 'Stroop' + str(dict_tmp['stroop_result']['sessionid'][0])[-1]
            duration = int(
                df_tmp['stroop_result']['latency'].astype(int).sum() // 1000 + (len(df_tmp['stroop_result']) + 1))
            #raw_data = df_tmp['stroop_result'][['latency','correct']]
            #handle first iteration
            if first:
                tmp_ID = stroop_ID
                first = False
            #detect new ID --> save dict sub to corresponding ID
            elif(stroop_ID != tmp_ID):
                dict_sub['stroop_results'] = result
                #dict_sub['stroop_raw'] = result_raw
                dict_sub['stroop_times'] = times
                dict_stroop_data[tmp_ID] = dict_sub
                tmp_ID = stroop_ID
                dict_sub = {}
                times = {}
                result = pd.DataFrame()
                #result_raw = pd.DataFrame()


            dict_tmp['stroop_result']['phase'] = stroop_n
            columns = ['phase'] + list(dict_tmp['stroop_result'])[8:16]
            result = result.append(dict_tmp['stroop_result'][columns]).reset_index(drop=True)

            #raw_data['phase'] = stroop_n
            #result_raw = result_raw.append(raw_data).reset_index(drop=True)
            times[stroop_n] = (df_tmp['stroop_result']['time'][0], str(pd.to_timedelta(
                df_tmp['stroop_result']['time'][0]) + pd.to_timedelta(duration, unit='s'))[7:])

            #calculate stroop phase time
            endtime = pd.to_timedelta(dict_tmp['stroop_result']['starttime'][0]) + pd.to_timedelta(
                dict_tmp['stroop_result']['elapsedtime'][0]//1000, unit='s')
            starttime = endtime - pd.to_timedelta(duration, unit='s')
            times[stroop_n] = (str(starttime)[7:], str(endtime)[7:])

        dict_sub['stroop_results'] = result
        dict_sub['stroop_times'] = times
        #dict_sub['stroop_raw'] = result_raw
        dict_stroop_data[stroop_ID] = dict_sub

        return dict_stroop_data

    def get_stroop_test_results(self, data_stroop: Union[pd.DataFrame, str, Dict]) -> Dict[str, pd.DataFrame]:
        ###TODO: kommentieren
        dict_result = {}

        if isinstance(data_stroop, pd.DataFrame):
            dict_result['stroop_result'] = data_stroop

        if isinstance(data_stroop, Dict):
            dict_result['stroop_result'] = pd.DataFrame(data_stroop, index=[0])

        if isinstance(data_stroop, str):
            dict_result['stroop_result'] = pd.read_csv(data_stroop, sep='\t')

        return dict_result
    def hr_ensemble_plot(
            self,
            data: Dict[str, pd.DataFrame],
            plot_params: Optional[Dict] = None,
            ylims: Optional[Sequence[float]] = None,
            ax: Optional[plt.Axes] = None,
            is_group_dict: Optional[bool] = False,
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
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        if plot_params:
            self.hr_ensemble_plot_params.update(plot_params)

        # sns.despine()
        sns.set_palette(self.hr_ensemble_plot_params['colormap'])
        line_styles = self.hr_ensemble_plot_params['line_styles']
        fontsize = self.hr_ensemble_plot_params['fontsize']
        xaxis_label = self.hr_ensemble_plot_params['xaxis_label']
        yaxis_label = self.hr_ensemble_plot_params['yaxis_label']
        xaxis_minor_ticks = self.hr_ensemble_plot_params['xaxis_minor_ticks']
        ensemble_alpha = self.hr_ensemble_plot_params['ensemble_alpha']
        bg_color = self.hr_ensemble_plot_params['background_color']
        bg_alpha = self.hr_ensemble_plot_params['background_alpha']
        phase_text = self.hr_ensemble_plot_params['phase_text']
        end_phase_text = self.hr_ensemble_plot_params['end_phase_text']
        end_phase_color = self.hr_ensemble_plot_params['end_phase_line_color']
        end_phase_line_style = self.hr_ensemble_plot_params['end_phase_line_style']
        end_phase_line_width = self.hr_ensemble_plot_params['end_phase_line_width']
        legend_loc = self.hr_ensemble_plot_params['legend_loc']
        legend_bbox_to_anchor = self.hr_ensemble_plot_params['legend_bbox_to_anchor']

        subphases = np.array(self.subphases)
        #mist_dur = [len(v) for v in data.values()]
        start_end = [(0,self.subphase_durations[0]),(self.subphase_durations[0],self.subphase_durations[0]+self.subphase_durations[1])]

        if is_group_dict:
            for j,condition in enumerate(data):
                mist_dur = [len(v) for v in data[condition].values()]
                for i, key in enumerate(data[condition]):
                    pal = sns.color_palette()[j]

                    hr_mist = data[condition][key]
                    x = hr_mist.index
                    hr_mean = hr_mist.mean(axis=1)
                    hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
                    ax.plot(x, hr_mean, zorder=2, label=phase_text.format(i + 1)+ ' - ' +condition, linestyle=line_styles[i], color=pal)
                    ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=ensemble_alpha)
                    ax.vlines(x=mist_dur[i] - 0.5, ymin=0, ymax=1, transform=ax.get_xaxis_transform(),
                              ls=end_phase_line_style, lw=end_phase_line_width,
                              colors=end_phase_color, zorder=3)
                    ax.annotate(
                        text=end_phase_text.format(i + 1),
                        xy=(mist_dur[i], 0.85 - 0.05 * i),
                        xytext=(-5, 0),
                        xycoords=ax.get_xaxis_transform(),
                        textcoords='offset points',
                        ha='right',
                        fontsize=fontsize - 4,
                        bbox=dict(facecolor='#e0e0e0', alpha=0.7, boxstyle='round'),
                        zorder=3
                    )
                ax.legend(loc=legend_loc, bbox_to_anchor=(0.20,0.3), prop={'size': fontsize})
        else:
            mist_dur = [len(v) for v in data.values()]
            for i, key in enumerate(data):
                hr_mist = data[key]
                x = hr_mist.index
                hr_mean = hr_mist.mean(axis=1)
                hr_stderr = hr_mist.std(axis=1) / np.sqrt(hr_mist.shape[1])
                ax.plot(x, hr_mean, zorder=2, label=phase_text.format(i + 1), linestyle=line_styles[i])
                ax.fill_between(x, hr_mean - hr_stderr, hr_mean + hr_stderr, zorder=1, alpha=ensemble_alpha)
                ax.vlines(x=mist_dur[i] - 0.5, ymin=0, ymax=1, transform=ax.get_xaxis_transform(),
                          ls=end_phase_line_style, lw=end_phase_line_width,
                          colors=end_phase_color, zorder=3)
                ax.annotate(
                    text=end_phase_text.format(i + 1),
                    xy=(mist_dur[i], 0.85 - 0.05 * i),
                    xytext=(-5, 0),
                    xycoords=ax.get_xaxis_transform(),
                    textcoords='offset points',
                    ha='right',
                    fontsize=fontsize - 4,
                    bbox=dict(facecolor='#e0e0e0', alpha=0.7, boxstyle='round'),
                    zorder=3
                )
            ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, prop={'size': fontsize})

        for (start, end), subphase in zip(start_end, subphases):
            ax.text(x=start + 0.5 * (end - start), y=0.95, transform=ax.get_xaxis_transform(),
                    s=subphase, ha='center', va='center', fontsize=fontsize)
        p = mpatch.Rectangle(xy=(0, 0.9), width=1, height=0.1, transform=ax.transAxes, color='white', alpha=0.4,
                             zorder=3, lw=0)
        ax.add_patch(p)

        for (start, end), color, alpha in zip(start_end, bg_color, bg_alpha):
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)

        ax.set_xlabel(xaxis_label, fontsize=fontsize)
        ax.set_xticks([start for (start, end) in start_end])
        ax.xaxis.set_minor_locator(xaxis_minor_ticks)
        ax.tick_params(axis="x", which='both', bottom=True)

        ax.set_ylabel(yaxis_label, fontsize=fontsize)
        ax.tick_params(axis="y", which='major', left=True)

        if ylims:
            ax.margins(x=0)
            ax.set_ylim(ylims)
        else:
            ax.margins(0, 0.1)


        if fig:
            fig.tight_layout()
            return fig, ax
    
    def hr_ensemble_plot_old(
            self,
            data: Dict[str, pd.DataFrame],
            plot_params: Optional[Dict] = None,
            ylims: Optional[Sequence[float]] = None,
            ax: Optional[plt.Axes] = None,
            is_group_dict: Optional[bool] = False,
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
        is_group_dict : boolean, optional
            ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
            Default: ``False``

        Returns
        -------
        tuple or none
            Tuple of Figure and Axes or None if Axes object was passed
        """
        # TODO add option to apply moving average filter before plotting ensemble plot

        import matplotlib.patches as mpatch

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        if plot_params:
            self.hr_ensemble_plot_params.update(plot_params)

        # sns.despine()
        sns.set_palette(self.hr_ensemble_plot_params['colormap'])
        line_styles = self.hr_ensemble_plot_params['line_styles']
        fontsize = self.hr_ensemble_plot_params['fontsize']
        xaxis_label = self.hr_ensemble_plot_params['xaxis_label']
        yaxis_label = self.hr_ensemble_plot_params['yaxis_label']
        xaxis_minor_ticks = self.hr_ensemble_plot_params['xaxis_minor_ticks']
        ensemble_alpha = self.hr_ensemble_plot_params['ensemble_alpha']
        bg_color = self.hr_ensemble_plot_params['background_color']
        bg_alpha = self.hr_ensemble_plot_params['background_alpha']
        legend_loc = self.hr_ensemble_plot_params['legend_loc']
        legend_bbox_to_anchor = self.hr_ensemble_plot_params['legend_bbox_to_anchor']

        phases = np.array(self.phases)
        mist_dur = [len(v) for v in data.values()]

        if is_group_dict:
            conditions=[]
            for j, condition in enumerate(data):
                start_end = []
                add=0
                pal = sns.color_palette()[j]
                conditions.append(condition)
                print(conditions)
                for i, key in enumerate(data[condition]):
                    stroop = data[condition][key]
                    start_end.append((add+1, add+len(stroop)))
                    x = stroop.index + add
                    add += len(stroop)
                    stroop_mean = stroop.mean(axis=1)
                    stroop_stderr = stroop.std(axis=1) / np.sqrt(stroop.shape[1])
                    ax.plot(x, stroop_mean, zorder=2, linestyle=line_styles[j],color=pal)
                    ax.fill_between(x, stroop_mean - stroop_stderr, stroop_mean + stroop_stderr, zorder=1, alpha=ensemble_alpha)
            lines = [Line2D([0], [0], color=pal, linewidth=3, linestyle=line_style) for line_style in line_styles[:2]]
            ax.legend(lines, conditions,loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, prop={'size': fontsize})
        else:
            start_end = []
            add=0
            pal = sns.color_palette()[0]
            for i, key in enumerate(data):

                stroop = data[key]
                start_end.append((add+1, add + len(stroop)))
                x = stroop.index + add
                add += len(stroop)
                stroop_mean = stroop.mean(axis=1)
                stroop_stderr = stroop.std(axis=1) / np.sqrt(stroop.shape[1])
                ax.plot(x, stroop_mean, zorder=2, label=key,color=pal, linestyle=line_styles[i])
                ax.fill_between(x, stroop_mean - stroop_stderr, stroop_mean + stroop_stderr, zorder=1, alpha=ensemble_alpha)
            ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, prop={'size': fontsize})

        for (start, end), phase in zip(start_end, phases):
            ax.text(x=start + 0.5 * (end - start), y=0.95, transform=ax.get_xaxis_transform(),
                    s=phase, ha='center', va='center', fontsize=fontsize)
        p = mpatch.Rectangle(xy=(0, 0.9), width=1, height=0.1, transform=ax.transAxes, color='white', alpha=0.4,
                             zorder=3, lw=0)
        ax.add_patch(p)

        for (start, end), color, alpha in zip(start_end, bg_color, bg_alpha):
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0, lw=0)

        ax.set_xlabel(xaxis_label, fontsize=fontsize)
        ax.set_xticks([start for (start, end) in start_end])
        ax.xaxis.set_minor_locator(xaxis_minor_ticks)
        ax.tick_params(axis="x", which='both', bottom=True)

        ax.set_ylabel(yaxis_label, fontsize=fontsize)
        ax.tick_params(axis="y", which='major', left=True)

        if ylims:
            ax.margins(x=0)
            ax.set_ylim(ylims)
        else:
            ax.margins(0, 0.1)


        if fig:
            fig.tight_layout()
            return fig, ax
    def hr_mean_subphases(
            self,
            data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
            is_group_dict: Optional[bool] = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Computes the heart rate mean and standard error per Stroop phase over all subjects.
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

        return super()._mean_se_subphases(data, subphases=self.subphases, is_group_dict=is_group_dict)

    def get_stroop_dataframe(self, dict_stroop=Dict[str,Dict], columns: Optional[Sequence[str]] = None,
                             is_group_dict: Optional[bool] = False):

        if columns:
            if 'phase' not in columns:
                columns = ['phase'] + columns

            if 'subject' not in columns:
                columns = ['subject'] + columns
        else:
            columns = ['subject', 'phase', 'propcorrect', 'meanRT']
        if is_group_dict:
            columns = ['condition'] + columns
        df_stroop = pd.DataFrame(columns=columns)
        df_tmp = pd.DataFrame(columns=columns)

        columns.remove('phase')
        columns.remove('subject')
        if is_group_dict: columns.remove('condition')
        if is_group_dict:
            for condition,dict_data in tqdm(dict_stroop.items(), desc="Subjects"):
                for subject_id, data in tqdm(dict_data.items(), desc="Subjects"):
                    df_tmp = data['stroop_results'][['phase'] + columns]
                    df_tmp['subject'] = subject_id
                    df_tmp['condition'] = condition
                    df_stroop = df_stroop.append(df_tmp, ignore_index=True)
            df_stroop = df_stroop.set_index(['condition','subject', 'phase']).sort_index()
        else:
            for subject_id, data_dict in tqdm(dict_stroop.items(), desc="Subjects"):
                df_tmp = data_dict['stroop_results'][['phase']+columns]
                df_tmp['subject'] = subject_id
                df_stroop = df_stroop.append(df_tmp, ignore_index=True)

            df_stroop = df_stroop.set_index(['subject', 'phase']).sort_index()

        df_stroop['propcorrect'] = df_stroop['propcorrect'].str.replace(',', '.').replace(np.nan, '1.0').astype(float)
        df_stroop['meanRT'] = df_stroop['meanRT'].str.replace(',', '.').replace(np.nan, '1.0').astype(float)

        return df_stroop

    def stroop_mean_se(self,data=pd.DataFrame,is_group_dict: Optional[bool]=False) -> pd.DataFrame:

        labels = self.subphases
        columns = list(data)
        data_concat = pd.DataFrame()
        if is_group_dict:
            for cols in columns:
                mean = pd.DataFrame({cols + '_mean-' + condition: {
                    stroop: data.loc[pd.IndexSlice[condition, :, stroop]][cols].mean() for stroop in
                    labels} for condition in ['IG', 'KG']})
                std = pd.DataFrame({cols + '_std-' + condition: {
                    stroop: data.loc[pd.IndexSlice[condition, :, stroop]][cols].std() for stroop in
                    labels} for condition in ['IG', 'KG']})
                data_concat = pd.concat([data_concat, pd.concat([mean, std], axis=1)], axis=1)

            data_concat = data_concat.reset_index().rename(columns={'index': 'phase'})

            result = pd.wide_to_long(data_concat,
                                        stubnames=['meanRT_mean', 'meanRT_std', 'propcorrect_mean', 'propcorrect_std'],
                                        sep='-', suffix='\D+', i=['phase'], j='condition')


        else:
            mean = pd.DataFrame({cols + '_mean': {stroop: data.xs(stroop, level='phase')[cols].mean() for stroop in labels} for
                    cols in columns})
            std = pd.DataFrame({cols + '_std': {stroop: data.xs(stroop, level='phase')[cols].std() for stroop in labels} for
                   cols in columns})
            result = pd.concat([mean, std], axis=1)
        result[['propcorrect_mean', 'propcorrect_std']] = result[['propcorrect_mean', 'propcorrect_std']]*100
        return result

    def stroop_plot(self, data=pd.DataFrame, variable: Optional[str] = 'meanRT',
                    is_group_dict: Optional[bool] = False,
                    group_col: Optional[str] = 'condition',
                    ylims: Optional[Sequence[float]] = None,
                    ax: Optional[plt.Axes] = None,
                    **kwargs) -> Union[Tuple[plt.Figure, plt.Axes], None]:
        """
        Plots the mean response time or correct answers during the different Stroop task (mean ± standard error per phase).

        In case of only one group a pandas dataframe can be passed.

        In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe belongs
        to one group, or one dataframe with a column indicating group membership (parameter ``group_col``).

        Regardless of the kind of input the dataframes need to be in the format of a 'mean dataframe', as returned
        by ``stroop_mean`` (see ``Stroop.stroop_mean`` for further information).


        Parameters
        ----------
        data : dataframe or dict
            Mean response/Correct answers data to plot. It has to be one dataframe which is in the kind of format as
            returned by `stroop_mean`
        variable : str
             Determines if the mean response times (``meanRT``) or correct answers (``propcorrect``) of the stroop
             test should be plotted.
             Default: ``meanRT``
        is_group_dict : bool, optional:
             List of group names. If ``None`` is passed, the groups and their order are inferred from the
             dictionary keys or from the unique values in `group_col`. If list is supplied the groups are
             plotted in that order.
             Default: ``None``
        group_col : str, optional
            Name of group column in the dataframe in case of multiple groups and one dataframe
        ylims : Tuple(int,int)
            Integer to scale the y axes.
            Default: ``None``
        ax : plt.Axes, optional
            Axes to plot on, otherwise create a new one. Default: ``None``
        kwargs: dict, optional
            optional parameters to be passed to the plot, such as:
                * figsize: tuple specifying figure dimensions
                * ylims: list to manually specify y-axis limits, float to specify y-axis margin (see ``Axes.margin()``
                for further information), None to automatically infer y-axis limits
        """

        fig: Union[plt.Figure, None] = None
        if ax is None:
            if 'figsize' in kwargs:
                figsize = kwargs['figsize']
            else:
                figsize = plt.rcParams['figure.figsize']
            fig, ax = plt.subplots(figsize=figsize)

        sns.set_palette(self.stroop_plot_params['colormap'])
        line_styles = self.stroop_plot_params['line_styles']
        fontsize = self.stroop_plot_params['fontsize']
        xaxis_label = self.stroop_plot_params['xaxis_label']
        xaxis_minor_ticks = self.stroop_plot_params['xaxis_minor_ticks']
        bg_color = self.stroop_plot_params['background_color']
        bg_alpha = self.stroop_plot_params['background_alpha']
        x_labels = self.subphases

        x = np.arange(len(x_labels))
        start_end = [(i - 0.5, i + 0.5) for i in x]
        if is_group_dict:
            conditions = list(set(data.index.get_level_values(group_col)))
            line1 = ax.errorbar(x, data.xs(conditions[0],level=group_col)[variable+'_mean'],
                                 yerr=data.xs(conditions[0],level=group_col)[variable+'_std'],
                                 color=sns.color_palette()[0],
                                label=conditions[0], lw=2, errorevery=1, ls=line_styles[0], marker="D", capsize=3)
            line2 = ax.errorbar(x, data.xs(conditions[1],level=group_col)[variable+'_mean'],
                                 yerr=data.xs(conditions[1],level=group_col)[variable+'_std'],
                                 color=sns.color_palette()[1],
                                 label=conditions[1], lw=2, errorevery=1, ls=line_styles[1], marker="D", capsize=3)
            plt.legend(handles=[line1, line2], loc='upper right', prop={'size': fontsize})
        else:
            ax.errorbar(x, data[variable+'_mean'],yerr=data[variable+'_std'],
                         color=sns.color_palette()[0], lw=2, errorevery=1, ls=line_styles[0],
                         marker="D", capsize=3)

        for (start, end), color, alpha in zip(start_end, bg_color, bg_alpha):
            ax.axvspan(start,end,color=color, alpha=alpha, zorder=0, lw=0)

        ax.set_xticklabels(x_labels, fontsize=fontsize)
        ax.set_xlabel(xaxis_label, fontsize=fontsize)
        ax.set_xticks([start + 0.5 for (start, end) in start_end])
        ax.xaxis.set_minor_locator(xaxis_minor_ticks)
        ax.tick_params(axis="x", which='both', bottom=True)

        if (variable == 'propcorrect'):
            ax.set_ylim(0, 105)
            ax.set_ylabel(r'$\Delta$Correct answers [%]',fontsize=fontsize)
        elif (variable == 'meanRT'):
            ax.set_ylabel(r'$\Delta$Response time [ms]', fontsize=fontsize)

        ax.tick_params(axis="y", which='major', left=True,labelsize=fontsize)

        if ylims:
            ax.margins(x=0)
            ax.set_ylim(ylims)
        else:
            ax.margins(0, 0.1)


        if fig:
            fig.tight_layout()
            return fig, ax


    def concat_phase_dict(
            self,
            dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
            **kwargs
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
        if 'phases' in kwargs:
            return super().concat_phase_dict(dict_hr_subject, kwargs['phases'])
        else:
            return super().concat_phase_dict(dict_hr_subject, self.phases)
    def split_groups_stroop(self,
                            dict_stroop=Dict[str,Dict[str,pd.DataFrame]],
                            condition_dict=Dict[str,Sequence[str]]) ->Dict[str, Dict[str, pd.DataFrame]]:
        """
        Splits 'Stroop dict' into group dict, i.e. one 'Stroop dict' per group.

        Parameters
        ----------
        phase_dict : dict
            'Dict stroop' to be split in groups. This is the outcome of 'stroop.load_stroop_test_data()'
        condition_dict : dict
            dictionary of group membership. Keys are the different groups, values are lists of subject IDs that
            belong to the respective group

        Returns
        -------
        dict
            group dict with one 'Stroop dict' per group

        """
        return {
            condition: {ID: dict_stroop[ID] for ID in IDs} for condition, IDs in condition_dict.items()
        }


    def split_groups(cls, phase_dict: Dict[str, pd.DataFrame],
                     condition_dict: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Splits 'Stroop Phase dict' into group dict, i.e. one 'Stroop Phase dict' per group.

        Parameters
        ----------
        phase_dict : dict
            'Stroop Phase dict' to be split in groups. See ``bp.protocols.utils.concat_phase_dict``
            for further information
        condition_dict : dict
            dictionary of group membership. Keys are the different groups, values are lists of subject IDs that
            belong to the respective group

        Returns
        -------
        dict
            nested group dict with one 'Stroop Phase dict' per group
        """

        return super().split_groups(phase_dict, condition_dict)


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
        Plots the course of heart rate during the complete Stroop test (mean ± standard error per phase).

        In case of only one group a pandas dataframe can be passed.

        In case of multiple groups either a dictionary of pandas dataframes can be passed, where each dataframe belongs
        to one group, or one dataframe with a column indicating group membership (parameter ``group_col``).

        Regardless of the kind of input the dataframes need to be in the format of a 'mse dataframe', as returned
        by ``stroop.hr_course_mist`` (see ``MIST.hr_course_mist`` for further information).


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
        return plot.hr_mean_plot(data=data, groups=groups, group_col=group_col, plot_params=self.hr_mean_plot_params,
                                 ax=ax, **kwargs)

    def hr_mean_se(
            self,
            data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
            is_group_dict: Optional[bool] = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Computes the heart rate mean and standard error per phase over all subjects.
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

        return super()._mean_se_subphases(data, is_group_dict=is_group_dict)

