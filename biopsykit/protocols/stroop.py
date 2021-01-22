from typing import Dict, Tuple, Union, Optional, Sequence
import csv
from biopsykit.protocols import base
import biopsykit.colors as colors
import pandas as pd
import os
from tqdm.notebook import tqdm
import numpy as np

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

        self.phases: Sequence[str] = ["Stroop1", "Stroop2", "Stroop3"]
        """
        Stroop Phases

        Names of Stroop phases
        """

        self.phase_durations: Sequence[int] = [1 * 60, 1 * 60, 1 * 60]
        """
        Stroop Phase Durations

        Total duration of phases in seconds
        """
        self.hr_ensemble_plot_params = {
            'colormap': colors.cmap_fau_blue('3'),
            'line_styles': ['-', '--', ':'],
            'background.color': ['#e0e0e0', '#9e9e9e', '#757575'],
            'background.alpha': [0.5, 0.5, 0.5],
            'fontsize': 14,
            'xaxis.label': r"Stroop Subphases",
            'yaxis.label': r"$\Delta$HR [%]",
            'mist.phase_text': "Stroop Phase {}",
            'mist.end_phase_text': "End Phase {}",
        }

        self.hr_mean_plot_params = {
            'colormap': colors.cmap_fau_blue('2_lp'),
            'line_styles': ['-', '--'],
            'markers': ['o', 'P'],
            'background.color': ["#e0e0e0", "#bdbdbd", "#9e9e9e"],
            'background.alpha': [0.5, 0.5, 0.5],
            'x_offsets': [0, 0.05],
            'fontsize': 14,
            'xaxis.label': "MIST Subphases",
            'yaxis.label': r"$\Delta$HR [%]",
            'mist.phase_text': "MIST Phase {}"
        }

        self.saliva_params = {
            'test.text': "Stroop",
            'xaxis.label': "Time relative to Stroop start [min]"
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

            #handle first iteration
            if first:
                tmp_ID = stroop_ID
                first = False
            #detect new ID --> save dict sub to corresponding ID
            elif(stroop_ID != tmp_ID):
                dict_sub['stroop_results'] = result
                dict_sub['stroop_times'] = times
                dict_stroop_data[tmp_ID] = dict_sub
                tmp_ID = stroop_ID
                dict_sub = {}
                times = {}
                result = pd.DataFrame()


            dict_tmp['stroop_result']['phase'] = stroop_n
            columns = ['phase'] + list(dict_tmp['stroop_result'])[8:16]
            result = result.append(dict_tmp['stroop_result'][columns]).reset_index(drop=True)
            times[stroop_n] = (df_tmp['stroop_result']['time'][0], str(pd.to_timedelta(
                df_tmp['stroop_result']['time'][0]) + pd.to_timedelta(duration, unit='s'))[7:])

            #calculate stroop phase time
            endtime = pd.to_timedelta(dict_tmp['stroop_result']['starttime'][0]) + pd.to_timedelta(
                dict_tmp['stroop_result']['elapsedtime'][0]//1000, unit='s')
            starttime = endtime - pd.to_timedelta(duration, unit='s')
            times[stroop_n] = (str(starttime)[7:], str(endtime)[7:])

        dict_sub['stroop_results'] = result
        dict_sub['stroop_times'] = times
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

    def get_stroop_dataframe(self, data=Dict[str,Dict], columns: Optional[Sequence[str]] = None):

        if columns:
            if 'phase' not in columns:
                columns = ['phase'] + columns
            if 'subject' not in columns:
                columns = ['subject'] + columns
        else:
            columns = ['subject', 'phase', 'propcorrect', 'meanRT']

        df_stroop = pd.DataFrame(columns=columns)
        df_tmp = pd.DataFrame(columns=columns)

        columns.remove('phase')
        columns.remove('subject')

        for subject_id, data_dict in tqdm(data.items(), desc="Subjects"):
            df_tmp = data_dict['stroop_results'][['phase']+columns]
            df_tmp['subject'] = subject_id
            df_stroop = df_stroop.append(df_tmp, ignore_index=True)

        df_stroop = df_stroop.set_index(['subject', 'phase']).sort_index()

        df_stroop['propcorrect'] = df_stroop['propcorrect'].str.replace(',', '.').replace(np.nan, '1.0').astype(float)
        df_stroop['meanRT'] = df_stroop['meanRT'].str.replace(',', '.').replace(np.nan, '1.0').astype(float)

        return df_stroop

    def get_stroop_mean(self,data=Union[Dict[str,Dict],pd.DataFrame], columns:Optional[Sequence[str]] = None) -> pd.DataFrame:

        if columns:
            if 'phase' not in columns:
                columns = ['phase'] + columns
            if 'subject' not in columns:
                columns = ['subject'] + columns
        else:
            columns = ['subject', 'phase', 'propcorrect', 'meanRT']

        if isinstance(data, Dict):
            df_stroop = self.get_stroop_dataframe(data, columns)

        if isinstance(data, pd.DataFrame):
            df_stroop = data
            columns.remove('phase')
            columns.remove('subject')

        labels = ['Stroop1', 'Stroop2', 'Stroop3']
        columns_mean = [col + '_mean' for col in columns]
        df_stroop_mean = pd.DataFrame(columns=['phase']+columns_mean)
        df_stroop_mean['phase'] = labels

        for label in labels:
            index = df_stroop_mean['phase'] == label
            df_stroop_mean[columns_mean[0]][index] = round(
                df_stroop.groupby('phase').get_group(label)[columns[0]].mean(), 4) * 100
            df_stroop_mean[columns_mean[1]][index] = round(
                df_stroop.groupby('phase').get_group(label)[columns[1]].mean(), 2)

        return df_stroop_mean