from typing import Dict, Tuple, Union, Optional, Sequence
import csv
from biopsykit.protocols import base
import biopsykit.colors as colors
import pandas as pd
import os

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
        #TODO: for Schleife für dict_sub einfügen --> momentan überschreibt das die Werte
        #TODO: kommentieren
        dict_stroop_data = {}
        dataset = os.listdir(folder)
        for data in dataset:
            dict_sub = {}
            dict_sub_sub = {}
            print(data)
            if ('raw' in data):
                continue

            if data.endswith('.csv'):
                dict_sub_sub = {'stroop_result': pd.read_csv(folder + data, sep=';')}
                df_tmp = pd.read_csv(folder + data.replace('summary', 'raw'), sep=';')

            elif data.endswith('.iqdat'):
                dict_sub_sub = self.get_stroop_test_results(data_stroop=folder + data)
                df_tmp = self.get_stroop_test_results(data_stroop=folder + data.replace('summary','raw'))

            stroop_ID = dict_sub_sub['stroop_result']['subjectid'][0]

            stroop_n = 'Stroop ' + str(dict_sub_sub['stroop_result']['sessionid'][0])[-1]
            print('Stroop ' + str(dict_sub_sub['stroop_result']['sessionid'][0])[-1])
            duration = int(
                df_tmp['stroop_result']['latency'].astype(int).sum() // 1000 + (len(df_tmp['stroop_result']) + 1) * 0.5)

            dict_sub_sub['stroop_times'] = (df_tmp['stroop_result']['time'][0], str(pd.to_timedelta(
                                            df_tmp['stroop_result']['time'][0]) + pd.to_timedelta(duration, unit='s'))[7:])
            dict_sub[stroop_n] = dict_sub_sub

            dict_stroop_data[stroop_ID] = dict_sub

        return dict_stroop_data

    def get_stroop_test_results(self, data_stroop: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ###TODO: kommentieren
        dict_result = {}

        if isinstance(data_stroop, pd.DataFrame):
            print('macht was 1')
            dict_result['stroop_result'] = data_stroop

        if isinstance(data_stroop, Dict):
            dict_result['stroop_result'] = pd.DataFrame(data_stroop, index=[0])

        if isinstance(data_stroop, str):
            first = True
            with open(data_stroop, 'r') as f:
                reader = csv.reader(f, dialect='excel', delimiter='\t')
                for row in reader:
                    if first:
                        columns = row
                        df = pd.DataFrame(columns=columns)
                        first = False
                        continue
                    df = df.append(pd.DataFrame(columns=columns, data=[row]), ignore_index =True)

                dict_result['stroop_result'] = df

        return dict_result