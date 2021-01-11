from typing import Dict, Tuple, Union, Optional, Sequence
import csv
from biopsykit.protocols import base
import biopsykit.colors as colors
import pandas as pd

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

    def get_stroop_test_results(df_stroop: Optional[pd.DataFrame] = None,
                                dict_stroop: Optional[Dict] = None,
                                iqdat_stroop: Optional[str] = None,
                          ) -> Dict[str, pd.DataFrame]:

        dict_result = {}
        parameters = ['subjectid','sessionid','starttime','elapsedtime','propcorrect','meanRT']
        print(df_stroop)
        print(dict_stroop)
        print(iqdat_stroop)
        if isinstance(df_stroop, pd.DataFrame):
            print('macht was 1')
            dict_result['stroop_result'] = df_stroop[parameters]

        if isinstance(df_stroop, Dict):
            dict_result['stroop_result'] = pd.DataFrame(df_stroop, index=[0])[parameters]

        if isinstance(df_stroop, str):
            with open(df_stroop,'r') as f:
                reader = csv.reader(f, dialect='excel', delimiter='\t')
                list_values = []
                for row in reader:
                    list_values.append(row)

                dict_result['stroop_result'] = pd.DataFrame(dict(zip(list_values[0], list_values[1])), index=[0])[parameters]

        return dict_result