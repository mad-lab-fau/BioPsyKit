from typing import Sequence

import pandas as pd


class CAR:
    """Class representing the cortisol awakening response (CAR)."""

    def __init__(self, saliva_times: Sequence[int]):
        self.saliva_times = saliva_times
        self.car_plot_params = {
            'xlabel': "Time after Awakening [min]",
            'ylabel': r"Cortisol [nmol/l]",
        }

    def car_plot(self, data: pd.DataFrame, **kwargs):
        from biopsykit.plotting import lineplot
        kwargs['x'] = kwargs.get('x', 'sample')
        kwargs['y'] = kwargs.get('y', kwargs.get('saliva_type', 'cortisol'))
        kwargs['xticklabels'] = self.saliva_times
        kwargs['xlabel'] = self.car_plot_params['xlabel']
        kwargs['ylabel'] = self.car_plot_params['ylabel']
        return lineplot(data, **kwargs)

    def saliva_feature_boxplot(self, data: pd.DataFrame, **kwargs):
        from biopsykit.protocols.plotting import saliva_feature_boxplot
        return saliva_feature_boxplot(data, **kwargs)
