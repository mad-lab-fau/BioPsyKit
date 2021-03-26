from typing import Sequence, Optional, Dict, Union

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

    def saliva_feature_boxplot(self, data: pd.DataFrame, x: str, saliva_type: str, feature: Optional[str] = None,
                               stats_kwargs: Optional[Dict] = None, **kwargs):
        from biopsykit.protocols.plotting import saliva_feature_boxplot
        return saliva_feature_boxplot(data, x, saliva_type, feature, stats_kwargs, **kwargs)

    def saliva_multi_feature_boxplot(self, data: pd.DataFrame, saliva_type: str, features: Union[str, Sequence[str]],
                                     filter_features: Optional[bool] = True, hue: Optional[str] = None,
                                     xticklabels: Optional[Dict[str, str]] = None,
                                     stats_kwargs: Optional[Dict] = None, **kwargs):
        from biopsykit.protocols.plotting import saliva_multi_feature_boxplot
        return saliva_multi_feature_boxplot(data, saliva_type, features, filter_features, hue, xticklabels,
                                            stats_kwargs, **kwargs)
