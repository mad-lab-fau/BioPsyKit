from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase

from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.array_handling import sliding_window



from statsmodels.distributions.empirical_distribution import ECDF

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional



class PerezPozuelo(_SleepWakeBase):

    quantile: float = 0.85
    minimum_length = 30
    gap = 60


    def __init__(self, **kwargs):


        self.quantile = kwargs.get("quantile", self.quantile)



    def predict(self, data: Union[pd.DataFrame, np.array]) -> Union[np.array, pd.DataFrame]:


        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)


        dist = ECDF(data)

        heart_rate = dist.x[np.searchsorted(dist.y, self.quantile, side="left")]

        preprocessed = data
        preprocessed[data<=heart_rate] = 1
        preprocessed[data>heart_rate] = 0

        preprocessed = sliding_window(preprocessed,window_samples=10,overlap_samples=9)

        i=0



        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])
