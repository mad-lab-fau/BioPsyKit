from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from typing import Dict, Union, Optional

from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.array_handling import sliding_window
from biopsykit.utils.rescore import rescore

import numpy as np
import pandas as pd

class Sazonov(_SleepWakeBase):

    def __init__(self):
        pass

    def predict(self, data: Union[pd.DataFrame, np.array], rescore_data: Optional[bool] = True) -> Union[np.array, pd.DataFrame]:



        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)


        w0 = data
        w1 = np.pad(np.max(sliding_window(data, window_samples=2,overlap_samples = 1),axis=1), (1, 0))
        w2 = np.pad(np.max(sliding_window(data, window_samples=3,overlap_samples = 2),axis=1), (2, 0))
        w3 = np.pad(np.max(sliding_window(data, window_samples=4,overlap_samples = 3),axis=1), (3, 0))
        w4 = np.pad(np.max(sliding_window(data, window_samples=5, overlap_samples=4), axis=1), (4, 0))

        scores = 1.727  - 0.256 * w0 - 0.154 * w1 - 0.136 * w2 - 0.140 * w3 - 0.176 * w4

        scores = 1 / (1 + np.exp(-scores))

        scores[scores >= 0.5] = 99
        scores[scores < 0.5] = 1
        scores[scores == 99] = 0


        if rescore_data:
            scores = rescore(scores)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])


        return scores
