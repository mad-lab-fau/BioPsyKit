import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.rescore import rescore




class Webster(_SleepWakeBase):
    """
    Runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and activity is represented
    by an activity index.
    """

    scale_factor: float = 0.025


    def __init__(self, **kwargs):
        """
        Create an instance of the Sadeh Algorithm class for sleep/wake detection.
        """

        self.scale_factor = kwargs.get("scale_factor", self.scale_factor)


    def fit(self):
        pass


    def predict(self, data: Union[pd.DataFrame, np.array], rescore_data: Optional[bool] = True) -> Union[np.array, pd.DataFrame]:
        """
        Perform the sleep/wake score prediction.

        Parameters
        ----------
        data : pd.DataFrame
            pandas dataframe of activity index values.

        Returns
        -------
        np.array
            predictions with sleep = True and wake = False
        """
        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)


        # ensure numpy
        # problem in my view: with np.convolve and "same" we dont make the "unsymmetric" convolution... ( formula is with A-4 and A+2)
        # possible solution: np.valid and append 4 zeros in front and 2 zeros at the back or like implemented
        sf = np.array(self.scale_factor)
        kernel = sf * np.array([0.15, 0.15, 0.15, 0.08, 0.21, 0.12, 0.13, 0 ,0])
        scores = np.convolve(data, kernel, "same")
        scores[scores >= 1] = 1     #wake
        scores[scores < 1] = 0      #sleep






        if rescore_data:
            scores = rescore(scores)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])

        return scores

