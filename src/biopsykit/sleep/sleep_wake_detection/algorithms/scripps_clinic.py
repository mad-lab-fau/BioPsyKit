import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.utils.array_handling import sanitize_input_1d
from biopsykit.utils.rescore import rescore




class ScrippsClinic(_SleepWakeBase):
    """
    Runs sleep wake detection on epoch level activity data. Epochs are 1 minute long and activity is represented
    by an activity index.
    """

    scale_factor: float = 0.30


    def __init__(self, **kwargs):
        """
        Create an instance of the Scripps Clinic Algorithm class for sleep/wake detection.
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
        kernel = sf * np.array([0.0064, 0.0074, 0.0112, 0.0112, 0.0118, 0.0118, 0.0128, 0.0188 ,0.0280, 0.0664, 0.0300, 0.0112, 0.100,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        scores = np.convolve(data, kernel, "same")

        scores[scores >= 1] = 99     #wake = 0
        scores[scores < 1] = 1      #sleep = 1
        scores[scores == 99] = 0     #wake = 0






        if rescore_data:
            scores = rescore(scores)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])

        return scores

