from biopsykit.sleep.sleep_wake.base import _SleepWakeBase

import numpy as np
import pandas as pd

class Sazonov(_SleepWakeBase):

    def __init__(self):
        pass

    def predict(self, data: Union[pd.DataFrame, np.array]) -> Union[np.array, pd.DataFrame]:
        pass
