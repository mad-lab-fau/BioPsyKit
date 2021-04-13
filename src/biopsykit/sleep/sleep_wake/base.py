from typing import Union

import pandas as pd
import numpy as np


class _SleepWakeBase:
    def predict(self, data: Union[pd.DataFrame, np.array]) -> Union[np.array, pd.DataFrame]:
        pass
