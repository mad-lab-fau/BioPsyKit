from typing import Sequence, Optional

import numpy as np
import pandas as pd


# TODO default split at 6pm because that's the time of day where the probability that people are sleeping is the lowest
def split_nights(data: pd.DataFrame, diff_hours: Optional[int] = 12) -> Sequence[pd.DataFrame]:
    idx_split = np.where(np.diff(data.index, prepend=data.index[0]) > pd.Timedelta(diff_hours, "hours"))[0]
    list_nights = np.split(data, idx_split)
    return list_nights
