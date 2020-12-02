from typing import Sequence, Optional

import numpy as np
import pandas as pd


def split_nights(data: pd.DataFrame, min_time: Optional[int] = 5) -> Sequence[pd.DataFrame]:
    idx_split = np.where(np.diff(data.index, prepend=data.index[0]) > pd.Timedelta(min_time, 'hours'))[0]
    list_nights = np.split(data, idx_split)
    return list_nights
