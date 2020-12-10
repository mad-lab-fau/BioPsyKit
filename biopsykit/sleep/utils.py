from typing import Sequence, Optional

import numpy as np
import pandas as pd


def split_nights(data: pd.DataFrame, diff_hours: Optional[int] = 12) -> Sequence[pd.DataFrame]:
    idx_split = np.where(np.diff(data.index, prepend=data.index[0]) > pd.Timedelta(diff_hours, 'hours'))[0]
    list_nights = np.split(data, idx_split)
    return list_nights
