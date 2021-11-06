"""Module with utility functions to handle sleep data."""
import datetime
from typing import Dict

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype


def split_nights(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split continuous data into individual nights.

    This function splits data into individual nights. The split is performed at 6pm because that's the time of day
    where the probability of sleeping is the lowest.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        input data

    Returns
    -------
    dict
        dictionary with recording date (keys) and dataframes split into individual nights (values).
        By convention, if the recording of one night started after 12 am the recording date is set to the previous date.

    """
    _assert_is_dtype(data.index, pd.DatetimeIndex)
    # split data per day
    date_diff = np.diff(data.index.date)
    date_diff = np.append(date_diff[0], date_diff)
    idx_date = np.where(date_diff)[0]

    # split data per time: split at 6 pm, because that's the time of day where the probability of sleeping is the lowest
    time_diff = data.index.time <= pd.Timestamp("18:00:00").time()
    time_diff = np.append(time_diff[0], time_diff)
    idx_time = np.where(np.diff(time_diff))[0]

    # concatenate both splitting criteria and split data
    idx_split = np.unique(np.concatenate([idx_date, idx_time]))
    data_split = np.split(data, idx_split)

    time_6pm = pd.Timestamp("18:00:00").time()

    # concatenate data from one night (data between 6 pm and 12 am from the previous day and
    # between 12 am and 6 pm of the next day)
    for i, df in enumerate(data_split):
        if i < (len(data_split) - 1):
            df_curr = df
            df_next = data_split[i + 1]

            date_curr = df_curr.index[0].date()
            date_next = df_next.index[0].date()
            time_curr = df_curr.index[0].time()
            time_next = df_next.index[0].time()

            # check if dates are consecutive and if first part is after 6 pm and second part is before 6 am
            if (date_next == date_curr + datetime.timedelta(days=1)) and (time_curr > time_6pm > time_next):
                data_split[i] = pd.concat([df_curr, df_next])
                # delete the second part
                del data_split[i + 1]

    # create dict with data from each night. dictionary keys are the dates.
    dict_data = {}
    for df in data_split:
        date = df.index[0].normalize().date()
        # By convention, if the recording started after 12 am the recording date is set to the previous date
        if df.index[0].time() < pd.Timestamp("18:00:00").time():
            date = date - pd.Timedelta("1d")
        dict_data[str(date)] = df

    return dict_data
