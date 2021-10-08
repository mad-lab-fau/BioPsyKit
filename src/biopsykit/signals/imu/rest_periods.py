"""Module for computing Rest Periods from raw acceleration signals."""
import datetime
from typing import Union

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_num_columns
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sliding_window


class RestPeriods:
    """Compute Rest Periods from raw acceleration signals.

    Rest periods are periods with large inactivity, characterized by low angle changes in the acceleration signal.

    The longest rest period, the *Major Rest Period*, is used to determine the sleep window (i.e., time in bed)
    of the day.


    References
    ----------
    van Hees, V. T., Sabia, S., Anderson, K. N., Denton, S. J., Oliver, J., Catt, M., Abell, J. G., Kivimäki, M.,
    Trenell, M. I., & Singh-Manoux, A. (2015). A Novel, Open Access Method to Assess Sleep Duration Using a
    Wrist-Worn Accelerometer. *PLoS ONE*, 10(11), 1–13. https://doi.org/10.1371/journal.pone.0142533

    """

    sampling_rate: float

    def __init__(self, sampling_rate: float):
        """Initialize a new ``RestPeriods`` instance.

        Parameters
        ----------
        sampling_rate : float
            sampling rate of recorded data in Hz

        """
        self.sampling_rate = sampling_rate

    def predict(self, data: arr_t) -> pd.DataFrame:
        """Predict Rest Periods from acceleration data.

        Parameters
        ----------
        data : array_like
            input acceleration data. Must be 3-d.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with start, end, and total duration of each rest period detected by the algorithm

        """
        if isinstance(data, pd.DataFrame):
            data = data.filter(like="acc")
        else:
            data = pd.DataFrame(data)

        _assert_num_columns(data, 3)

        # rolling median 5 second
        data = data.rolling(int(5 * self.sampling_rate), min_periods=0).median()

        # get angle
        angle = np.arctan(data["acc_z"] / ((data["acc_x"] ** 2 + data["acc_y"] ** 2) ** 0.5)) * (180.0 / np.pi)

        window_s = 5  # 5 seconds
        overlap = 0

        angle_sliding = sliding_window(
            angle, window_sec=window_s, sampling_rate=self.sampling_rate, overlap_percent=overlap
        )

        index_resample = self._resample_index(data, window_s, overlap)

        df_angle = pd.DataFrame(
            np.abs(np.diff(np.nanmean(angle_sliding, axis=1))), columns=["angle"], index=index_resample[1:]
        )
        df_angle = df_angle.rolling(60).median()  # rolling median, 60 * 5 seconds per sample = 5 minutes

        minimum_rest_threshold = 0.0
        maximum_rest_threshold = 1000.0

        # calculate and apply threshold
        thresh = np.min(
            [
                np.max([np.percentile(df_angle["angle"].dropna().values, 10) * 15.0, minimum_rest_threshold]),
                maximum_rest_threshold,
            ]
        )

        df_angle[df_angle < thresh] = 0.0
        df_angle[df_angle >= thresh] = 1.0

        minimum_rest_block = 30
        allowed_rest_break = 60

        # drop rest blocks < minimum_rest_block minutes (except first and last)
        df_angle["block"] = (df_angle["angle"].diff().ne(0)).cumsum()
        groups = list(df_angle.groupby(by="block"))
        # exclude first and last rest block
        for _, group in groups[1:-1]:
            if group["angle"].sum() == 0 and len(group) < (12 * minimum_rest_block):
                # 5 second intervals => 12x for 1min
                df_angle.loc[group.index[0] : group.index[-1], "angle"] = 1

        # drop active blocks < allowed_rest_break minutes (except first and last)
        df_angle["block"] = (df_angle["angle"].diff().ne(0)).cumsum()
        groups = list(df_angle.groupby(by="block"))
        for _, group in groups[1:-1]:
            if group["angle"].sum() == len(group) and len(group) < (12 * allowed_rest_break):
                # 5 second intervals => 12x for 1min
                df_angle.loc[group.index[0] : group.index[-1], "angle"] = 0

        # get longest block
        df_angle["block"] = (df_angle["angle"].diff().ne(0)).cumsum()
        group = df_angle.groupby("block")
        grp_max = group.get_group(group.size().idxmax())

        total_duration = data.index[-1] - data.index[0]

        return self._major_rest_period(data, total_duration, grp_max)

    def _resample_index(self, data: pd.DataFrame, window_s: int, overlap: float):
        index_resample = sliding_window(
            data.index.values, window_sec=window_s, sampling_rate=self.sampling_rate, overlap_percent=overlap
        )[:, 0]

        if isinstance(data.index, pd.DatetimeIndex):
            index_resample = pd.DatetimeIndex(index_resample)
            index_resample = index_resample.tz_localize("UTC").tz_convert(data.index.tzinfo)
        return index_resample

    def _major_rest_period(
        self, data: pd.DataFrame, total_duration: Union[datetime.timedelta, float], grp_max: pd.DataFrame
    ) -> pd.DataFrame:
        if isinstance(data.index, pd.DatetimeIndex):
            total_duration = total_duration.total_seconds() / 3600.0
            start = grp_max.index[0]
            end = grp_max.index[-1]
        else:
            total_duration = (total_duration / self.sampling_rate) / 3600.0
            start = int(grp_max.index[0] / (self.sampling_rate * 60))
            end = int(grp_max.index[-1] / (self.sampling_rate * 60))

        mrp = {"start": start, "end": end, "total_duration": total_duration}
        return pd.DataFrame(mrp, index=[0])
