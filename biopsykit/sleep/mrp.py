from typing import Union

import pandas as pd
import numpy as np

from biopsykit.utils.array_handling import sliding_window


class MajorRestPeriod:
    sampling_rate: int

    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate

    def predict(self, data: Union[pd.DataFrame, pd.Series, np.array]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df_acc = data.filter(like='acc')
        else:
            df_acc = pd.DataFrame(data)

        # rolling median 5 second
        df_acc = df_acc.rolling(int(5 * self.sampling_rate), min_periods=0).median()

        # get angle
        angle = np.arctan(
            df_acc['acc_z'] / ((df_acc['acc_x'] ** 2 + df_acc['acc_y'] ** 2) ** 0.5)
        ) * (180.0 / np.pi)

        window_s = 5  # 5 seconds
        overlap = 0

        angle_sliding = sliding_window(angle, window_sec=window_s,
                                       sampling_rate=self.sampling_rate,
                                       overlap_percent=overlap)
        index_resample = sliding_window(df_acc.index.values, window_sec=window_s,
                                        sampling_rate=self.sampling_rate,
                                        overlap_percent=overlap)[:, 0]

        if isinstance(df_acc.index, pd.DatetimeIndex):
            index_resample = pd.DatetimeIndex(index_resample)
            index_resample = index_resample.tz_localize('UTC').tz_convert(df_acc.index.tzinfo)

        df_angle = pd.DataFrame(np.abs(np.diff(np.nanmean(angle_sliding, axis=1))), columns=['angle'],
                                index=index_resample[1:])
        df_angle = df_angle.rolling(60).median()  # rolling median, 60 * 5 seconds per sample = 5 minutes

        minimum_rest_threshold = 0.0
        maximum_rest_threshold = 1000.0

        # calculate and apply threshold
        thresh = np.min(
            [np.max([np.percentile(df_angle['angle'].dropna().values, 10) * 15.0, minimum_rest_threshold]),
             maximum_rest_threshold])

        df_angle[df_angle < thresh] = 0.0
        df_angle[df_angle >= thresh] = 1.0

        minimum_rest_block = 30
        allowed_rest_break = 60

        # drop rest blocks < minimum_rest_block minutes (except first and last)
        df_angle['block'] = (df_angle['angle'].diff().ne(0)).cumsum()
        groups = list(df_angle.groupby(by='block'))
        # exclude first and last rest block
        for idx, group in groups[1:-1]:
            if group['angle'].sum() == 0 and len(group) < (12 * minimum_rest_block):
                # 5 second intervals => 12x for 1min
                df_angle.loc[group.index[0]: group.index[-1], 'angle'] = 1

        # drop active blocks < allowed_rest_break minutes (except first and last)
        df_angle["block"] = (df_angle['angle'].diff().ne(0)).cumsum()
        groups = list(df_angle.groupby(by="block"))
        for idx, group in groups[1:-1]:
            if group['angle'].sum() == len(group) and len(group) < (12 * allowed_rest_break):
                # 5 second intervals => 12x for 1min
                df_angle.loc[group.index[0]: group.index[-1], 'angle'] = 0

        # get longest block
        df_angle["block"] = (df_angle['angle'].diff().ne(0)).cumsum()
        group = df_angle.groupby("block")
        grp_max = group.get_group(group.size().idxmax())

        total_duration = (df_acc.index[-1] - df_acc.index[0])
        if isinstance(df_acc.index, pd.DatetimeIndex):
            total_duration = total_duration.total_seconds() / 3600.0
            start = grp_max.index[0]
            end = grp_max.index[-1]
        else:
            total_duration = (total_duration / self.sampling_rate) / 3600.0
            start = int(grp_max.index[0] / (self.sampling_rate * 60))
            end = int(grp_max.index[-1] / (self.sampling_rate * 60))

        mrp = {'start': start, 'end': end, 'total_duration': total_duration}

        return pd.DataFrame(mrp, index=[0])