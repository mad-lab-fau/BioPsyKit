from typing import Union

import numpy as np
import pandas as pd

import biopsykit.signals.utils as su

class WearDetection:
    sampling_rate: int

    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate

    def predict(self, data: Union[pd.DataFrame, pd.Series, np.array]) -> pd.DataFrame:

        index = None
        index_resample = None
        if isinstance(data, (pd.DataFrame, pd.Series)):
            index = data.index

        if isinstance(data, pd.DataFrame):
            data = data.filter(like='acc')
        else:
            data = pd.DataFrame(data)

        window = 60  # min
        overlap = 15  # min
        overlap_percent = 1.0 - (overlap / window)

        acc_sliding = {
            col: su.sliding_window(data[col].values, window_sec=window * 60, sampling_rate=self.sampling_rate,
                                   overlap_percent=overlap_percent) for col in data
        }

        if index is not None:
            index_resample = su.sliding_window(index.values, window_sec=window * 60,
                                               sampling_rate=self.sampling_rate,
                                               overlap_percent=overlap_percent)[:, 0]
            if isinstance(index, pd.DatetimeIndex):
                index_resample = pd.DatetimeIndex(index_resample)
                index_resample = index_resample.tz_localize('UTC').tz_convert(index.tzinfo)

        acc_std = pd.DataFrame({axis: acc_sliding[axis].std(axis=1) for axis in acc_sliding})

        acc_std[acc_std >= 0.013] = 1
        acc_std[acc_std < 0.013] = 0
        acc_std = np.sum(acc_std, axis=1)

        acc_range = pd.DataFrame(
            {axis: acc_sliding[axis].max(axis=1) - acc_sliding[axis].min(axis=1) for axis in acc_sliding}
        )

        acc_range[acc_range >= 0.15] = 1
        acc_range[acc_range < 0.15] = 0
        acc_range = np.sum(acc_range, axis=1)

        wear = np.ones(shape=acc_std.shape)
        wear[np.logical_or(acc_std < 1.0, acc_range < 1.0)] = 0.0

        wear = pd.DataFrame(wear, columns=['wear'], index=index_resample)

        # apply rescoring three times
        wear = self._rescore_wear_detection(wear)
        wear = self._rescore_wear_detection(wear)
        wear = self._rescore_wear_detection(wear)

        return wear

    @staticmethod
    def _rescore_wear_detection(data: pd.DataFrame) -> pd.DataFrame:
        # group classifications into wear and non-wear blocks
        data['block'] = data['wear'].diff().ne(0).cumsum()
        blocks = list(data.groupby('block'))

        # iterate through blocks
        for (idx_prev, prev), (idx_curr, curr), (idx_post, post) in zip(blocks[0:-2], blocks[1:-1], blocks[2:]):
            if curr['wear'].unique():
                # get hour lengths of the previous, current, and next blocks
                dur_prev, dur_curr, dur_post = (len(dur) * 0.25 for dur in [prev, curr, post])

                if dur_curr < 3 and dur_curr / (dur_prev + dur_post) < 0.8:
                    # if the current block is less than 3 hours and the ratio to previous and post blocks is
                    # less than 80% rescore the wear period as non-wear
                    data.loc[data['block'] == idx_curr, "wear"] = 0
                elif dur_curr < 6 and dur_curr / (dur_prev + dur_post) < 0.3:
                    # if the current block is less than 6 hours and the ratio to previous and post blocks is
                    # less than 30% rescore the wear period as non-wear
                    data.loc[data['block'] == idx_curr, "wear"] = 0
        data.drop(columns=["block"], inplace=True)
        return data
