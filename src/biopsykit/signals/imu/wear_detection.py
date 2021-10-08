"""Module for detection non-wear times from raw acceleration signals."""
import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_has_columns
from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import sliding_window


class WearDetection:
    """Detect non-wear times from raw acceleration signals.

    Non-wear times are estimated over 15 minute intervals over the day.

    """

    sampling_rate: float

    def __init__(self, sampling_rate: float):
        """Initialize a new ``WearDetection`` instance.

        Parameters
        ----------
        sampling_rate : float
            sampling rate of recorded data in Hz

        """
        self.sampling_rate = sampling_rate

    def predict(self, data: arr_t) -> pd.DataFrame:
        """Predict non-wear times from acceleration data.

        Parameters
        ----------
        data : array_like
            input acceleration data. Must be 3-d.

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with wear (1) and non-wear (0) times per 15 minute interval

        """
        index = None
        index_resample = None
        if isinstance(data, (pd.DataFrame, pd.Series)):
            index = data.index

        if isinstance(data, pd.DataFrame):
            data = data.filter(like="acc")
        else:
            data = pd.DataFrame(data)

        window = 60  # min
        overlap = 15  # min
        overlap_percent = 1.0 - (overlap / window)

        acc_sliding = {
            col: sliding_window(
                data[col].values,
                window_sec=window * 60,
                sampling_rate=self.sampling_rate,
                overlap_percent=overlap_percent,
            )
            for col in data
        }

        if index is not None:
            index_resample = self._resample_index(index, window, overlap_percent)

        acc_std = pd.DataFrame({axis: np.nanstd(acc_sliding[axis], ddof=1, axis=1) for axis in acc_sliding})

        acc_std[acc_std >= 0.013] = 1
        acc_std[acc_std < 0.013] = 0
        acc_std = np.nansum(acc_std, axis=1)

        acc_range = pd.DataFrame(
            {axis: np.nanmax(acc_sliding[axis], axis=1) - np.nanmin(acc_sliding[axis], axis=1) for axis in acc_sliding}
        )

        acc_range[acc_range >= 0.15] = 1
        acc_range[acc_range < 0.15] = 0
        acc_range = np.nansum(acc_range, axis=1)

        wear = np.ones(shape=acc_std.shape)
        wear[np.logical_or(acc_std < 1.0, acc_range < 1.0)] = 0.0

        wear = pd.DataFrame(wear, columns=["wear"])
        if index_resample is not None:
            wear = wear.join(index_resample)

        # apply rescoring three times
        wear = self._rescore_wear_detection(wear)
        wear = self._rescore_wear_detection(wear)
        wear = self._rescore_wear_detection(wear)

        return wear

    def _resample_index(self, index: pd.Index, window: int, overlap_percent: float):
        index_resample = sliding_window(
            np.arange(0, len(index)),
            window_sec=window * 60,
            sampling_rate=self.sampling_rate,
            overlap_percent=overlap_percent,
        )[:, :]
        start_end = index_resample[:, [0, -1]]
        if np.isnan(start_end[-1, -1]):
            last_idx = index_resample[-1, np.where(~np.isnan(index_resample[-1, :]))[0][-1]]
            start_end[-1, -1] = last_idx

        start_end = start_end.astype(int)

        if isinstance(index, pd.DatetimeIndex):
            index_resample = pd.DataFrame(index.values[start_end], columns=["start", "end"])
            index_resample = index_resample.apply(
                lambda df: pd.to_datetime(df).dt.tz_localize("UTC").dt.tz_convert(index.tzinfo)
            )
        return index_resample

    @staticmethod
    def _rescore_wear_detection(data: pd.DataFrame) -> pd.DataFrame:
        # group classifications into wear and non-wear blocks
        data["block"] = data["wear"].diff().ne(0).cumsum()
        blocks = list(data.groupby("block"))

        # iterate through blocks
        for (_, prev), (idx_curr, curr), (_, post) in zip(blocks[0:-2], blocks[1:-1], blocks[2:]):
            if curr["wear"].unique():
                # get hour lengths of the previous, current, and next blocks
                dur_prev, dur_curr, dur_post = (len(dur) * 0.25 for dur in [prev, curr, post])

                if dur_curr < 3 and dur_curr / (dur_prev + dur_post) < 0.8:
                    # if the current block is less than 3 hours and the ratio to previous and post blocks is
                    # less than 80% rescore the wear period as non-wear
                    data.loc[data["block"] == idx_curr, "wear"] = 0
                elif dur_curr < 6 and dur_curr / (dur_prev + dur_post) < 0.3:
                    # if the current block is less than 6 hours and the ratio to previous and post blocks is
                    # less than 30% rescore the wear period as non-wear
                    data.loc[data["block"] == idx_curr, "wear"] = 0
        data.drop(columns=["block"], inplace=True)
        return data

    @staticmethod
    def get_major_wear_block(data: pd.DataFrame) -> Tuple[Union[datetime.datetime, int], Union[datetime.datetime, int]]:
        """Return major wear block.

        The major wear block is the longest continuous wear block in the data.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            data with wear detection applied. The dataframe is expected to have a "wear" column.

        Returns
        -------
        start : :class:`~datetime.datetime` or int
            start of major wear block as datetime or int index
        end : :class:`~datetime.datetime` or int
            end of major wear block as datetime or int index

        See Also
        --------
        :meth:`~biopsykit.signals.imu.wear_detection.WearDetection.predict`
            apply wear detection on accelerometer data

        """
        data = data.copy()
        _assert_has_columns(data, [["wear"]])

        data["block"] = data["wear"].diff().ne(0).cumsum()
        wear_blocks = list(data.groupby("block").filter(lambda x: (x["wear"] == 1.0).all()).groupby("block"))
        max_block = wear_blocks[np.argmax([len(b) for i, b in wear_blocks])][1]
        max_block = (max_block["start"].iloc[0], max_block["end"].iloc[-1])
        return max_block

    @staticmethod
    def cut_to_wear_block(
        data: pd.DataFrame, wear_block: Tuple[Union[datetime.datetime, int], Union[datetime.datetime, int]]
    ) -> pd.DataFrame:
        """Cut data to wear block.

        Parameters
        ----------
        data : :class:`~pandas.DataFrame`
            input data that contains wear block
        wear_block : tuple
            tuple with start and end times of wear block. The type of ``wear_block`` depends on the index of ``data``.
            (datetime or int)

        Returns
        -------
        :class:`~pandas.DataFrame`
            data cut to wear block

        """
        if isinstance(data.index, pd.DatetimeIndex):
            return data.loc[wear_block[0] : wear_block[-1]]
        return data.iloc[wear_block[0] : wear_block[-1]]
