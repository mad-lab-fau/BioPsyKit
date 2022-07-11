"""Module for generating Activity Counts from raw acceleration signals."""
import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytz
from scipy import signal

from biopsykit.utils._types import arr_t
from biopsykit.utils.array_handling import add_datetime_index, downsample, sanitize_input_nd
from biopsykit.utils.datatype_helper import is_acc1d_dataframe, is_acc3d_dataframe
from biopsykit.utils.time import tz


class ActivityCounts:
    """Generate Activity Counts from raw acceleration signals.

    Actigraph Activity Counts are a unit used in many human activity studies.
    However, it can only be outputted by the official Actigraph Software.
    The following implementation uses a reverse engineered version of the Actigraph filter based on
    (Brønd et al., 2017).


    References
    ----------
    Brønd, J. C., Andersen, L. B., & Arvidsson, D. (2017). Generating ActiGraph Counts from Raw Acceleration Recorded
    by an Alternative Monitor. *Medicine and Science in Sports and Exercise*, 49(11), 2351–2360.
    https://doi.org/10.1249/MSS.0000000000001344

    """

    data: pd.DataFrame = None
    sampling_rate: float = None
    activity_counts_: np.ndarray = None
    timezone: datetime.tzinfo = tz

    def __init__(self, sampling_rate: float, timezone: Optional[str] = None):
        """Initialize a new ``ActivityCounts`` instance.

        Parameters
        ----------
        sampling_rate : float
            sampling rate of recorded data in Hz
        timezone: str
            timezone to which wear times will be converted

        """
        self.sampling_rate = sampling_rate
        if timezone:
            self.timezone = pytz.timezone(timezone)

    @staticmethod
    def _compute_norm(data: np.ndarray) -> np.ndarray:
        return np.linalg.norm(data, axis=1)

    @staticmethod
    def _aliasing_filter(data: np.ndarray, sampling_rate: Union[int, float]) -> np.ndarray:
        sos = signal.butter(5, [0.01, 7], "bp", fs=sampling_rate, output="sos")
        return signal.sosfiltfilt(sos, data)

    @staticmethod
    def _actigraph_filter(data: np.ndarray) -> np.ndarray:
        b = [
            0.04910898,
            -0.12284184,
            0.14355788,
            -0.11269399,
            0.05380374,
            -0.02023027,
            0.00637785,
            0.01851254,
            -0.03815411,
            0.04872652,
            -0.05257721,
            0.04784714,
            -0.04601483,
            0.03628334,
            -0.01297681,
            -0.00462621,
            0.01283540,
            -0.00937622,
            0.00344850,
            -0.00080972,
            -0.00019623,
        ]
        a = [
            1.00000000,
            -4.16372603,
            7.57115309,
            -7.98046903,
            5.38501191,
            -2.46356271,
            0.89238142,
            0.06360999,
            -1.34810513,
            2.47338133,
            -2.92571736,
            2.92983230,
            -2.78159063,
            2.47767354,
            -1.68473849,
            0.46482863,
            0.46565289,
            -0.67311897,
            0.41620323,
            -0.13832322,
            0.01985172,
        ]
        return signal.filtfilt(b, a, data)

    @staticmethod
    def _downsample(
        data: np.ndarray,
        sampling_rate: Union[int, float],
        final_sampling_rate: Union[int, float],
    ) -> np.ndarray:
        return downsample(data, sampling_rate, final_sampling_rate)

    @staticmethod
    def _truncate(data: np.ndarray) -> np.ndarray:
        upper_threshold = 2.13  # g
        lower_threshold = 0.068  # g
        data[data > upper_threshold] = upper_threshold
        data[data < lower_threshold] = 0
        return data

    @staticmethod
    def _digitize_8bit(data: np.ndarray) -> np.ndarray:
        max_val = 2.13  # g
        data //= max_val / (2**7)
        return data

    @staticmethod
    def _accumulate_second_bins(data: np.ndarray) -> np.ndarray:
        n_samples = 10
        #  Pad data at end to "fill" last bin
        padded_data = np.pad(data, (0, n_samples - len(data) % n_samples), "constant", constant_values=0)
        return padded_data.reshape((len(padded_data) // n_samples, -1)).sum(axis=1)

    def calculate(self, data: arr_t) -> arr_t:
        """Calculate Activity Counts from acceleration data.

        Parameters
        ----------
        data : array_like
            input data. Must either be 3-d or 1-d (e.g., norm, or a specific axis) acceleration data

        Returns
        -------
        array_like
            output data with Activity Counts

        """
        start_idx = None
        if isinstance(data, pd.DataFrame):
            # if dataframe, assert to be a acceleration dataframe according to biopsykit's convention
            if data.shape[1] == 3:
                is_acc3d_dataframe(data)
            if data.shape[1] == 1:
                is_acc1d_dataframe(data)
            data = data.filter(like="acc")
            if isinstance(data.index, pd.DatetimeIndex):
                start_idx = data.index[0]

        arr = sanitize_input_nd(data, ncols=(1, 3))

        if arr.shape[1] not in (1, 3):
            raise ValueError(
                f"{self.__class__.__name__} takes only 1-d or 3-d accelerometer data! Got {arr.shape[1]}-d data."
            )
        if arr.shape[1] != 1:
            arr = self._compute_norm(arr)

        arr = self._downsample(arr, self.sampling_rate, 30)
        arr = self._aliasing_filter(arr, 30)
        arr = self._actigraph_filter(arr)
        arr = self._downsample(arr, 30, 10)
        arr = np.abs(arr)
        arr = self._truncate(arr)
        arr = self._digitize_8bit(arr)
        arr = self._accumulate_second_bins(arr)
        if start_idx is not None:
            arr = add_datetime_index(arr, start_idx, 1 / 60, column_name=["activity_counts"])

        return arr
