import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.dtypes import (
    CPointDataFrame,
    HeartbeatSegmentationDataFrame,
    IcgRawDataFrame,
    is_b_point_dataframe,
    is_c_point_dataframe,
    is_heartbeat_segmentation_dataframe,
    is_icg_raw_dataframe,
)
from biopsykit.utils.exceptions import EventExtractionError

__all__ = ["BPointExtractionMiljkovic2022"]


class BPointExtractionMiljkovic2022(BaseBPointExtraction, CanHandleMissingEventsMixin):
    """B-point extraction algorithm by Miljkovic and Sekara (2022).

    This algorithm extracts B-points by transforming the ICG signal using a weighted time window applied to the
    segment preceding the maximal ICG peak (C-point). This transformation amplifies the characteristics of the B-point,
    facilitating B-point identification.


    For more information, see [Mil22]_.

    References
    ----------
    .. [Mil22] Miljković, N., & Šekara, T. B. (2022). A New Weighted Time Window-based Method to Detect B-point in
        Impedance Cardiogram (Version 3). arXiv. https://doi.org/10.48550/ARXIV.2207.04490

    """

    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new ``BPointExtractionMiljkovic2022`` instance.

        Parameters
        ----------
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle failing event extraction. Can be one of:
                * "warn": issue a warning and set the event to NaN
                * "raise": raise an ``EventExtractionError``
                * "ignore": ignore the error and continue with the next event
            Default: "warn"

        """
        super().__init__(handle_missing_events=handle_missing_events)

    def extract(
        self,
        *,
        icg: IcgRawDataFrame,
        heartbeats: HeartbeatSegmentationDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
    ):
        """Extract B-points from given ICG derivative signal.

        This algorithm extracts B-points by transforming the ICG signal using a weighted time window applied to the
        segment preceding the maximal ICG peak (C-point). This transformation amplifies the characteristics of the
        B-point, facilitating B-point identification.

        Parameters
        ----------
        icg : IcgRawDataFrame
            The raw ICG signal data.
        heartbeats : HeartbeatSegmentationDataFrame
            The heartbeat segmentation data.
        c_points : CPointDataFrame
            The C-point data.
        sampling_rate_hz : float
            The sampling rate of the ICG signal in Hz.

        Returns
        -------
        BPointDataFrame
            The extracted B-point data.

        """
        self._check_valid_missing_handling()
        is_icg_raw_dataframe(icg)
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_c_point_dataframe(c_points)
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe
        c_points = c_points["c_point_sample"]

        # get zero crossings of icg
        zero_crossings = np.where(np.diff(np.signbit(icg)))[0]
        # scaling factor for the window
        alpha = -0.1

        # iterate over each heartbeat
        for idx, _data in heartbeats.iterrows():
            # Get the C-Point location at the current heartbeat id
            c_point = c_points[idx]

            if pd.isna(c_point):
                b_points.loc[idx, "b_point_sample"] = np.nan
                b_points.loc[idx, "nan_reason"] = "c_point_nan"
                continue

            # Calculate the start position of the window (250 ms before the C-Point) and ensure that the start
            # position is not negative
            start_window = max(c_point - int((250 / 1000) * sampling_rate_hz), 0)

            icg_slice = icg.iloc[start_window:c_point].reset_index(drop=True)

            idx_start = icg_slice.idxmin()
            idx_stop = icg_slice.idxmax()
            icg_slice_window = icg_slice[idx_start:idx_stop]

            height = icg_slice.max() - icg_slice.min()

            # shift the segment so that the minimal value equals zero
            icg_slice -= icg_slice.min()

            window = np.ones(shape=(len(icg_slice),))
            window *= alpha

            window_slope = np.linspace(alpha + height, 0, num=len(icg_slice_window) + 1, endpoint=True)
            window[idx_stop - (idx_stop - idx_start) : idx_stop + 1] = window_slope

            icg_slice = icg_slice * window

            # peak detection on the transformed signal with minimal peak distance of 50ms and a height threshold of the
            # maximum value divided by 2000
            peaks, height = find_peaks(icg_slice, distance=int(0.05 * sampling_rate_hz), height=icg_slice.max() / 2000)

            if len(peaks) == 1:
                # get the closest zero crossing *before* the C-point
                zero_crossings_diff = zero_crossings - c_point
                zero_crossings_diff = zero_crossings_diff[zero_crossings_diff < 0]
                b_point = zero_crossings[np.argmax(zero_crossings_diff)]

            else:
                # get the two closest peaks to the C-point
                peaks = peaks[-2:]
                # define the b_point as the minimum between the two highest peaks
                search_window = icg_slice[peaks[0] : peaks[-1]]
                b_point = start_window + np.argmin(search_window) + peaks[0]

            b_points.loc[idx, "b_point_sample"] = b_point

        idx_nan = b_points["b_point_sample"].isna()
        if idx_nan.sum() > 0:
            idx_nan = list(b_points.index[idx_nan])

            missing_str = (
                f"The C-point contains NaN at heartbeats {idx_nan}! The index of the B-points were also set to NaN."
            )
            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)

        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(b_points)

        self.points_ = b_points
        return self
