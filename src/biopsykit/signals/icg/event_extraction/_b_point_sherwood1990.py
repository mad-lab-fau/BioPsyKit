import warnings
from typing import Optional

import numpy as np
import pandas as pd

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, CanHandleMissingEventsMixin
from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.array_handling import sanitize_input_dataframe_1d
from biopsykit.utils.exceptions import EventExtractionError


class BPointExtractionSherwood1990(BaseBPointExtraction, CanHandleMissingEventsMixin):
    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Initialize new BPointExtractionSherwood1990 algorithm instance."""
        super().__init__(handle_missing_events=handle_missing_events)

    def extract(
        self,
        *,
        icg: pd.Series,
        heartbeats: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: Optional[float],  # noqa: ARG002
    ):
        self._check_valid_missing_handling()
        # sanitize input
        icg = sanitize_input_dataframe_1d(icg, column="icg_der")
        icg = icg.squeeze()

        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])

        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        c_points = c_points["c_point_sample"]
        check_c_points = pd.isna(c_points)

        # get zero crossings of icg
        zero_crossings = np.where(np.diff(np.signbit(icg)))[0]

        # go through each R-C interval independently and search for the local minima
        for idx, data in heartbeats.iterrows():
            # check if r_peaks/c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_c_points[idx]:
                b_points["b_point_sample"].iloc[idx] = np.NaN
                b_points["nan_reason"].iloc[idx] = "c_point_nan"
                missing_str = f"The c_point contains NaN at position {idx}! B-Point was set to NaN."
                if self.handle_missing_events == "warn":
                    warnings.warn(missing_str)
                elif self.handle_missing_events == "raise":
                    raise EventExtractionError(missing_str)
                continue

            # get the closest zero crossing *before* the C-point
            c_point = c_points[idx]
            zero_crossings_diff = zero_crossings - c_point
            zero_crossings_diff = zero_crossings_diff[zero_crossings_diff < 0]
            zero_crossing_idx = np.argmax(zero_crossings_diff)

            b_point = zero_crossings[zero_crossing_idx]
            # assert that b_point is within the R-C interval
            if not (data["r_peak_sample"] < b_point < c_point):
                b_point = np.NaN
                b_points["nan_reason"].iloc[idx] = "no_zero_crossing"

            # Add the detected B-point to the b_points Dataframe
            b_points["b_point_sample"].iloc[idx] = b_point

        _assert_is_dtype(b_points, pd.DataFrame)
        _assert_has_columns(b_points, [["b_point_sample", "nan_reason"]])
        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        assert_sample_columns_int(b_points)

        self.points_ = b_points

        return self
