from typing import Optional

import pandas as pd
import numpy as np

from biopsykit.signals._base_extraction import EXTRACTION_HANDLING_BEHAVIOR
from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseOutlierCorrection

from biopsykit.utils._datatype_validation_helper import _assert_is_dtype, _assert_has_columns

__all__ = ["OutlierCorrectionInterpolation"]


# TODO add verbosity option


class OutlierCorrectionInterpolation(BaseOutlierCorrection):
    """algorithm to correct outliers based on Linear Interpolation"""

    # @make_action_safe
    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        """function which corrects outliers of given B-Point dataframe

        Args:
            b_points:
                pd.DataFrame containing the extracted B-Points per heartbeat, index functions as id of heartbeat
            c-points:
                pd.DataFrame containing the extracted C-Points per heartbeat, index functions as id of heartbeat
            sampling_rate_hz:
                sampling rate of ICG signal in hz

        Returns:
            saves resulting corrected B-point locations (samples) in points_ attribute of super class,
            index is B-point (/heartbeat) id
        """
        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point_sample"])

        # stationarize the B-Point time data
        stationary_data = self.stationarize_b_points(b_points, c_points, sampling_rate_hz)

        # detect outliers
        outliers = self.detect_b_point_outlier(stationary_data)
        print(f"Detected {len(outliers)} outliers in correction cycle 1!")
        if len(outliers) == 0:
            _assert_is_dtype(corrected_b_points, pd.DataFrame)
            _assert_has_columns(corrected_b_points, [["b_point_sample"]])
            return self

        # Perform the outlier correction until no more outliers are detected
        counter = 2
        while len(outliers) > 0:
            if counter > 200:
                break
            corrected_b_points = self._correct_outlier_linear_interpolation(
                b_points, c_points, stationary_data, outliers, stationary_data["baseline"], sampling_rate_hz
            )
            # print(corrected_b_points)
            stationary_data = self.stationarize_b_points(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_b_point_outlier(stationary_data)
            print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        print(f"No more outliers got detected!")

        _assert_is_dtype(corrected_b_points, pd.DataFrame)
        _assert_has_columns(corrected_b_points, [["b_point_sample"]])

        self.points_ = corrected_b_points
        return self

    @staticmethod
    def _correct_outlier_linear_interpolation(
        b_points_uncorrected: pd.DataFrame,
        c_points: pd.DataFrame,
        statio_data: pd.DataFrame,
        outliers: pd.DataFrame,
        baseline: pd.DataFrame,
        sampling_rate_hz: int,
    ) -> pd.DataFrame:
        data = statio_data["statio_data"].to_frame()
        # insert NaN at the heartbeat id of the outliers
        data.loc[outliers.index, "statio_data"] = np.NaN

        # interpolate the outlier positions using linear interpolation
        data_interpol = data["statio_data"].astype(float).interpolate()

        corrected_b_points = b_points_uncorrected.copy()
        # Add the baseline back to the interpolated values
        corrected_b_points.loc[data.index, "b_point_sample"] = (
            ((c_points["c_point_sample"][c_points.index[data.index]] - (data_interpol + baseline) * sampling_rate_hz))
            .fillna(0)
            .astype(int)
        )
        corrected_b_points["b_point_sample"] = corrected_b_points["b_point_sample"].replace(0, np.nan)
        return corrected_b_points
