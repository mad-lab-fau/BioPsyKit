from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseOutlierCorrection
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype

__all__ = ["OutlierCorrectionInterpolation"]


# TODO add verbosity option


class OutlierCorrectionInterpolation(BaseOutlierCorrection):
    """algorithm to correct outliers based on Linear Interpolation."""

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",  # noqa: ARG002
        **kwargs,
    ):
        """Correct outliers of given B-Point dataframe using Linear Interpolation.

        The results of the outlier correction are saved in the `points_` attribute of the class instance.

        Parameters
        ----------
        b_points : :class:`~pandas.DataFrame`
            Dataframe containing the extracted B-Points per heartbeat, index functions as id of heartbeat
        c_points : :class:`~pandas.DataFrame`
            Dataframe containing the extracted C-Points per heartbeat, index functions as id of heartbeat
        sampling_rate_hz : int
            Sampling rate of ICG signal in hz
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Not used in this function.
        **kwargs
            Additional keyword arguments:
                * verbose : bool, optional
                    Whether to print additional information. Default: False

        Returns
        -------
        self


        """
        verbose = kwargs.get("verbose", False)

        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point_sample"])
        # stationarize the B-Point time data
        stationary_data = self.stationarize_b_points(b_points, c_points, sampling_rate_hz)

        # detect outliers
        outliers = self.detect_b_point_outlier(stationary_data)

        counter = 1
        if verbose:
            print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")

        if len(outliers) == 0:
            _assert_is_dtype(b_points, pd.DataFrame)
            _assert_has_columns(b_points, [["b_point_sample", "nan_reason"]])
            self.points_ = b_points
            return self

        # Perform the outlier correction until no more outliers are detected
        while len(outliers) > 0:
            if counter >= 200:
                break
            corrected_b_points = self._correct_outlier_linear_interpolation(
                b_points, c_points, stationary_data, outliers, stationary_data["baseline"], sampling_rate_hz
            )

            stationary_data = self.stationarize_b_points(corrected_b_points, c_points, sampling_rate_hz)
            outliers = self.detect_b_point_outlier(stationary_data)
            if verbose:
                print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")
            counter += 1

        if verbose:
            print("No more outliers got detected!")

        _assert_is_dtype(corrected_b_points, pd.DataFrame)
        _assert_has_columns(corrected_b_points, [["b_point_sample"]])

        self.points_ = corrected_b_points.convert_dtypes(infer_objects=True)
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
            (c_points["c_point_sample"][c_points.index[data.index]] - (data_interpol + baseline) * sampling_rate_hz)
            .fillna(0)
            .astype(int)
        )
        corrected_b_points["b_point_sample"] = corrected_b_points["b_point_sample"].replace(0, np.nan)
        return corrected_b_points