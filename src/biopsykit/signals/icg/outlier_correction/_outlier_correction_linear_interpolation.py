import numpy as np
import pandas as pd

from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseOutlierCorrection
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype

__all__ = ["OutlierCorrectionLinearInterpolation"]


# TODO add verbosity option


class OutlierCorrectionLinearInterpolation(BaseOutlierCorrection):
    """algorithm to correct outliers based on Linear Interpolation."""

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: float,
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
        sampling_rate_hz : float
            Sampling rate of ICG signal in hz
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
        b_points_nan = b_points.loc[b_points["b_point_sample"].isna()]
        stationary_data.loc[b_points_nan.index, "statio_data"] = np.NaN

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
        _assert_has_columns(corrected_b_points, [["b_point_sample", "nan_reason"]])
        corrected_b_points = corrected_b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        assert_sample_columns_int(corrected_b_points)

        self.points_ = corrected_b_points
        return self

    @staticmethod
    def _correct_outlier_linear_interpolation(
        b_points_uncorrected: pd.DataFrame,
        c_points: pd.DataFrame,
        statio_data: pd.DataFrame,
        outliers: pd.DataFrame,
        baseline: pd.DataFrame,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        data = statio_data["statio_data"].to_frame()

        # insert NaN at the heartbeat id of the outliers
        data.loc[outliers.index, "statio_data"] = np.NaN

        # interpolate the outlier positions using linear interpolation
        data_interpol = (
            data["statio_data"]
            .astype(float)
            .interpolate(method="linear")
            .interpolate(method="ffill")  # make sure that the first values are not NaN
            .interpolate(method="bfill")  # make sure that the last values are not NaN
        )

        corrected_b_points = b_points_uncorrected.copy()

        # Add the baseline back to the interpolated values
        corrected_b_points.loc[data.index, "b_point_sample"] = (
            (c_points["c_point_sample"][c_points.loc[data.index].index] - (data_interpol + baseline) * sampling_rate_hz)
            .fillna(0)
            .astype(int)
        )
        corrected_b_points["b_point_sample"] = corrected_b_points["b_point_sample"].replace(0, np.nan)
        return corrected_b_points
