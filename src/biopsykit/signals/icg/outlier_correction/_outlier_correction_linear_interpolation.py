import numpy as np
import pandas as pd

from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseBPointOutlierCorrection

__all__ = ["OutlierCorrectionLinearInterpolation"]


from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame, is_b_point_dataframe, is_c_point_dataframe

# TODO add verbosity option


class OutlierCorrectionLinearInterpolation(BaseBPointOutlierCorrection):
    """B-point outlier correction algorithm based on linear interpolation.

    This algorithm corrects outliers in B-Point data using linear interpolation.

    """

    def correct_outlier(
        self,
        *,
        b_points: BPointDataFrame,
        c_points: CPointDataFrame,
        sampling_rate_hz: float,
        **kwargs,
    ):
        """Perform outlier correction on B-Point data.

        The outliers are corrected using linear interpolation.

        The results of the outlier correction are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        b_points : :class:`~pandas.DataFrame`
            Extracted B-points. Each row contains the B-point location (in samples from beginning of signal) for each
            heartbeat, index functions as id of heartbeat. B-point locations can be NaN if no B-points were detected
            for certain heartbeats.
        c_points : :class:`~pandas.DataFrame`
            Extracted C-points. Each row contains the C-point location (in samples from beginning of signal) for each
            heartbeat, index functions as id of heartbeat. C-point locations can be NaN if no C-points were detected
            for certain heartbeats.
        sampling_rate_hz : int
            sampling rate of ICG derivative signal in hz
        kwargs : dict
            Additional keyword arguments:
                * verbose: bool, optional
                    Whether to print additional information. Default: False

        Returns
        -------
        self

        """
        verbose = kwargs.get("verbose", False)
        is_b_point_dataframe(b_points)
        is_c_point_dataframe(c_points)

        corrected_b_points = pd.DataFrame(index=b_points.index, columns=["b_point_sample"])
        # stationarize the B-Point time data
        stationary_data = self.stationarize_b_points(b_points, c_points, sampling_rate_hz)
        b_points_nan = b_points.loc[b_points["b_point_sample"].isna()]
        stationary_data.loc[b_points_nan.index, "statio_data"] = np.nan

        # detect outliers
        outliers = self.detect_b_point_outlier(stationary_data)

        counter = 1
        if verbose:
            print(f"Detected {len(outliers)} outliers in correction cycle {counter}!")

        if len(outliers) == 0:
            is_b_point_dataframe(b_points)
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

        corrected_b_points = corrected_b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        is_b_point_dataframe(corrected_b_points)

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
        data.loc[outliers.index, "statio_data"] = np.nan

        # interpolate the outlier positions using linear interpolation
        data_interpol = (
            data["statio_data"]
            .astype(float)
            .interpolate(method="linear")
            .ffill()  # make sure that the first values are not NaN
            .bfill()  # make sure that the last values are not NaN
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
