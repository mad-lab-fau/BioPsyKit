from typing import Optional

from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection

__all__ = ["OutlierCorrectionDummy"]

from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame, is_b_point_dataframe


class OutlierCorrectionDummy(BaseOutlierCorrection):
    """Dummy class for outlier correction. Does nothing and passes through the input data unchanged."""

    def correct_outlier(
        self,
        *,
        b_points: BPointDataFrame,
        c_points: Optional[CPointDataFrame],  # noqa: ARG002
        sampling_rate_hz: Optional[float],  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        """Correct no outliers, just pass through the input data unchanged.

        The results of the outlier correction are saved in the `points_` attribute of the class instance.

        Parameters
        ----------
        b_points : :class:`~pandas.DataFrame`
            Dataframe containing the extracted B-Points per heartbeat, index functions as id of heartbeat
        c_points: :class:`~pandas.DataFrame`
            Dataframe containing the extracted C-Points per heartbeat, index functions as id of heartbeat. Not used.
        sampling_rate_hz : int
            Sampling rate of ICG signal in hz. Not used.
        kwargs: dict
            Additional keyword arguments. Not used.

        Returns
        -------
        self

        """
        is_b_point_dataframe(b_points)
        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})

        self.points_ = b_points
        return self
