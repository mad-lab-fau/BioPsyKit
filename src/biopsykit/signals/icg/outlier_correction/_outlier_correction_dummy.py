from biopsykit.signals.icg.outlier_correction import BaseBPointOutlierCorrection

__all__ = ["OutlierCorrectionDummy"]

from biopsykit.utils.dtypes import BPointDataFrame, CPointDataFrame, is_b_point_dataframe


class OutlierCorrectionDummy(BaseBPointOutlierCorrection):
    """B-point outlier correction algorithm that does nothing and passes through the input data unchanged.

    This class is used as a placeholder for the outlier correction in the ICG pipeline.

    """

    def correct_outlier(
        self,
        *,
        b_points: BPointDataFrame,
        c_points: CPointDataFrame | None,  # noqa: ARG002
        sampling_rate_hz: float | None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        """Perform outlier correction.

        This method does nothing and passes through the input data unchanged.

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
