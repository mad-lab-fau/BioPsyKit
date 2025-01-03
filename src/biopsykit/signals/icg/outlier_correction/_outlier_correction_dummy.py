from typing import Optional

import pandas as pd

from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection

__all__ = ["OutlierCorrectionDummy"]

from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype


class OutlierCorrectionDummy(BaseOutlierCorrection):
    """Dummy class for outlier correction. Does nothing and passes through the input data unchanged."""

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: Optional[pd.DataFrame],  # noqa: ARG002
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
        _assert_is_dtype(b_points, pd.DataFrame)
        _assert_has_columns(b_points, [["b_point_sample", "nan_reason"]])
        b_points = b_points.astype({"b_point_sample": "Int64", "nan_reason": "object"})
        assert_sample_columns_int(b_points)

        self.points_ = b_points
        return self
