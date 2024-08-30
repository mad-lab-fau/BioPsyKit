from typing import Optional

import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection

__all__ = ["OutlierCorrectionDummy"]


class OutlierCorrectionDummy(BaseOutlierCorrection):
    """Dummy class for outlier correction. Does nothing and passes through the input data unchanged."""

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,  # noqa: ARG002
        sampling_rate_hz: int,  # noqa: ARG002
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",  # noqa: ARG002
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
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Not used.
        kwargs: dict
            Additional keyword arguments. Not used.

        Returns
        -------
        self

        """
        self.points_ = b_points.convert_dtypes(infer_objects=True)
        return self
