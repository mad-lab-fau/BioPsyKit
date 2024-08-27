from typing import Optional

import pandas as pd

from biopsykit.signals._base_extraction import EXTRACTION_HANDLING_BEHAVIOR
from biopsykit.signals.icg.outlier_correction import BaseOutlierCorrection


__all__ = ["DummyOutlierCorrection"]


class DummyOutlierCorrection(BaseOutlierCorrection):
    """Dummy class for outlier correction. Does nothing."""

    def correct_outlier(
        self,
        *,
        b_points: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[EXTRACTION_HANDLING_BEHAVIOR] = "warn",
    ):
        self.points_ = b_points.copy()
        return self
