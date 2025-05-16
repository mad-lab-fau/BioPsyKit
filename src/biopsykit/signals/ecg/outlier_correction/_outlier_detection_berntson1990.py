import numpy as np
import pandas as pd
from scipy.stats import iqr

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection

__all__ = ["RPeakOutlierDetectionBerntson1990"]


class RPeakOutlierDetectionBerntson1990(BaseRPeakOutlierDetection):
    """R-peak outlier detection algorithm by Berntson et al. (1990) [Ber90]_.

    This algorithm detects outliers in R-peak data using the method described by Berntson et al. (1990).

    """

    def __init__(self) -> None:
        """Initialize new ``RPeakOutlierDetectionBerntson1990`` algorithm instance."""
        super().__init__()

    def detect_outlier(self, *, ecg: pd.DataFrame, rpeaks: pd.DataFrame, sampling_rate_hz: float):  # noqa: ARG002
        # QD = Quartile Deviation = IQR / 2
        qd = iqr(rpeaks["rr_interval_ms"], nan_policy="omit") / 2.0
        # MAD = Minimal Artifact Difference
        mad = (rpeaks["rr_interval_ms"].median() - 2.9 * qd) / 3.0
        # MED = Maximum Expected Difference
        med = 3.32 * qd
        criterion = np.mean([mad, med])

        outlier = np.abs(rpeaks["rr_interval_ms"] - rpeaks["rr_interval_ms"].median()) > criterion
        outlier = outlier.to_frame(name="is_outlier")

        self.points_ = outlier
