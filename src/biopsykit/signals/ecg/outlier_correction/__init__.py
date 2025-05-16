"""Outlier correction for R-peaks in ECG signals."""

from biopsykit.signals.ecg.outlier_correction._base_outlier_detection import BaseRPeakOutlierDetection
from biopsykit.signals.ecg.outlier_correction._outlier_correction import RPeakOutlierCorrection
from biopsykit.signals.ecg.outlier_correction._outlier_correction_hrv_lipponen2019 import (
    RPeakOutlierCorrectionHrvLipponen2019,
)
from biopsykit.signals.ecg.outlier_correction._outlier_detection_berntson1990 import RPeakOutlierDetectionBerntson1990
from biopsykit.signals.ecg.outlier_correction._outlier_detection_correlation import RPeakOutlierDetectionCorrelation
from biopsykit.signals.ecg.outlier_correction._outlier_detection_physiological import RPeakOutlierDetectionPhysiological
from biopsykit.signals.ecg.outlier_correction._outlier_detection_quality import RPeakOutlierDetectionQuality
from biopsykit.signals.ecg.outlier_correction._outlier_detection_statistical_rr import (
    RPeakOutlierDetectionRRIntervalStatistics,
)
from biopsykit.signals.ecg.outlier_correction._outlier_detection_statistical_rr_diff import (
    RPeakOutlierDetectionRRDiffIntervalStatistics,
)

__all__ = [
    "BaseRPeakOutlierDetection",
    "RPeakOutlierCorrection",
    "RPeakOutlierCorrectionHrvLipponen2019",
    "RPeakOutlierDetectionBerntson1990",
    "RPeakOutlierDetectionCorrelation",
    "RPeakOutlierDetectionPhysiological",
    "RPeakOutlierDetectionQuality",
    "RPeakOutlierDetectionRRDiffIntervalStatistics",
    "RPeakOutlierDetectionRRIntervalStatistics",
]
