"""Module for outlier correction of ICG signals."""
from biopsykit.signals.icg.outlier_correction._base_outlier_correction import BaseOutlierCorrection
from biopsykit.signals.icg.outlier_correction._outlier_correction_dummy import OutlierCorrectionDummy
from biopsykit.signals.icg.outlier_correction._outlier_correction_forouzanfar2018 import (
    OutlierCorrectionForouzanfar2018,
)
from biopsykit.signals.icg.outlier_correction._outlier_correction_interpolation import OutlierCorrectionInterpolation

__all__ = [
    "BaseOutlierCorrection",
    "OutlierCorrectionForouzanfar2018",
    "OutlierCorrectionInterpolation",
    "OutlierCorrectionDummy",
]
