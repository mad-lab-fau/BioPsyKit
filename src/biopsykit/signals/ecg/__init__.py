"""Module for ECG data analysis and visualization."""
from biopsykit.signals.ecg.ecg import EcgProcessor, normalize_heart_rate
import biopsykit.signals.ecg.plotting as plotting

__all__ = ["EcgProcessor", "normalize_heart_rate", "plotting"]
