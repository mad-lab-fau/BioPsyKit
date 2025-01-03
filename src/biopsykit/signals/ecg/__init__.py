"""Module for ECG data analysis and visualization."""
from biopsykit.signals.ecg import event_extraction, plotting, preprocessing, segmentation
from biopsykit.signals.ecg.ecg import EcgProcessor

__all__ = ["EcgProcessor", "event_extraction", "plotting", "preprocessing", "segmentation"]
