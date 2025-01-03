"""Module for ECG segmentation."""
from biopsykit.signals.ecg.segmentation._base_segmentation import BaseHeartbeatSegmentation
from biopsykit.signals.ecg.segmentation._heartbeat_segmentation import HeartbeatSegmentationNeurokit

__all__ = ["BaseHeartbeatSegmentation", "HeartbeatSegmentationNeurokit"]
