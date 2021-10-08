"""Module to perform sleep/wake detection based on Actigraphy or IMU data."""

from biopsykit.sleep.sleep_wake_detection import algorithms
from biopsykit.sleep.sleep_wake_detection.sleep_wake_detection import SleepWakeDetection

__all__ = [
    "algorithms",
    "SleepWakeDetection",
]
