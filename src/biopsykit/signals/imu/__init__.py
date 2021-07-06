"""Module for processing IMU data."""
from biopsykit.signals.imu.imu import sliding_windows_imu, var_norm_windows, convert_acc_data_to_g
from biopsykit.signals.imu import (
    activity_counts,
    static_moment_detection,
    wear_detection,
    feature_extraction,
    rest_periods,
)


__all__ = [
    "var_norm_windows",
    "convert_acc_data_to_g",
    "sliding_windows_imu",
    "rest_periods",
    "wear_detection",
    "static_moment_detection",
    "activity_counts",
    "feature_extraction",
]
