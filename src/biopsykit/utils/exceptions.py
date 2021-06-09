"""A set of custom exceptions."""

__all__ = [
    "ValidationError",
    "FileExtensionError",
    "DataFrameTransformationError",
    "ValueRangeError",
    "FeatureComputationError",
]


class ValidationError(Exception):
    """An error indicating that data-object does not comply with the guidelines."""


class FileExtensionError(Exception):
    """An error indicating that the file name has the wrong file extension."""


class DataFrameTransformationError(Exception):
    """An error indicating that dataframe transformation failed."""


class ValueRangeError(Exception):
    """An error indicating that scores are not in the expected range."""


class FeatureComputationError(Exception):
    """An error indicating that feature computation failed."""
