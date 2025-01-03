"""A set of custom exceptions."""

__all__ = [
    "DataFrameTransformationError",
    "EcgProcessingError",
    "EventExtractionError",
    "FeatureComputationError",
    "FileExtensionError",
    "ValidationError",
    "ValueRangeError",
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


class EcgProcessingError(Exception):
    """An error indicating that ECG processing failed."""


class EventExtractionError(Exception):
    """An error indicating that event extraction failed."""
