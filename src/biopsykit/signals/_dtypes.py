import pandas as pd

from biopsykit.utils.exceptions import ValidationError

__all__ = ["assert_sample_columns_int"]


def assert_sample_columns_int(data: pd.DataFrame) -> None:
    """Assert that the columns of a DataFrame that have "_sample" in their name are of type int."""
    if not any(data.columns.str.contains("_sample")):
        raise ValidationError("DataFrame does not contain any columns with '_sample' in their name!")
    for col in data.columns:
        if "_sample" in col and not pd.api.types.is_integer_dtype(data[col]):
            raise ValidationError(f"Column '{col}' is not of type 'int'!")
