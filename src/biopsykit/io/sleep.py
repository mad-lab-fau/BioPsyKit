"""Module containing different I/O functions to load and save sleep data."""
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import is_sleep_endpoint_dataframe, is_sleep_endpoint_dict

__all__ = ["save_sleep_endpoints"]


def save_sleep_endpoints(file_path: path_t, df_or_dict: Union[pd.DataFrame, Dict]):
    """Save sleep endpoints as csv or json file.

    Parameters
    ----------
    file_path: :class:`~pathlib.Path` or str
        file path to export
    df_or_dict : :class:`~pandas.DataFrame` or dict
         dataframe or dict with sleep endpoints to export

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if ``df_or_dict`` is a dataframe and ``file_path`` is not a csv file

    """
    file_path = Path(file_path)
    if isinstance(df_or_dict, pd.DataFrame):
        _assert_file_extension(file_path, ".csv")
        is_sleep_endpoint_dataframe(df_or_dict)
        df_or_dict.to_csv(file_path)
    else:
        is_sleep_endpoint_dict(df_or_dict)
        # TODO save dict as json
        raise NotImplementedError(
            "Exporting sleep endpoint dictionary not implemented yet! Consider importing sleep endpoints as dataframe."
        )
