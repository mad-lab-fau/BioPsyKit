"""Module containing different I/O functions to load and save sleep data."""
from typing import Union, Dict

import pandas as pd

from biopsykit.utils._types import path_t


def save_sleep_endpoints(file_path: path_t, df_or_dict: Union[pd.DataFrame, Dict]) -> None:
    """Save sleep endpoints as csv or json file.

    Parameters
    ----------
    file_path: :any:`pathlib.Path` or str
        file path to export
    df_or_dict : :class:`~pandas.DataFrame` or dict
         dataframe or dict with sleep endpoints to export

    """
    if isinstance(df_or_dict, pd.DataFrame):
        df_or_dict.to_csv(file_path)
    else:
        # TODO save dict as json
        raise NotImplementedError(
            "Exporting sleep endpoint dictionary not implemented yet! Consider importing sleep endpoints as dataframe."
        )
