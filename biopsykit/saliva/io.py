import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import numpy as np

from biopsykit.utils import path_t


def load_saliva_plate(file_path: path_t, sample_col: str, data_col: str, feature_name: str,
                      saliva_times: Optional[Sequence[int]] = None,
                      condition_list: Optional[pd.Index] = None,
                      excluded_subjects: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Reads saliva from an Excel sheet in 'plate' format."""
    # TODO documentation
    # TODO add remove_nan option (all or any)

    # ensure pathlib
    file_path = Path(file_path)

    df_saliva = pd.read_excel(file_path, sheet_name="Sheet1", skiprows=2, usecols=[sample_col, data_col])

    df_saliva[["subject", "sample"]] = df_saliva['sample ID'].str.extract("Vp(\d+) S(\d)").astype(int)

    df_saliva.drop(columns=[sample_col], inplace=True, errors='ignore')
    df_saliva.rename(columns={data_col: feature_name}, inplace=True)
    df_saliva.set_index(["subject", "sample"], inplace=True)
    # Subject Exclusion
    if excluded_subjects:
        try:
            df_saliva.drop(index=excluded_subjects, inplace=True)
        except KeyError:
            warnings.warn("Not all subjects of {} exist in the dataset!".format(excluded_subjects))

    if condition_list:
        # Add Condition as new index level
        df_saliva = df_saliva.set_index(condition_list.repeat(len(df_saliva.index.get_level_values('sample').unique())),
                                        append=True).reorder_levels(["condition", "subject", "sample"])
    num_subjects = len(df_saliva.index.get_level_values("subject").unique())
    if saliva_times:
        df_saliva["time"] = np.array(saliva_times * num_subjects)
    return df_saliva
