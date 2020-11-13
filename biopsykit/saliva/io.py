import warnings
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import numpy as np

from biopsykit.utils import path_t


def load_saliva_plate(file_path: path_t, sample_id_col: str, data_col: str, biomarker_type: str,
                      saliva_times: Optional[Sequence[int]] = None,
                      condition_list: Optional[pd.Index] = None,
                      sheet_name: Optional[str] = None,
                      excluded_subjects: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Reads saliva from an Excel sheet in 'plate' format."""
    # TODO documentation
    # TODO add remove_nan option (all or any)

    # ensure pathlib
    file_path = Path(file_path)

    if sheet_name is None:
        df_saliva = pd.read_excel(file_path, skiprows=2, usecols=[sample_id_col, data_col])
    else:
        df_saliva = pd.read_excel(file_path, skiprows=2, sheet_name=sheet_name, usecols=[sample_id_col, data_col])

    df_saliva[["subject", "sample"]] = df_saliva[sample_id_col].str.extract("Vp(\d+) S(\d)").astype(int)

    df_saliva.drop(columns=[sample_id_col], inplace=True, errors='ignore')
    df_saliva.rename(columns={data_col: biomarker_type}, inplace=True)
    df_saliva.set_index(["subject", "sample"], inplace=True)
    # Subject Exclusion
    # TODO move subject exclusion to extra function
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


def save_saliva_biopsykit(file_path: path_t, data: pd.DataFrame) -> None:
    if 'time' in data:
        # drop saliva times for export
        data.drop('time', axis=1, inplace=True)
    data = data.unstack()
    data.columns = data.columns.droplevel(level=0)
    data.to_csv(file_path)


def load_saliva_biopsykit(
        file_path: path_t,
        biomarker_type: str,
        subject_col: Optional[str] = "subject",
        saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    data = pd.read_csv(file_path, index_col=subject_col)

    num_subjects = len(data)
    data.columns = pd.MultiIndex.from_product([[biomarker_type], data.columns])
    data = data.stack()

    if saliva_times:
        _check_saliva_times(len(data), num_subjects, saliva_times)
        data['time'] = np.array(saliva_times * num_subjects)

    return data
