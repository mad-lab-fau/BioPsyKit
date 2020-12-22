import warnings
from pathlib import Path
from typing import Optional, Sequence, Dict

import pandas as pd
import numpy as np
import biopsykit.utils as utils

from biopsykit.utils import path_t


def load_saliva_plate(file_path: path_t, sample_id_col: str, data_col: str, biomarker_type: str,
                      saliva_times: Optional[Sequence[int]] = None,
                      condition_list: Optional[pd.Index] = None,
                      sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Reads saliva from an Excel sheet in 'plate' format.

    Parameters
    ----------
    file_path: str or path
        path to the Excel sheet in 'plate' format containing saliva data
    sample_id_col: str
        column name of the Excel sheet containing the sample ID
    data_col: str
        column name of the Excel sheet containing data to be analyzed
    biomarker_type: str
        type of the used biomarker
    saliva_times: list or int, optional
        times at which saliva samples were collected
    condition_list: 1d-array, optional
        list of conditions which are assigned to ID
    sheet_name: str, optional
        name of the excel sheet
    Returns
    -------
    pd.DataFrame
        dataframe with saliva parameters
    """
    # TODO add remove_nan option (all or any)

    # ensure pathlib
    file_path = Path(file_path)

    if sheet_name is None:
        df_saliva = pd.read_excel(file_path, skiprows=2, usecols=[sample_id_col, data_col])
    else:
        df_saliva = pd.read_excel(file_path, skiprows=2, sheet_name=sheet_name, usecols=[sample_id_col, data_col])

    df_saliva[["subject", "sample"]] = df_saliva[sample_id_col].str.extract("Vp(\d+) S(\d)")
    df_saliva["sample"] = df_saliva["sample"].astype(int)

    df_saliva.drop(columns=[sample_id_col], inplace=True, errors='ignore')
    df_saliva.rename(columns={data_col: biomarker_type}, inplace=True)
    df_saliva.set_index(["subject", "sample"], inplace=True)

    if condition_list is not None:
        if isinstance(condition_list, Sequence):
            # Add Condition as new index level
            df_saliva = df_saliva.set_index(
                condition_list.repeat(len(df_saliva.index.get_level_values('sample').unique())),
                append=True).reorder_levels(["subject", "condition", "sample"])
        elif isinstance(condition_list, Dict):
            df_cond = pd.DataFrame(condition_list)
            df_cond = df_cond.stack().reset_index(level=1).set_index('level_1').sort_values(0)
            df_cond.index.name = 'condition'
            df_saliva = df_saliva.set_index(
                df_cond.index.repeat(len(df_saliva.index.get_level_values('sample').unique())),
                append=True).reorder_levels(["subject", "condition", "sample"])
        elif isinstance(condition_list, pd.DataFrame):
            condition_list = condition_list.reset_index().set_index("condition")
            df_saliva = df_saliva.set_index(
                condition_list.index.repeat(len(df_saliva.index.get_level_values('sample').unique())),
                append=True).reorder_levels(["subject", "condition", "sample"])

    num_subjects = len(df_saliva.index.get_level_values("subject").unique())
    if saliva_times:
        _check_saliva_times(len(df_saliva), num_subjects, saliva_times)
        df_saliva["time"] = np.array(saliva_times * num_subjects)
    return df_saliva


def save_saliva(file_path: path_t, data: pd.DataFrame) -> None:
    if 'time' in data:
        # drop saliva times for export
        data.drop('time', axis=1, inplace=True)
    data = data.unstack()
    data.columns = data.columns.droplevel(level=0)
    data.to_csv(file_path)


def load_saliva(
        file_path: path_t,
        biomarker_type: str,
        subject_col: Optional[str] = "subject",
        condition_col: Optional[str] = 'condition',
        saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    index_cols = [subject_col]
    if condition_col:
        index_cols.append(condition_col)
    data = pd.read_csv(file_path, dtype={subject_col: str})
    data.set_index(index_cols, inplace=True)

    num_subjects = len(data)
    data.columns = data.columns.astype(int)
    data.columns = pd.MultiIndex.from_product([[biomarker_type], data.columns], names=["", "sample"])

    data = data.stack()

    if saliva_times:
        _check_saliva_times(len(data), num_subjects, saliva_times)
        data['time'] = np.array(saliva_times * num_subjects)

    return data


def _check_saliva_times(num_samples: int, num_subjects: int, saliva_times: Sequence[int]):
    if not np.all(np.diff(saliva_times) > 0):
        raise ValueError("`saliva_times` must be increasing!")
    if (len(saliva_times) * num_subjects) != num_samples:
        raise ValueError(
            "Length of `saliva_times` does not match the number of saliva samples! Expected: {}, got: {}".format(
                int(num_samples / num_subjects), len(saliva_times)))

