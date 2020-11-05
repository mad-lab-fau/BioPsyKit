import warnings
from pathlib import Path
from typing import Optional, Dict, Sequence

import pandas as pd
import numpy as np

from biopsykit.utils import path_t


def read_saliva_plate(file_path: path_t, sample_col: str, data_col: str, feature_name: str,
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


def saliva_extract_columns(data: pd.DataFrame, var: str, col_pattern: Optional[str] = None):
    # TODO documentation
    if col_pattern is None:
        col_suggs = saliva_get_column_suggestions(data, var)
        if len(col_suggs) > 1:
            raise KeyError(
                "More than one possible column pattern was found! Please check manually which pattern is correct: {}".format(
                    col_suggs))
        else:
            col_pattern = col_suggs[0]
    return data.filter(regex=col_pattern)


def saliva_get_column_suggestions(data: pd.DataFrame, var: str) -> Sequence[str]:
    # TODO documentation
    import re

    sugg_filt = list(filter(lambda col: any(k in col for k in _dict_words[var]), data.columns))
    sugg_filt = list(filter(lambda s: any(str(i) in s for i in range(0, 20)), sugg_filt))
    sugg_filt = list(
        filter(
            lambda s: all(
                k not in s for k in ("AUC", "auc", "TSST", "max", "log", "inc", "lg", "ln", "GenExp", "inv")),
            sugg_filt
        )
    )
    # replace il{} with il6 since this was removed out by the previous filter operation
    sugg_filt = [re.sub("\d", '{}', s).replace("il{}", "il6").replace("IL{}", "IL6") for s in sugg_filt]
    sugg_filt = sorted(list(filter(lambda s: "{}" in s, set(sugg_filt))))

    # build regex for column extraction
    sugg_filt = ['^{}$'.format(s.replace("{}", "(\d)")) for s in sugg_filt]
    return sugg_filt


def saliva_wide_to_long(data: pd.DataFrame, feature_name: str, col_pattern: str,
                        saliva_times: Optional[Sequence[int]] = None) -> pd.DataFrame:
    data = data.copy()
    data.index.name = "subject"
    df_day_sample = data.columns.str.extract(col_pattern)
    df_day_sample = df_day_sample.astype(int)
    if len(df_day_sample.columns) > 1:
        # we have multi-day recordings => create MultiIndex
        data.columns = pd.MultiIndex.from_arrays(df_day_sample.T.values, names=["day", "sample"])
        var_name = ["day", "sample"]
    else:
        data.columns = df_day_sample.values
        var_name = "sample"

    df_long = pd.melt(data.reset_index(), id_vars=['subject'], value_name=feature_name, var_name=var_name)
    df_long.set_index('subject', inplace=True)
    df_long.set_index(var_name, append=True, inplace=True)
    df_long.sort_index(inplace=True)

    if saliva_times:
        df_long["time"] = np.array(saliva_times * int(len(df_long) / len(saliva_times)))
    return df_long


_dict_words: Dict[str, Sequence[str]] = {
    'cortisol': ['cortisol', 'cort', 'Cortisol', '_c_'],
    'amylase': ['amylase', 'amy', 'Amylase', 'sAA'],
    'il6': ['il6', 'IL6', 'il-6', 'IL-6', "il_6", "IL_6"]
}
