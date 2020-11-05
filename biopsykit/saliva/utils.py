from typing import Optional, Dict, Sequence

import pandas as pd
import numpy as np


def extract_saliva_columns(data: pd.DataFrame, var: str, col_pattern: Optional[str] = None):
    # TODO documentation
    if col_pattern is None:
        col_suggs = get_saliva_column_suggestions(data, var)
        if len(col_suggs) > 1:
            raise KeyError(
                "More than one possible column pattern was found! Please check manually which pattern is correct: {}".format(
                    col_suggs))
        else:
            col_pattern = col_suggs[0]
    return data.filter(regex=col_pattern)


def get_saliva_column_suggestions(data: pd.DataFrame, var: str) -> Sequence[str]:
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


def convert_saliva_wide_to_long(data: pd.DataFrame, feature_name: str, col_pattern: str,
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
