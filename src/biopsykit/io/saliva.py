from pathlib import Path
from typing import Optional, Sequence, Dict

import pandas as pd
import numpy as np

from biopsykit.utils._types import path_t

_DATA_COL_NAMES = {"cortisol": "cortisol (nmol/l)", "amylase": "amylase (U/ml)"}


def load_saliva_plate(
    file_path: path_t,
    biomarker_type: str,
    sample_id_col: Optional[str] = "sample ID",
    data_col: Optional[str] = None,
    data_col_names: Optional[Sequence[str]] = None,
    regex_str: Optional[str] = None,
    saliva_times: Optional[Sequence[int]] = None,
    condition_list: Optional[pd.Index] = None,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    r"""
    Reads saliva from an Excel sheet in 'plate' format.

    To extract the subject ID, day ID and sample ID from the saliva sample identifier, you can pass a
    regular expression string. Here are some examples on how sample identifier might look like and what the
    corresponding ``regex_str`` would output:

    * "Vp01 S1" => ``r"(Vp\d+) (S\d)"`` (this is the default pattern, you can also just set ``regex_str`` to ``None``) => data ``[Vp01, S1]`` in two columns: ``'subject'``, ``'sample'`` (unless column names are explicitely specified in ``data_col_names``)
    * "Vp01 T1 S1" ... "Vp01 T1 S5" (only *numeric* characters in day/sample) => ``r"(Vp\d+) (T\d) (S\d)"`` => three columns: ``'subject'``, ``'sample'`` with data ``[Vp01, T1, S1]`` (unless column names are explicitely specified in ``data_col_names``)
    * "Vp01 T1 S1" ... "Vp01 T1 SA" (also *letter* characters in day/sample) => ``r"(Vp\d+) (T\w) (S\w)"`` => three columns: ``'subject'``, ``'sample'`` with data ``[Vp01, T1, S1]`` (unless column names are explicitely specified in ``data_col_names``)

    If you **don't** want to extract the 'S' or 'T' prefixed in saliva or day IDs, respectively, you have to move it **out** of the capture group in the ``regex_str`` (round brackets), like this: ``(S\d)`` (would give ``S1, S2, ...``) => ``S(\d)`` (would give ``1, 2, ...``)


    Parameters
    ----------
    file_path: str or path
        path to the Excel sheet in 'plate' format containing saliva data
    biomarker_type: str
        type of the used biomarker
    sample_id_col: str, optional
        column name of the Excel sheet containing the sample ID. Default: "sample ID"
    data_col: str, optional
        column name of the Excel sheet containing data to be analyzed. Default: Select default column name based on ``biomarker_type``
    data_col_names: list of str, optional
        column names of the extracted columns. ``None`` to use the default column names (['subject', 'day', 'sample'])
    regex_str: str, optional
        regular expression to extract subject ID, day ID and sample ID from the sample identifier. ``None`` to use default regex string
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

    if regex_str is None:
        regex_str = r"(Vp\d+) (S\w)"

    # ensure pathlib
    file_path = Path(file_path)

    if data_col is None:
        data_col = _DATA_COL_NAMES[biomarker_type]

    if sheet_name is None:
        df_saliva = pd.read_excel(file_path, skiprows=2, usecols=[sample_id_col, data_col])
    else:
        df_saliva = pd.read_excel(
            file_path,
            skiprows=2,
            sheet_name=sheet_name,
            usecols=[sample_id_col, data_col],
        )

    cols = df_saliva[sample_id_col].str.extract(regex_str)
    if data_col_names is None:
        if len(cols.columns) == 2:
            data_col_names = ["subject", "sample"]
        elif len(cols.columns) == 3:
            data_col_names = ["subject", "day", "sample"]

    df_saliva[data_col_names] = cols
    # df_saliva["sample"] = df_saliva["sample"].astype(int)

    df_saliva.drop(columns=[sample_id_col], inplace=True, errors="ignore")
    df_saliva.rename(columns={data_col: biomarker_type}, inplace=True)
    df_saliva.set_index(data_col_names, inplace=True)

    if condition_list is not None:
        if isinstance(condition_list, Sequence):
            # Add Condition as new index level
            condition_list = pd.DataFrame(
                condition_list,
                columns=["condition"],
                index=df_saliva.index.get_level_values("subject").unique(),
            )
        elif isinstance(condition_list, Dict):
            condition_list = pd.DataFrame(condition_list)
            condition_list = condition_list.stack().reset_index(level=1).set_index("level_1").sort_values(0)
            condition_list.index.name = "condition"
        elif isinstance(condition_list, pd.DataFrame):
            condition_list = condition_list.reset_index().set_index("subject")

        df_saliva = (
            df_saliva.join(condition_list)
            .set_index("condition", append=True)
            .reorder_levels(["condition", "subject", "sample"])
        )

    num_subjects = len(df_saliva.index.get_level_values("subject").unique())

    _check_num_samples(len(df_saliva), num_subjects)

    if saliva_times:
        _check_saliva_times(len(df_saliva), num_subjects, saliva_times)
        df_saliva["time"] = np.array(saliva_times * num_subjects)

    try:
        df_saliva[biomarker_type] = df_saliva[biomarker_type].astype(float)
    except ValueError as e:
        raise ValueError(
            """Error converting  all saliva values into numbers: '{}'
            Please check your saliva values whether there is any text etc. in the column '{}' and delete the values or replace them by NaN!""".format(
                e, data_col
            )
        )
    return df_saliva


def save_saliva(file_path: path_t, data: pd.DataFrame) -> None:
    if "time" in data:
        # drop saliva times for export
        data.drop("time", axis=1, inplace=True)
    data = data.unstack()
    data.columns = data.columns.droplevel(level=0)
    data.to_csv(file_path)


def load_saliva_wide_format(
    file_path: path_t,
    biomarker_name: str,
    subject_col: Optional[str] = "subject",
    condition_col: Optional[str] = None,
    saliva_times: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    index_cols = [subject_col]

    data = pd.read_csv(file_path, dtype={subject_col: str})

    if condition_col is None:
        if "condition" in data.columns:
            condition_col = "condition"

    if condition_col is not None:
        index_cols = [condition_col] + index_cols

    data.set_index(index_cols, inplace=True)

    num_subjects = len(data)
    data.columns = pd.MultiIndex.from_product([[biomarker_name], data.columns], names=["", "sample"])

    data = data.stack()

    _check_num_samples(len(data), num_subjects)

    if saliva_times is not None:
        _check_saliva_times(len(data), num_subjects, saliva_times)
        data["time"] = np.array(saliva_times * num_subjects)

    return data


def _check_num_samples(num_samples: int, num_subjects: int):
    if num_samples % num_subjects != 0:
        raise ValueError(
            "Error during import: Number of samples not equal for all subjects! Got {} samples for {} subjects.".format(
                num_samples, num_subjects
            )
        )


def _check_saliva_times(num_samples: int, num_subjects: int, saliva_times: Sequence[int]):
    if not np.all(np.diff(saliva_times) > 0):
        raise ValueError("`saliva_times` must be increasing!")
    if (len(saliva_times) * num_subjects) != num_samples:
        raise ValueError(
            "Length of `saliva_times` does not match the number of saliva samples! Expected: {}, got: {}".format(
                int(num_samples / num_subjects), len(saliva_times)
            )
        )
