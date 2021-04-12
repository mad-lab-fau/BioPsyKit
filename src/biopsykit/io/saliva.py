"""Module containing different I/O functions for saliva data."""
from pathlib import Path
from typing import Optional, Sequence, Union, Dict

import numpy as np
import pandas as pd
from biopsykit.utils._datatype_validation_helper import _assert_file_extension
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import SalivaRawDataFrame, is_raw_saliva_dataframe

_DATA_COL_NAMES = {"cortisol": "cortisol (nmol/l)", "amylase": "amylase (U/ml)"}


def load_saliva_plate(
    file_path: path_t,
    saliva_type: str,
    sample_id_col: Optional[str] = "sample ID",
    data_col: Optional[str] = None,
    id_col_names: Optional[Sequence[str]] = None,
    regex_str: Optional[str] = None,
    sample_times: Optional[Sequence[int]] = None,
    condition_list: Optional[Union[Sequence, Dict[str, Sequence], pd.Index]] = None,
    sheet_name: Optional[Union[str, int]] = 0,
) -> SalivaRawDataFrame:
    r"""Read saliva from an Excel sheet in 'plate' format.

    This function automatically extracts identifier like subject, day and sample IDs from the saliva sample names.
    To extract them, a regular expression string can be passed via ``regex_str``.

    Here are some examples on how sample identifiers might look like and what the corresponding
    ``regex_str`` would output:
        * "Vp01 S1"
            => ``r"(Vp\d+) (S\d)"`` (this is the default pattern, you can also just set ``regex_str`` to ``None``)
            => data ``[Vp01, S1]`` in two columns: ``subject``, ``sample``
            (unless column names are explicitly specified in ``data_col_names``)
        * "Vp01 T1 S1" ... "Vp01 T1 S5" (only *numeric* characters in day/sample)
            => ``r"(Vp\d+) (T\d) (S\d)"``
            => three columns: ``subject``, ``sample`` with data ``[Vp01, T1, S1]``
            (unless column names are explicitely specified in ``data_col_names``)
        * "Vp01 T1 S1" ... "Vp01 T1 SA" (also *letter* characters in day/sample)
            => ``r"(Vp\d+) (T\w) (S\w)"``
            => three columns: ``subject``, ``sample`` with data ``[Vp01, T1, S1]``
            (unless column names are explicitly specified in ``data_col_names``)

    If you **don't** want to extract the 'S' or 'T' prefixes in saliva or day IDs, respectively,
    you have to move it **out** of the capture group in the ``regex_str`` (round brackets), like this:
    ``(S\d)`` (would give ``S1``, ``S2``, ...)
    => ``S(\d)`` (would give ``1``, ``2``, ...)


    Parameters
    ----------
    file_path: :any:`pathlib.Path` or str
        path to the Excel sheet in 'plate' format containing saliva data
    saliva_type: str
        saliva type to load from file
    sample_id_col: str, optional
        column name of the Excel sheet containing the sample ID. Default: "sample ID"
    data_col: str, optional
        column name of the Excel sheet containing saliva data to be analyzed.
        Default: Select default column name based on ``biomarker_type``, e.g. ``cortisol`` => ``cortisol (nmol/l)``
    id_col_names: list of str, optional
        names of the extracted ID column names. ``None`` to use the default column names (['subject', 'day', 'sample'])
    regex_str: str, optional
        regular expression to extract subject ID, day ID and sample ID from the sample identifier.
        ``None`` to use default regex string (``r"(Vp\d+) (S\d)"``)
    sample_times: list of int, optional
        times at which saliva samples were collected
    condition_list: 1d-array, optional
        list of conditions which subjects were assigned to
    sheet_name: str, optional
        name or index of the Excel sheet. Default: ``0`` to use the first sheet in the file

    Returns
    -------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format

    Raises
    ------
    :class:`~biopsykit.exceptions.FileExtensionError`
        if file is no Excel file (.xls or .xlsx)
    ValueError
        if any saliva sample can not be converted into a float (e.g. because there was text in one of the columns)
    :exc:`biopsykit.exceptions.ValidationError`
        if imported data can not be parsed to a SalivaRawDataFrame

    """
    # TODO add remove_nan option (all or any)
    if regex_str is None:
        regex_str = r"(Vp\d+) (S\w)"

    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, (".xls", ".xlsx"))

    if data_col is None:
        data_col = _DATA_COL_NAMES[saliva_type]

    df_saliva = pd.read_excel(
        file_path,
        skiprows=2,
        sheet_name=sheet_name,
        usecols=[sample_id_col, data_col],
    )

    cols = df_saliva[sample_id_col].str.extract(regex_str)
    if id_col_names is None:
        id_col_names = ["subject", "sample"]
        if len(cols.columns) == 3:
            id_col_names = ["subject", "day", "sample"]

    df_saliva[id_col_names] = cols

    df_saliva.drop(columns=[sample_id_col], inplace=True, errors="ignore")
    df_saliva.rename(columns={data_col: saliva_type}, inplace=True)
    df_saliva.set_index(id_col_names, inplace=True)

    if condition_list is not None:
        condition_list = _parse_condition_list(df_saliva, condition_list)

        df_saliva = (
            df_saliva.join(condition_list)
            .set_index("condition", append=True)
            .reorder_levels(["condition", "subject", "sample"])
        )

    num_subjects = len(df_saliva.index.get_level_values("subject").unique())

    _check_num_samples(len(df_saliva), num_subjects)

    if sample_times:
        _check_sample_times(len(df_saliva), num_subjects, sample_times)
        df_saliva["time"] = np.array(sample_times * num_subjects)

    try:
        df_saliva[saliva_type] = df_saliva[saliva_type].astype(float)
    except ValueError as e:
        raise ValueError(
            """Error converting all saliva values into numbers: '{}'
            Please check your saliva values whether there is any text etc. in the column '{}'
            and delete the values or replace them by NaN!""".format(
                e, data_col
            )
        )

    is_raw_saliva_dataframe(df_saliva, saliva_type)

    return df_saliva


def save_saliva(file_path: path_t, data: SalivaRawDataFrame, saliva_type: Optional[str] = "cortisol") -> None:
    """Save saliva data to csv file.

    Parameters
    ----------
    file_path: :any:`pathlib.Path` or str
        file path to export
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str
        type of saliva data in the dataframe

    Raises
    ------
    :exc:`biopsykit.exceptions.ValidationError`
        if ``data`` is not a SalivaRawDataFrame

    """
    is_raw_saliva_dataframe(data, saliva_type)
    data = data[saliva_type]
    if "time" in data:
        # drop saliva times for export
        data.drop("time", axis=1, inplace=True)
    data = data.unstack()
    data.to_csv(file_path)


def load_saliva_wide_format(
    file_path: path_t,
    saliva_type: str,
    subject_col: Optional[str] = "subject",
    condition_col: Optional[str] = None,
    sample_times: Optional[Sequence[int]] = None,
) -> SalivaRawDataFrame:
    """Load saliva data that is in wide-format from csv file.

    Parameters
    ----------
    file_path: :any:`pathlib.Path` or str
        path to file
    saliva_type: str
        saliva type to load from file
    subject_col: str, optional
        name of the column containing subject IDs. Default: "sample ID"
    condition_col : str, optional
        name of the column containing condition assignments or ``None`` if no conditions are present. Default: ``None``
    sample_times: list of int, optional
        times at which saliva samples were collected

    Returns
    -------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format

    Raises
    ------
    :class:`~biopsykit.exceptions.FileExtensionError`
        if file is no csv file

    """
    index_cols = [subject_col]

    _assert_file_extension(file_path, ".csv")
    data = pd.read_csv(file_path, dtype={subject_col: str})

    if condition_col is None and "condition" in data.columns:
        condition_col = "condition"

    if condition_col is not None:
        index_cols = [condition_col] + index_cols

    data.set_index(index_cols, inplace=True)

    num_subjects = len(data)
    data.columns = pd.MultiIndex.from_product([[saliva_type], data.columns], names=["", "sample"])

    data = data.stack()

    _check_num_samples(len(data), num_subjects)

    if sample_times is not None:
        _check_sample_times(len(data), num_subjects, sample_times)
        data["time"] = np.array(sample_times * num_subjects)

    return data


def _check_num_samples(num_samples: int, num_subjects: int) -> None:
    """Check that number of imported samples is the same for all subjects.

    Parameters
    ----------
    num_samples : int
        total number of saliva samples in the current dataframe
    num_subjects : int
        total number of subjects in the current dataframe

    Raises
    ------
    ValueError
        if number of samples is not equal for all subjects

    """
    if num_samples % num_subjects != 0:
        raise ValueError(
            "Error during import: Number of samples not equal for all subjects! Got {} samples for {} subjects.".format(
                num_samples, num_subjects
            )
        )


def _check_sample_times(num_samples: int, num_subjects: int, sample_times: Sequence[int]) -> None:
    """Check that sample times have the correct number of samples and are monotonously increasing.

    Parameters
    ----------
    num_samples : int
        total number of saliva samples in the current dataframe
    num_subjects : int
        total number of subjects in the current dataframe
    sample_times : array-like
        list of sample times

    Raises
    ------
    ValueError
        if values in ``sample_times`` are not monotonously increasing or
        if number of saliva times does not match the number of saliva samples per subject

    """
    if np.any(np.diff(sample_times) <= 0):
        raise ValueError("`saliva_times` must be increasing!")
    if (len(sample_times) * num_subjects) != num_samples:
        raise ValueError(
            "Length of `saliva_times` does not match the number of saliva samples! Expected: {}, got: {}".format(
                int(num_samples / num_subjects), len(sample_times)
            )
        )


def _parse_condition_list(
    data: pd.DataFrame, condition_list: Union[Sequence, Dict[str, Sequence], pd.Index]
) -> pd.DataFrame:
    if isinstance(condition_list, (list, np.ndarray)):
        # Add Condition as new index level
        condition_list = pd.DataFrame(
            condition_list,
            columns=["condition"],
            index=data.index.get_level_values("subject").unique(),
        )
    elif isinstance(condition_list, dict):
        condition_list = pd.DataFrame(condition_list)
        condition_list = condition_list.stack().reset_index(level=1).set_index("level_1").sort_values(0)
        condition_list.index.name = "condition"
    elif isinstance(condition_list, pd.DataFrame):
        condition_list = condition_list.reset_index().set_index("subject")
    return condition_list
