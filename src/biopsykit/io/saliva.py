"""Module containing different I/O functions for saliva data."""
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Tuple

import numpy as np
import pandas as pd
from biopsykit.io.io import _apply_index_cols
from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_has_columns
from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import (
    SalivaRawDataFrame,
    is_saliva_raw_dataframe,
    is_subject_condition_dataframe,
    SubjectConditionDataFrame,
    _SubjectConditionDataFrame,
    _SalivaRawDataFrame,
)

__all__ = ["load_saliva_plate", "save_saliva", "load_saliva_wide_format"]

_DATA_COL_NAMES = {"cortisol": "cortisol (nmol/l)", "amylase": "amylase (U/ml)"}


def load_saliva_plate(
    file_path: path_t,
    saliva_type: str,
    sample_id_col: Optional[str] = None,
    data_col: Optional[str] = None,
    id_col_names: Optional[Sequence[str]] = None,
    regex_str: Optional[str] = None,
    sample_times: Optional[Sequence[int]] = None,
    condition_list: Optional[Union[Sequence, Dict[str, Sequence], pd.Index]] = None,
    **kwargs,
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
          (unless column names are explicitly specified in ``data_col_names``)
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
    file_path: :class:`~pathlib.Path` or str
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
    **kwargs
        Additional parameters that are passed to :func:`pandas.read_excel`

    Returns
    -------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if file is no Excel file (.xls or .xlsx)
    ValueError
        if any saliva sample can not be converted into a float (e.g. because there was text in one of the columns)
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if imported data can not be parsed to a SalivaRawDataFrame

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, (".xls", ".xlsx"))

    # TODO add remove_nan option (all or any)
    if regex_str is None:
        regex_str = r"(Vp\d+) (S\w)"

    if sample_id_col is None:
        sample_id_col = "sample ID"

    if data_col is None:
        data_col = _DATA_COL_NAMES[saliva_type]

    df_saliva = pd.read_excel(file_path, skiprows=2, usecols=[sample_id_col, data_col], **kwargs)
    cols = df_saliva[sample_id_col].str.extract(regex_str)
    id_col_names = _get_id_columns(id_col_names, cols)

    df_saliva[id_col_names] = cols

    df_saliva = df_saliva.drop(columns=[sample_id_col], errors="ignore")
    df_saliva = df_saliva.rename(columns={data_col: saliva_type})
    df_saliva = df_saliva.set_index(id_col_names)

    if condition_list is not None:
        df_saliva = _apply_condition_list(df_saliva, condition_list)

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

    is_saliva_raw_dataframe(df_saliva, saliva_type)

    return _SalivaRawDataFrame(df_saliva)


def save_saliva(
    file_path: path_t,
    data: SalivaRawDataFrame,
    saliva_type: Optional[str] = "cortisol",
    as_wide_format: Optional[bool] = False,
):
    """Save saliva data to csv file.

    Parameters
    ----------
    file_path: :class:`~pathlib.Path` or str
        file path to export. Must be a csv or an Excel file
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format
    saliva_type : str
        type of saliva data in the dataframe
    as_wide_format : bool, optional
        ``True`` to save data in wide format (and flatten all index levels), ``False`` to save data in long-format.
        Default: ``False``

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``data`` is not a SalivaRawDataFrame
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if ``file_path`` is not a csv or Excel file

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, [".csv", ".xls", ".xlsx"])

    is_saliva_raw_dataframe(data, saliva_type)
    data = data[saliva_type]
    if as_wide_format:
        levels = list(data.index.names)
        levels.remove("subject")
        data = data.unstack(level=levels)
        data.columns = ["_".join(col) for col in data.columns]

    if file_path.suffix in [".csv"]:
        data.to_csv(file_path)
    else:
        writer = pd.ExcelWriter(file_path, engine="xlsxwriter")  # pylint:disable=abstract-class-instantiated
        data.to_excel(writer)
        writer.close()


def load_saliva_wide_format(
    file_path: path_t,
    saliva_type: str,
    subject_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    additional_index_cols: Optional[Union[str, Sequence[str]]] = None,
    sample_times: Optional[Sequence[int]] = None,
    **kwargs,
) -> SalivaRawDataFrame:
    """Load saliva data that is in wide-format from csv file.

    It will return a `SalivaRawDataFrame`, a long-format dataframe that complies with BioPsyKit's naming convention,
    i.e., the subject ID index will be named ``subject``, the sample index will be names ``sample``,
    and the value column will be named after the saliva biomarker type.

    Parameters
    ----------
    file_path: :class:`~pathlib.Path` or str
        path to file
    saliva_type: str
        saliva type to load from file. Example: ``cortisol``
    subject_col: str, optional
        name of column containing subject IDs or ``None`` to use the default column name ``subject``.
        According to BioPsyKit's convention, the subject ID column is expected to have the name ``subject``.
        If the subject ID column in the file has another name, the column will be renamed in the dataframe
        returned by this function. Default: ``None``
    condition_col : str, optional
        name of the column containing condition assignments or ``None`` if no conditions are present.
        According to BioPsyKit's convention, the condition column is expected to have the name ``condition``.
        If the condition column in the file has another name, the column will be renamed in the dataframe
        returned by this function. Default: ``None``
    additional_index_cols : str or list of str, optional
        additional index levels to be added to the dataframe, e.g., "day" index. Can either be a string or a list
        strings to indicate column name(s) that should be used as index level(s),
        or ``None`` for no additional index levels. Default: ``None``
    sample_times: list of int, optional
        times at which saliva samples were collected or ``None`` if no sample times should be specified.
        Default: ``None``
    **kwargs
        Additional parameters that are passed to :func:`pandas.read_csv` or :func:`pandas.read_excel`

    Returns
    -------
    data : :class:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame`
        saliva data in `SalivaRawDataFrame` format

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if file is no csv or Excel file

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, [".csv", ".xls", ".xlsx"])
    data = _read_dataframe(file_path, **kwargs)

    if subject_col is None:
        subject_col = "subject"

    _assert_has_columns(data, [[subject_col]])

    if subject_col != "subject":
        # rename column
        data = data.rename(columns={subject_col: "subject"})
        subject_col = "subject"

    index_cols = [subject_col]

    data, condition_col = _get_condition_col(data, condition_col)

    index_cols = _get_index_cols(condition_col, index_cols, additional_index_cols)
    data = _apply_index_cols(data, index_cols=index_cols)

    num_subjects = len(data)
    data.columns = pd.MultiIndex.from_product([[saliva_type], data.columns], names=[None, "sample"])
    data = data.stack()

    _check_num_samples(len(data), num_subjects)

    if sample_times is not None:
        _check_sample_times(len(data), num_subjects, sample_times)
        data["time"] = np.array(sample_times * num_subjects)

    is_saliva_raw_dataframe(data, saliva_type)

    return _SalivaRawDataFrame(data)


def _get_index_cols(condition_col: str, index_cols: Sequence[str], additional_index_cols: Sequence[str]):
    if condition_col is not None:
        index_cols = [condition_col] + index_cols

    if additional_index_cols is None:
        additional_index_cols = []
    if isinstance(additional_index_cols, str):
        additional_index_cols = [additional_index_cols]

    return index_cols + additional_index_cols


def _read_dataframe(file_path: Path, **kwargs):
    if file_path.suffix in [".csv"]:
        return pd.read_csv(file_path, **kwargs)
    return pd.read_excel(file_path, **kwargs)


def _check_num_samples(num_samples: int, num_subjects: int):
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


def _check_sample_times(num_samples: int, num_subjects: int, sample_times: Sequence[int]):
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
) -> SubjectConditionDataFrame:
    if isinstance(condition_list, (list, np.ndarray)):
        # Add Condition as new index level
        condition_list = pd.DataFrame(
            condition_list,
            columns=["condition"],
            index=data.index.get_level_values("subject").unique(),
        )
    elif isinstance(condition_list, dict):
        condition_list = [(subject, cond) for cond in condition_list for subject in condition_list[cond]]
        condition_list = pd.DataFrame(condition_list, columns=["subject", "condition"])
        condition_list = condition_list.set_index("subject")
    elif isinstance(condition_list, pd.DataFrame):
        condition_list = condition_list.reset_index().set_index("subject")

    is_subject_condition_dataframe(condition_list)
    return _SubjectConditionDataFrame(condition_list)


def _apply_condition_list(
    data: pd.DataFrame,
    condition_list: Optional[Union[Sequence, Dict[str, Sequence], pd.Index]] = None,
):
    condition_list = _parse_condition_list(data, condition_list)

    data = (
        data.join(condition_list).set_index("condition", append=True).reorder_levels(["condition", "subject", "sample"])
    )
    return data


def _get_id_columns(id_col_names: Sequence[str], extracted_cols: pd.DataFrame):
    if id_col_names is None:
        id_col_names = ["subject", "sample"]
        if len(extracted_cols.columns) == 3:
            id_col_names = ["subject", "day", "sample"]
    else:
        if len(id_col_names) != len(extracted_cols.columns):
            raise ValueError(
                "Number of 'id_col_names' must match length of extracted index columns! Expected {}, got {}.".format(
                    len(extracted_cols), len(id_col_names)
                )
            )

    return id_col_names


def _get_condition_col(data: pd.DataFrame, condition_col: str) -> Tuple[pd.DataFrame, str]:
    if condition_col is None:
        if "condition" in data.columns:
            condition_col = "condition"
    else:
        _assert_has_columns(data, [[condition_col]])
        if condition_col != "condition":
            # rename column
            data = data.rename(columns={condition_col: "condition"})
            condition_col = "condition"
    return data, condition_col
