"""Module wrapping biopsykit.io.biomarker including only I/O functions for saliva data."""
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from biopsykit.io import biomarker

__all__ = ["load_saliva_plate", "load_saliva_wide_format", "save_saliva"]

from biopsykit.utils._types import path_t
from biopsykit.utils.datatype_helper import SalivaRawDataFrame, SubjectConditionDataFrame


def load_saliva_plate(
    file_path: path_t,
    saliva_type: str,
    sample_id_col: Optional[str] = None,
    data_col: Optional[str] = None,
    id_col_names: Optional[Sequence[str]] = None,
    regex_str: Optional[str] = None,
    sample_times: Optional[Sequence[int]] = None,
    condition_list: Optional[Union[Sequence, dict[str, Sequence], pd.Index]] = None,
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
    return biomarker.load_saliva_plate(
        file_path, saliva_type, sample_id_col, data_col, id_col_names, regex_str, sample_times, condition_list, **kwargs
    )


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
    biomarker.save_saliva(file_path, data, saliva_type, as_wide_format)


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
    return biomarker.load_saliva_wide_format(
        file_path, saliva_type, subject_col, condition_col, additional_index_cols, sample_times, **kwargs
    )


def _get_index_cols(condition_col: str, index_cols: Sequence[str], additional_index_cols: Sequence[str]):
    return biomarker._get_index_cols(condition_col, index_cols, additional_index_cols)


def _read_dataframe(file_path: Path, **kwargs):
    return biomarker._read_dataframe(file_path, **kwargs)


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
    biomarker._check_num_samples(num_samples, num_subjects)


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
    biomarker._check_sample_times(num_samples, num_subjects, sample_times)


def _parse_condition_list(
    data: pd.DataFrame, condition_list: Union[Sequence, dict[str, Sequence], pd.Index]
) -> SubjectConditionDataFrame:
    return biomarker._parse_condition_list(data, condition_list)


def _apply_condition_list(
    data: pd.DataFrame,
    condition_list: Optional[Union[Sequence, dict[str, Sequence], pd.Index]] = None,
):
    return biomarker._apply_condition_list(data, condition_list)


def _get_id_columns(id_col_names: Sequence[str], extracted_cols: pd.DataFrame):

    return biomarker._get_id_columns(id_col_names, extracted_cols)


def _get_condition_col(data: pd.DataFrame, condition_col: str) -> tuple[pd.DataFrame, str]:
    return biomarker._get_condition_col(data, condition_col)
