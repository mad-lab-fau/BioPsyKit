"""Module containing different I/O functions to load time log data, subject condition lists, questionnaire data, etc."""

from pathlib import Path
from typing import Optional, Union, Sequence, Dict

import datetime
import pytz

import numpy as np
import pandas as pd
from nilspodlib import Dataset

from biopsykit.utils.file_handling import is_excel_file
from biopsykit.utils.exceptions import ValidationError
from biopsykit.utils.dataframe_handling import convert_nan
from biopsykit.utils.time import tz

from biopsykit.utils.datatype_helper import (
    SubjectConditionDataFrame,
    SubjectConditionDict,
    is_subject_condition_dict,
    is_subject_condition_dataframe,
    CodebookDataFrame,
    is_codebook_dataframe,
    _SubjectConditionDataFrame,
    _CodebookDataFrame,
)

from biopsykit.utils._datatype_validation_helper import _assert_file_extension, _assert_has_columns, _assert_is_dtype
from biopsykit.utils._types import path_t

__all__ = [
    "load_time_log",
    "load_subject_condition_list",
    "load_questionnaire_data",
    "load_pandas_dict_excel",
    "load_codebook",
    "convert_time_log_datetime",
    "write_pandas_dict_excel",
    "write_result_dict",
]


def load_time_log(
    file_path: path_t,
    index_cols: Optional[Union[str, Sequence[str], Dict[str, str]]] = None,
    phase_cols: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    continuous_time: Optional[bool] = True,
    **kwargs,
) -> pd.DataFrame:
    """Load time log information from file.

    This function can be used to load a file containing "time logs", i.e.,
    information about start and stop times of recordings or recording phases per subject.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to time log file. Must either be an Excel or csv file
    index_cols : str, list of str or dict, optional
        specifies dataframe index. Can either be a string or a list of strings to indicate column name(s) from the
        dataframe that should be used as index level(s), or ``None`` for no index.
        If the index levels of the time log dataframe should have different names than the columns in the file,
        a dict specifying the mapping (column_name : new_index_name) can be passed. Default: ``None``
    phase_cols : list of str or dict, optional
        list of column names that contain time log information or ``None`` to use all columns.
        If the column names of the time log dataframe should have different names than the columns in the file,
        a dict specifying the mapping (column_name : new_column_name) can be passed. Default: ``None``
    continuous_time: bool, optional
        flag indicating whether phases are continuous, i.e., whether the end of the previous phase is also the
        beginning of the next phase or not. Default: ``True``.
        If ``continuous_time`` is set to ``False``, the start and end columns of all phases must have the
        suffixes "_start" and "_end", respectively
    **kwargs
        Additional parameters that are passed to :func:`pandas.read_csv` or :func:`pandas.read_excel`

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with time log information

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if file format is none of [".xls", ".xlsx", ".csv"]
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if ``continuous_time`` is ``False``, but "start" and "end" time columns of each phase do not match or
        none of these columns were found in the dataframe

    Examples
    --------
    >>> import biopsykit as bp
    >>> file_path = "./timelog.csv"
    >>> # Example 1:
    >>> # load time log file into a pandas dataframe
    >>> data = bp.io.load_time_log(file_path)
    >>> # Example 2:
    >>> # load time log file into a pandas dataframe and specify the 'subject_id' column
    >>> # in the time log file to be the index of the dataframe
    >>> data = bp.io.load_time_log(file_path, index_cols="subject_id")
    >>> # Example 3:
    >>> # load time log file into a pandas dataframe, specify the columns "subject_id" and "condition"
    >>> # to be the new index, and the columns 'Phase1' 'Phase2' and 'Phase3' to be used
    >>> # for extracting time information
    >>> data = bp.io.load_time_log(
    >>>     file_path, index_cols=["subject_id", "condition"], phase_cols=["Phase1", "Phase2", "Phase3"]
    >>> )
    >>> # Example 4:
    >>> # load time log file into a pandas dataframe and specify the column "subject_id" as index, but the index name
    >>> # in the dataframe should be "subject"
    >>> data = bp.io.load_time_log(file_path,
    >>>     index_cols={"subject_id": "subject"},
    >>>     phase_cols=["Phase1", "Phase2", "Phase3"]
    >>> )

    """
    # ensure pathlib
    file_path = Path(file_path)

    _assert_file_extension(file_path, expected_extension=[".xls", ".xlsx", ".csv"])
    # assert times in the excel sheet are imported as strings,
    # not to be automatically converted into datetime objects
    kwargs["dtype"] = str
    data = _load_dataframe(file_path, **kwargs)

    data = _apply_index_cols(data, index_cols)
    data = _apply_phase_cols(data, phase_cols)

    if not continuous_time:
        data = _parse_time_log_not_continuous(data, index_cols)

    for val in data.values.flatten():
        if val is np.nan:
            continue
        _assert_is_dtype(val, str)

    return data


def _apply_phase_cols(data: pd.DataFrame, phase_cols: Union[Dict[str, Sequence[str]], Sequence[str]]) -> pd.DataFrame:
    new_phase_cols = None
    if isinstance(phase_cols, dict):
        new_phase_cols = phase_cols
        phase_cols = list(phase_cols.keys())

    if phase_cols:
        _assert_has_columns(data, [phase_cols])
        data = data.loc[:, phase_cols]
    if new_phase_cols:
        data = data.rename(columns=new_phase_cols)

    return data


def _parse_time_log_not_continuous(
    data: pd.DataFrame, index_cols: Union[str, Sequence[str], Dict[str, str]]
) -> pd.DataFrame:
    start_cols = np.squeeze(data.columns.str.extract(r"(\w+)_start").dropna().values)
    end_cols = np.squeeze(data.columns.str.extract(r"(\w+)_end").dropna().values)
    if start_cols.size == 0:
        raise ValidationError(
            "No 'start' and 'end' columns were found. "
            "Make sure that each phase has columns with 'start' and 'end' suffixes!"
        )
    if not np.array_equal(start_cols, end_cols):
        raise ValidationError("Not all phases have 'start' and 'end' columns!")

    if index_cols is None:
        index_cols = [s for s in ["subject", "condition"] if s in data.columns]
        data = data.set_index(index_cols)
    if isinstance(index_cols, dict):
        index_cols = data.index.names

    data = pd.wide_to_long(
        data.reset_index(),
        stubnames=start_cols,
        i=index_cols,
        j="time",
        sep="_",
        suffix="(start|end)",
    )

    # ensure that "start" is always before "end"
    data = data.reindex(["start", "end"], level=-1)
    # unstack start|end level
    data = data.unstack()
    return data


def load_subject_condition_list(
    file_path: path_t,
    subject_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    return_dict: Optional[bool] = False,
    **kwargs,
) -> Union[SubjectConditionDataFrame, SubjectConditionDict]:
    """Load subject condition assignment from file.

    This function can be used to load a file that contains the assignment of subject IDs to study conditions.
    It will return a dataframe or a dictionary that complies with BioPsyKit's naming convention, i.e.,
    the subject ID index will be named ``subject`` and the condition column will be named ``condition``.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to time log file. Must either be an Excel or csv file
    subject_col : str, optional
        name of column containing subject IDs or ``None`` to use default column name ``subject``.
        According to BioPsyKit's convention, the subject ID column is expected to have the name ``subject``.
        If the subject ID column in the file has another name, the column will be renamed in the dataframe
        returned by this function.
    condition_col : str, optional
        name of column containing condition assignments or ``None`` to use default column name ``condition``.
        According to BioPsyKit's convention, the condition column is expected to have the name ``condition``.
        If the condition column in the file has another name, the column will be renamed in the dataframe
        returned by this function.
    return_dict : bool, optional
        whether to return a dict with subject IDs per condition (``True``) or a dataframe (``False``).
        Default: ``False``
    **kwargs
        Additional parameters that are passed tos :func:`pandas.read_csv` or :func:`pandas.read_excel`

    Returns
    -------
    :class:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame` or
    :class:`~biopsykit.utils.datatype_helper.SubjectConditionDict`
        a standardized pandas dataframe with subject IDs and condition assignments (if ``return_dict`` is ``False``) or
        a standardized dict with subject IDs per group (if ``return_dict`` is ``True``)

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if file is not a csv or Excel file
    :exc:`~biopsykit.utils.exceptions.ValidationError`
        if result is not a :class:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame` or a
        :class:`~biopsykit.utils.datatype_helper.SubjectConditionDict`

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, expected_extension=[".xls", ".xlsx", ".csv"])

    data = _load_dataframe(file_path, **kwargs)

    if subject_col is None:
        subject_col = "subject"

    if condition_col is None:
        condition_col = "condition"

    _assert_has_columns(data, [[subject_col, condition_col]])

    if subject_col != "subject":
        # rename column
        subject_col = {subject_col: "subject"}
        data = data.rename(columns=subject_col)
        subject_col = "subject"

    if condition_col != "condition":
        # rename column
        condition_col = {condition_col: "condition"}
        data = data.rename(columns=condition_col)
        condition_col = "condition"
    data = data.set_index(subject_col)

    if return_dict:
        data = data.groupby(condition_col).groups
        is_subject_condition_dict(data)
        return data
    is_subject_condition_dataframe(data)
    return _SubjectConditionDataFrame(data)


def _get_subject_col(data: pd.DataFrame, subject_col: str):
    if subject_col is None:
        subject_col = "subject"
    _assert_has_columns(data, [[subject_col]])
    return subject_col


def load_questionnaire_data(
    file_path: path_t,
    subject_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    additional_index_cols: Optional[Union[str, Sequence[str]]] = None,
    replace_missing_vals: Optional[bool] = True,
    remove_nan_rows: Optional[bool] = True,
    sheet_name: Optional[Union[str, int]] = 0,
    **kwargs,
) -> pd.DataFrame:
    """Load questionnaire data from file.

    The resulting dataframe will comply with BioPsyKit's naming conventions, i.e., the subject ID index will be
    named ``subject`` and a potential condition index will be named ``condition``.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to time log file. Must either be an Excel or csv file
    subject_col : str, optional
        name of column containing subject IDs or ``None`` to use default column name ``subject``.
        According to BioPsyKit's convention, the subject ID column is expected to have the name ``subject``.
        If the subject ID column in the file has another name, the column will be renamed in the dataframe
        returned by this function.
    condition_col : str, optional
        name of column containing condition assignments or ``None`` to use default column name ``condition``.
        According to BioPsyKit's convention, the condition column is expected to have the name ``condition``.
        If the condition column in the file has another name, the column will be renamed in the dataframe
        returned by this function.
    additional_index_cols : str, list of str, optional
        additional index levels to be added to the dataframe.
        Can either be a string or a list strings to indicate column name(s) that should be used as index level(s),
        or ``None`` for no additional index levels. Default: ``None``
    replace_missing_vals : bool, optional
        ``True`` to replace encoded "missing values" from software like SPSS (e.g. -77, -99, or -66)
        to "actual" missing values (NaN).
        Default: ``True``
    remove_nan_rows : bool, optional
        ``True`` to remove rows that only contain NaN values (except the index cols), ``False`` to keep NaN rows.
        Default: ``True``
    sheet_name : str or int, optional
        sheet_name identifier (str) or sheet_name index (int) if file is an Excel file.
        Default: 0 (i.e. first sheet in Excel file)

    Returns
    -------
    :class:`~pandas.DataFrame`
        dataframe with imported questionnaire data

    Raises
    ------
    :class:`~biopsykit.utils.exceptions.FileExtensionError`
        if file format is none of [".xls", ".xlsx", ".csv"]

    """
    # ensure pathlib
    file_path = Path(file_path)

    _assert_file_extension(file_path, expected_extension=[".xls", ".xlsx", ".csv"])
    if file_path.suffix != ".csv":
        kwargs["sheet_name"] = sheet_name
    data = _load_dataframe(file_path, **kwargs)

    subject_col = _get_subject_col(data, subject_col)
    data = data.rename(columns={subject_col: "subject"})
    subject_col = "subject"
    index_cols = [subject_col]

    if condition_col is not None:
        _assert_has_columns(data, [[condition_col]])
        data = data.rename(columns={condition_col: "condition"})
        condition_col = "condition"
        index_cols.append(condition_col)

    if additional_index_cols is None:
        additional_index_cols = []
    if isinstance(additional_index_cols, str):
        additional_index_cols = [additional_index_cols]

    index_cols = index_cols + additional_index_cols
    data = _apply_index_cols(data, index_cols=index_cols)

    if replace_missing_vals:
        data = convert_nan(data)
    if remove_nan_rows:
        data = data.dropna(how="all")
    return data


def load_codebook(file_path: path_t, **kwargs) -> CodebookDataFrame:
    """Load codebook from file.

    A codebook is used to convert numerical values from a dataframe (e.g., from questionnaire data)
    to categorical values.


    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        file path to codebook
    **kwargs
        additional arguments to pass to :func:`pandas.read_csv` or :func:`pandas.read_excel`


    Returns
    -------
    :class:`~pandas.DataFrame`
        :obj:`~biopsykit.utils.datatype_helper.CodebookDataFrame`, a dataframe in a standardized format


    See Also
    --------
    :func:`~biopsykit.utils.dataframe_handling.apply_codebook`
        apply codebook to data

    """
    # ensure pathlib
    file_path = Path(file_path)

    _assert_file_extension(file_path, expected_extension=[".xls", ".xlsx", ".csv"])
    if file_path.suffix in [".xls", ".xlsx"]:
        data = pd.read_excel(file_path, **kwargs)
    else:
        data = pd.read_csv(file_path, **kwargs)

    _assert_has_columns(data, [["variable"]])
    data = data.set_index("variable")
    is_codebook_dataframe(data)

    return _CodebookDataFrame(data)


# def load_stroop_inquisit_data(folder_path: path_t, cols: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
#     """Load Inquisit data collected during "Stroop Test".
#
#     Stroop Test reusults (mean response time, number of correct answers, etc.)
#     are exported per Stroop phase and are stored in a common folder. This function loads all exported `.iqdat` files,
#     transforms them into dataframes and combines them into a dictionary.
#
#     Parameters
#     ----------
#     folder_path : :any:`pathlib.Path` or str
#         path to the folder in which the Stroop test export files are stored
#     cols : list of str, optional
#         names of columns which should be imported and added to the dictionary
#
#     Returns
#     -------
#     dict
#         dictionary with Stroop Test parameters per Stroop Phase
#
#     """
#     dict_stroop = {}
#     # ensure pathlib
#     folder_path = Path(folder_path)
#     # look for all Inquisit files in the folder
#     dataset_list = list(sorted(folder_path.glob("*.iqdat")))
#     subject = ""
#     # iterate through data
#     for data_path in dataset_list:
#         df_stroop = pd.read_csv(data_path, sep="\t")
#         if subject != df_stroop["subject"][0]:
#             dict_stroop = {}
#         # set subject, stroop phase
#         subject = df_stroop["subject"][0]
#         subphase = "Stroop{}".format(str(df_stroop["sessionid"][0])[-1])
#         df_mean = df_stroop.mean(axis=0).to_frame().T
#
#         if cols:
#             dict_stroop[subphase] = df_mean[cols]
#         else:
#             dict_stroop[subphase] = df_mean
#
#     return dict_stroop


def convert_time_log_datetime(
    time_log: pd.DataFrame,
    dataset: Optional[Dataset] = None,
    df: Optional[pd.DataFrame] = None,
    date: Optional[Union[str, datetime.datetime]] = None,
    timezone: Optional[Union[str, datetime.tzinfo]] = None,
) -> pd.DataFrame:
    """Convert the time log information into datetime objects.

    This function converts time log information (containing only time, but no date)
    into datetime objects, thus, adds the `start date` of the recording. To specify the recording date,
    either a NilsPod :class:`~nilspodlib.dataset.Dataset` or a pandas dataframe with a :class:`~pandas.DatetimeIndex`
    must be supplied from which the recording date can be extracted.
    As an alternative, the date can be specified explicitly via ``date`` parameter.

    Parameters
    ----------
    time_log : :class:`~pandas.DataFrame`
        pandas dataframe with time log information
    dataset : :class:`~nilspodlib.dataset.Dataset`, optional
        NilsPod Dataset object extract time and date information. Default: ``None``
    df : :class:`~pandas.DataFrame`, optional
        dataframe with :class:`~pandas.DatetimeIndex` to extract time and date information. Default: ``None``
    date : str or datetime, optional
        datetime object or date string used to convert time log information into datetime.
        If ``date`` is a string, it must be supplied in a common date format, e.g. "dd.mm.yyyy" or "dd/mm/yyyy".
        Default: ``None``
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of the acquired data to convert, either as string of as tzinfo object.
        Default: "Europe/Berlin"

    Returns
    -------
    :class:`~pandas.DataFrame`
        pandas dataframe with log time converted into datetime

    Raises
    ------
    ValueError
        if none of ``dataset``, ``df`` and ``date`` are supplied as argument,
        or if index of ``df`` is not a :class:`~pandas.DatetimeIndex`

    """
    if dataset is None and date is None and df is None:
        raise ValueError("Either `dataset`, `df` or `date` must be supplied as argument to retrieve date information!")

    date = _extract_date(dataset, df, date)

    if timezone is None:
        timezone = tz
    if isinstance(timezone, str):
        timezone = pytz.timezone(timezone)

    if isinstance(time_log.values.flatten()[0], str):
        # convert time strings into datetime.time object
        time_log = time_log.applymap(pd.to_datetime)
        time_log = time_log.applymap(lambda val: val.time())

    time_log = time_log.applymap(lambda x: timezone.localize(datetime.datetime.combine(date, x)))
    return time_log


def _extract_date(dataset: Dataset, df: pd.DataFrame, date: Union[str, datetime.datetime]) -> datetime.datetime:
    if dataset is not None:
        date = dataset.info.utc_datetime_start.date()
    if df is not None:
        if isinstance(df.index, pd.DatetimeIndex):
            date = df.index.normalize().unique()[0]
            date = date.to_pydatetime()
        else:
            raise ValueError("'df' must have a DatetimeIndex!")
    if isinstance(date, str):
        # ensure datetime
        date = pd.to_datetime(date)
        date = date.date()

    return date


def load_pandas_dict_excel(
    file_path: path_t, index_col: Optional[str] = "time", timezone: Optional[Union[str, datetime.tzinfo]] = None
) -> Dict[str, pd.DataFrame]:
    """Load Excel file containing pandas dataframes with time series data of one subject.

    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    index_col : str, optional
        name of index columns of dataframe or ``None`` if no index column is present. Default: "time"
    timezone : str or :class:`datetime.tzinfo`, optional
        timezone of the acquired data for localization (since Excel does not support localized timestamps),
        either as string of as tzinfo object.
        Default: "Europe/Berlin"

    Returns
    -------
    dict
        dictionary with multiple pandas dataframes

    Raises
    ------
    :class:`~biopsykit.utils.exceptions.FileExtensionError`
        if file is no Excel file (".xls" or ".xlsx")

    See Also
    --------
    write_pandas_dict_excel : Write dictionary with dataframes to file

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, (".xls", ".xlsx"))

    # assure that the file is an Excel file
    is_excel_file(file_path)

    dict_df: Dict[str, pd.DataFrame] = pd.read_excel(file_path, index_col=index_col, sheet_name=None)

    # (re-)localize each sheet since Excel does not support timezone-aware dates (if index is DatetimeIndex)
    for key in dict_df:
        if isinstance(dict_df[key].index, pd.DatetimeIndex):
            dict_df[key] = dict_df[key].tz_localize(timezone)
    return dict_df


def write_pandas_dict_excel(
    data_dict: Dict[str, pd.DataFrame],
    file_path: path_t,
    index_col: Optional[bool] = True,
):
    """Write a dictionary with pandas dataframes to an Excel file.

    Parameters
    ----------
    data_dict : dict
        dictionary with pandas dataframes
    file_path : :class:`~pathlib.Path` or str
        path to exported Excel file
    index_col : bool, optional
        ``True`` to include dataframe index in Excel export, ``False`` otherwise. Default: ``True``

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if ``file_path`` is not an Excel file

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, [".xls", ".xlsx"])

    writer = pd.ExcelWriter(file_path, engine="xlsxwriter")  # pylint:disable=abstract-class-instantiated
    for key in data_dict:
        if isinstance(data_dict[key].index, pd.DatetimeIndex):
            # un-localize DateTimeIndex because Excel doesn't support timezone-aware dates
            data_dict[key].tz_localize(None).to_excel(writer, sheet_name=key, index=index_col)
        else:
            data_dict[key].to_excel(writer, sheet_name=key, index=index_col)
    writer.close()


def write_result_dict(
    result_dict: Dict[str, pd.DataFrame],
    file_path: path_t,
    index_name: Optional[str] = "subject",
):
    """Write dictionary with processing results (e.g. HR, HRV, RSA) to csv file.

    The keys in the dictionary should be the subject IDs (or any other identifier),
    the values should be :class:`~pandas.DataFrame`. The index level(s) of the exported dataframe can be specified
    by the ``index_col`` parameter.

    The dictionary will be concatenated to one large dataframe which will then be saved as csv file.

    Parameters
    ----------
    result_dict : dict
        Dictionary containing processing results for all subjects. The keys in the dictionary should be the Subject IDs
        (or any other identifier), the values should be pandas dataframes
    file_path : :class:`~pathlib.Path`, str
        path to file
    index_name : str, optional
        name of the index resulting from concatenting dataframes. Default: ``subject``

    Raises
    ------
    :exc:`~biopsykit.utils.exceptions.FileExtensionError`
        if ``file_path`` is not a csv or Excel file

    Examples
    --------
    >>>
    >>> from biopsykit.io import write_result_dict
    >>>
    >>> file_path = "./param_results.csv"
    >>>
    >>> dict_param_output = {
    >>> 'S01' : pd.DataFrame(), # e.g., dataframe from mist_param_subphases,
    >>> 'S02' : pd.DataFrame(),
    >>> # ...
    >>> }
    >>>
    >>> write_result_dict(dict_param_output, file_path=file_path, index_name="subject")

    """
    # ensure pathlib
    file_path = Path(file_path)
    _assert_file_extension(file_path, [".csv", ".xls", ".xlsx"])
    df_result_concat = pd.concat(result_dict, names=[index_name])
    if file_path.suffix in [".csv"]:
        df_result_concat.to_csv(file_path)
    else:
        writer = pd.ExcelWriter(file_path, engine="xlsxwriter")  # pylint:disable=abstract-class-instantiated
        df_result_concat.to_excel(writer)
        writer.close()


def _load_dataframe(file_path, **kwargs):
    if file_path.suffix in [".csv"]:
        return pd.read_csv(file_path, **kwargs)
    return pd.read_excel(file_path, **kwargs)


def _apply_index_cols(
    data: pd.DataFrame, index_cols: Optional[Union[str, Sequence[str], Dict[str, str]]] = None
) -> pd.DataFrame:
    new_index_cols = None
    if isinstance(index_cols, str):
        index_cols = [index_cols]
    elif isinstance(index_cols, dict):
        new_index_cols = list(index_cols.values())
        index_cols = list(index_cols.keys())

    if index_cols is not None:
        _assert_has_columns(data, [index_cols])
        data = data.set_index(index_cols)

    if new_index_cols is not None:
        data.index = data.index.set_names(new_index_cols)

    return data
