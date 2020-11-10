from typing import Dict, Optional, List

import pandas as pd

from biopsykit.utils import path_t, tz


def load_hr_excel(file_path: path_t) -> Dict[str, pd.DataFrame]:
    """
    Loads excel file containing heart rate data of one subject (as exported by `write_hr_to_excel`).

    The dictionary will have the following format: { "<Phase>" : hr_dataframe }

    Each hr_dataframe has the following format:
        * 'date' Index: DateTimeIndex with timestamps of the heart rate samples
        * 'ECG_Rate' Column: heart rate samples

    Parameters
    ----------
    file_path : path or str
        path to file

    Returns
    -------
    dict
        Excel sheet dictionary

    """
    dict_hr = pd.read_excel(file_path, index_col="date", sheet_name=None)
    # (re-)localize each sheet since Excel does not support timezone-aware dates
    dict_hr = {k: v.tz_localize(tz) for k, v in dict_hr.items()}
    return dict_hr


def load_hr_excel_all_subjects(base_path: path_t, subject_folder_pattern: str,
                               filename_pattern: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Loads HR processing results (as exported by `write_hr_to_excel`) from all subjects and combines them into one
    dictionary ('HR subject dict').

    The dictionary will have the following format:

    { "<Subject_ID>" : {"<Phase>" : hr_dataframe } }@


    Parameters
    ----------
    base_path : str or path
        path to top-level folder containing all subject folders
    subject_folder_pattern : str
        subject folder pattern. Folder names are assumed to be Subject IDs
    filename_pattern : str
        filename pattern of HR result files

    Returns
    -------

    Examples
    --------
    >>> import biopsykit as ep
    >>> base_path = "./"
    >>> dict_hr_subjects = ep.load_hr_excel_all_subjects(
    >>>                         base_path, subject_folder_pattern="Vp_*",
    >>>                         filename_pattern="ecg_result*.xlsx")
    >>>
    >>> print(dict_hr_subjects)
    >>> {
    >>>     'Vp_01': {}, # dict as returned by load_hr_excel
    >>>     'Vp_02': {},
    >>>     # ...
    >>> }
    """
    subject_dirs = list(sorted(base_path.glob(subject_folder_pattern)))
    dict_hr_subjects = {}
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        # check whether old processing results already exist
        hr_files = sorted(subject_dir.glob(filename_pattern))
        if len(hr_files) == 1:
            dict_hr_subjects[subject_id] = load_hr_excel(hr_files[0])
        else:
            print("No HR data for subject {}".format(subject_id))
    return dict_hr_subjects


def write_hr_to_excel(ecg_processor: 'EcgProcessor', file_path: path_t) -> None:
    """
    Writes heart rate dictionary of one subject to an Excel file.
    Each of the phases in the dictionary will be a separate sheet in the file.

    The Excel file will have the following columns:
        * date: timestamps of the heart rate samples (string, will be converted to DateTimeIndex)
        * ECG_Rate: heart rate samples (float)


    Parameters
    ----------
    ecg_processor : EcgProcessor
        EcgProcessor instance
    file_path : path or str
        path to file
    """

    write_dict_to_excel(ecg_processor.heart_rate, file_path)


def write_dict_to_excel(data_dict: Dict[str, pd.DataFrame], file_path: path_t,
                        index_col: Optional[bool] = True) -> None:
    """
    Writes a dictionary containing pandas dataframes to an Excel file.

    Parameters
    ----------
    data_dict : dict
        dict with pandas dataframes
    file_path : str or path
        filepath
    index_col : bool, optional
        ``True`` to include dataframe index in Excel export, ``False`` otherwise. Default: ``True``
    """

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    for key in data_dict:
        if isinstance(data_dict[key].index, pd.DatetimeIndex):
            # un-localize DateTimeIndex because Excel doesn't support timezone-aware dates
            data_dict[key].tz_localize(None).to_excel(writer, sheet_name=key, index=index_col)
        else:
            data_dict[key].to_excel(writer, sheet_name=key, index=index_col)
    writer.save()


def write_result_dict_to_csv(result_dict: Dict[str, pd.DataFrame], file_path: path_t,
                             identifier_col: Optional[str] = "Subject_ID",
                             index_cols: Optional[List[str]] = ["Phase", "Subphase"],
                             overwrite_file: Optional[bool] = False) -> None:
    """
    Saves dictionary with processing results (e.g. HR, HRV, RSA) of all subjects as csv file.

    Simply pass a dictionary with processing results. The keys in the dictionary should be the Subject IDs
    (or any other identifier), the values should be pandas dataframes. The resulting index can be specified by the
    `identifier_col` parameter.

    The dictionary will be concatenated to one large dataframe which will then be saved as csv file.

    *Notes*:
        * If a file with same name exists at the specified location, it is assumed that this is a result file from a
          previous run and the current results should be appended to this file.
          (To disable this behavior, set `overwrite_file` to ``False``).
        * Per default, it is assumed that the 'values' dataframes has a multi-index with columns ["Phase", "Subphase"].
          The resulting dataframe would, thus, per default have the columns ["Subject_ID", "Phase", "Subphase"].
          This can be changed by the parameter `index_cols`.

    Parameters
    ----------
    result_dict : dict
        Dictionary containing processing results for all subjects. The keys in the dictionary should be the Subject IDs
        (or any other identifier), the values should be pandas dataframes
    file_path : path, str
        path to file
    identifier_col : str, optional
        Name of the index in the concatenated dataframe. Default: "Subject_ID"
    index_cols : list of str, optional
        List of index columns of the single dataframes in the dictionary. Not needed if `overwrite_file` is ``False``.
        Default: ["Phase", "Subphase"]
    overwrite_file : bool, optional
        ``True`` to overwrite the file if it already exists, ``False`` otherwise. Default: ``True``

    Examples
    --------
    >>>
    >>> import biopsykit as ep
    >>>
    >>> file_path = "./param_results.csv"
    >>>
    >>> dict_param_output = {
    >>> 'S01' : pd.DataFrame(), # dataframe from mist_param_subphases,
    >>> 'S02' : pd.DataFrame(),
    >>> # ...
    >>> }
    >>>
    >>> ep.write_result_dict_to_csv(dict_param_output, file_path=file_path,
    >>>                             identifier_col="Subject_ID", index_cols=["Phase", "Subphase"])
    """

    # TODO check if index_cols is really needed?

    identifier_col = [identifier_col]

    if index_cols is None:
        index_cols = []

    df_result_concat = pd.concat(result_dict, names=identifier_col)
    if file_path.exists() and not overwrite_file:
        # ensure that all identifier columns are read as str
        df_result_old = pd.read_csv(file_path, dtype={col: str for col in identifier_col})
        df_result_old.set_index(identifier_col + index_cols, inplace=True)
        df_result_concat = df_result_concat.combine_first(df_result_old).sort_index(level=0)
    df_result_concat.reset_index().to_csv(file_path, index=False)
