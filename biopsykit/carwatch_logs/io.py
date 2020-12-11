import re
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, Union

import pandas as pd
from tqdm.notebook import tqdm

from biopsykit.utils import path_t, utc, tz

LOG_FILENAME_PATTERN = "logs_(.*?)"


def load_logs_all_subjects(path: path_t, has_subfolder: Optional[bool] = True,
                           log_filename_pattern: Optional[str] = None,
                           return_df: Optional[bool] = True) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """

    Parameters
    ----------
    has_subfolder
    path : path or str
        path to folder containing logs
    log_filename_pattern : str, optional
    return_df : bool, optional

    Returns
    -------
    dict
        dictionary with log data per subject

    """
    # ensure pathlib
    path = Path(path)

    dict_log_files = {}
    if has_subfolder:
        folder_list = [p for p in sorted(path.glob("*")) if p.is_dir() and not p.name.startswith('.')]
        for folder in tqdm(folder_list):
            subject_id = folder.name
            dict_log_files[subject_id] = load_log_one_subject(folder)
    else:
        # first, look for available csv files
        file_list = [p for p in sorted(path.glob("*.csv"))]
        if len(file_list) > 0:
            if log_filename_pattern is None:
                log_filename_pattern = LOG_FILENAME_PATTERN + ".csv"
            for file in tqdm(file_list):
                subject_id = re.search(log_filename_pattern, file.name).group(1)
                df = pd.read_csv(file, sep=';')
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                dict_log_files[subject_id] = df
        else:
            # fallback: look for zip files
            file_list = [p for p in sorted(path.glob("*.zip"))]
            if len(file_list) > 0:
                if log_filename_pattern is None:
                    log_filename_pattern = LOG_FILENAME_PATTERN + ".zip"
                for file in tqdm(file_list):
                    subject_id = re.search(log_filename_pattern, file.name).group(1)
                    dict_log_files[subject_id] = load_log_one_subject(file)

    if return_df:
        df = pd.concat(dict_log_files, names=['subject_id'])
        return df
    else:
        return dict_log_files


def load_log_one_subject(path: path_t, log_filename_pattern: Optional[str] = None,
                         overwrite_logs_unzip: Optional[bool] = False) -> pd.DataFrame:
    """

    Parameters
    ----------
    path : path or str
        path to subject folder containing log files or path to log file
    log_filename_pattern : str, optional
    overwrite_logs_unzip : bool, optional

    Returns
    -------
    pd.DataFrame
        log data for one subject

    """
    import zipfile

    if log_filename_pattern is None:
        log_filename_pattern = LOG_FILENAME_PATTERN + ".zip"

    path = Path(path)

    if path.is_dir():
        # TODO add error messages if no log files are found (e.g. wrong folder path etc.)
        return log_folder_to_dataframe(path)
    if path.is_file():
        if path.suffix == '.zip':
            with zipfile.ZipFile(path, 'r') as zip_ref:
                export_folder = path.parent.joinpath(re.search(log_filename_pattern, path.name).group(1))
                export_folder.mkdir(exist_ok=True)
                if overwrite_logs_unzip or len(list(export_folder.glob("*"))) == 0:
                    zip_ref.extractall(export_folder)
                    # call recursively with newly exported folder
                    return log_folder_to_dataframe(export_folder)
                else:
                    # folder not empty => inform user and load folder
                    warnings.warn(
                        "Folder {} already contains log files which will be loaded. "
                        "Set `overwrite_logs_unzip = True` to overwrite log files.".format(export_folder.name))
                    return log_folder_to_dataframe(export_folder)
        if path.suffix == '.csv':
            return log_folder_to_dataframe(path)


def log_folder_to_dataframe(folder_path: path_t) -> pd.DataFrame:
    file_list = list(sorted(folder_path.glob("*.csv")))
    df = pd.concat(
        [pd.read_csv(file, sep=";", header=None, names=["time", "action", "extras"]) for file in file_list])

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    df.index = df.index.tz_localize(utc).tz_convert(tz)
    df.sort_index(inplace=True)
    df = df.apply(_parse_date, axis=1)
    return df


def save_log_data(log_data: Union[pd.DataFrame, 'LogData'], path: path_t, subject_id: Optional[str] = None,
                  overwrite: Optional[bool] = False, show_skipped: Optional[bool] = False):
    from biopsykit.carwatch_logs import LogData

    if isinstance(log_data, pd.DataFrame):
        if isinstance(log_data.index, pd.MultiIndex):
            # dataframe has a multiindex => it's a combined dataframe for all subjects
            log_data.to_csv(path, sep=';')
            return
        else:
            log_data = LogData(log_data)

    if subject_id is None:
        subject_id = log_data.subject_id

    export_path = path.joinpath("logs_{}.csv".format(subject_id))
    if not export_path.exists() or overwrite:
        log_data.df.to_csv(export_path, sep=";")
    elif show_skipped:
        print("Skipping {}. Already exported.".format(subject_id))


def _parse_date(row: pd.Series) -> pd.Series:
    json_extra = json.loads(row['extras'])
    row_cpy = row.copy()
    keys = ['timestamp', 'timestamp_hidden']
    for key in keys:
        if key in json_extra:
            # convert to datetime and localize
            time = utc.localize(pd.to_datetime(json_extra[key], unit='ms'))
            # convert to correct time zone
            time = time.astimezone(tz)
            json_extra[key] = str(time)

    # fix wrong key for saliva_id in "alarm_ring" action (old app versions)
    if 'extra_saliva_id' in json_extra:
        json_extra['saliva_id'] = json_extra['extra_saliva_id']
        del json_extra['extra_saliva_id']
    row_cpy['extras'] = json.dumps(json_extra)
    return row_cpy
