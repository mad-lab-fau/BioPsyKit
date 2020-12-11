import re
import json
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
    log_filename_pattern

    Returns
    -------
    dict
        dictionary with log data per subject

    """
    # ensure pathlib
    path = Path(path)

    if log_filename_pattern is None:
        log_filename_pattern = LOG_FILENAME_PATTERN + ".csv"

    dict_log_files = {}
    if has_subfolder:
        folder_list = [p for p in sorted(path.glob("*")) if p.is_dir()]
        for folder in tqdm(folder_list):
            subject_id = folder.name
            dict_log_files[subject_id] = load_log_one_subject(folder)
    else:
        file_list = [p for p in sorted(path.glob("*.csv"))]
        for file in tqdm(file_list):
            subject_id = re.search(log_filename_pattern, file.name).group(1)
            df = pd.read_csv(file, sep=';')
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            dict_log_files[subject_id] = df

    if return_df:
        return pd.concat(dict_log_files)
    else:
        return dict_log_files


def load_log_one_subject(path: path_t, log_filename_pattern: Optional[str] = None,
                         overwrite_logs: Optional[bool] = False) -> pd.DataFrame:
    """

    Parameters
    ----------
    path : path or str
        path to subject folder containing log files or path to log file
    log_filename_pattern : str, optional
    overwrite_logs : bool, optional

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
        return log_folder_to_dataframe(path)
    if path.is_file():
        if path.suffix == '.zip':
            with zipfile.ZipFile(path, 'r') as zip_ref:
                export_folder = path.parent.joinpath(re.search(log_filename_pattern, path.name).group(1))
                export_folder.mkdir(exist_ok=True)
                if overwrite_logs or len(list(export_folder.glob("*"))) == 0:
                    zip_ref.extractall(export_folder)
                    # call recursively with newly exported folder
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


def export_log_dataframe(df: pd.DataFrame, subject_id: str, folder_path: path_t,
                         overwrite: Optional[bool] = False, show_skipped: Optional[bool] = False) -> bool:
    export_path = folder_path.joinpath("logs_{}.csv".format(subject_id))

    if not export_path.exists() or overwrite:
        # print("--- Summary {} ---".format(subject_id))
        # log_data = LogData(df, error_handling='warn')
        # print("Logged Days: {}".format([str(date) for date in log_data.log_days]))
        # print("Finished Days: {}".format(log_data.num_finished_days))
        # print("---------------------")
        # log_data.export_csv(export_path)
        df.to_csv(export_path, sep=";")
        return True
    elif show_skipped:
        print("Skipping {}. Already exported.".format(subject_id))

    return False


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
