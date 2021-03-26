from pathlib import Path
from typing import Optional, Union, Callable

import pandas as pd

import ipywidgets.widgets

from biopsykit._types import path_t
from biopsykit.carwatch_logs import LogData

LOG_FILENAME_PATTERN = "logs_(.*?)"


def log_file_subject_dropdown(path: path_t, input_type: Optional[str] = 'file', value_type: Optional[str] = 'file_path',
                              callback: Optional[Callable] = None) -> ipywidgets.Dropdown:
    import re

    input_types = ['folder', 'file']
    value_types = ['file_path', 'file_name', 'folder_name', 'subject_id']

    if input_type not in input_types:
        raise ValueError("Invalid input_type!")

    if value_type not in value_types:
        raise ValueError("Invalid value_type!")

    path = Path(path)
    if input_type == 'file':
        log_file_pattern = LOG_FILENAME_PATTERN + ".csv"
        log_file_list = [log_file for log_file in list(sorted(path.glob("*.csv")))]
        subject_list = [re.search(log_file_pattern, log_file.name).group(1) for log_file in log_file_list]
    if input_type == 'folder':
        log_file_list = [folder for folder in path.glob("*") if folder.is_dir() and not folder.name.startswith(".")]
        subject_list = [folder.name for folder in log_file_list]

    option_list = [("Select Subject", None)]
    if value_type == 'file_path':
        option_list = option_list + list(zip(subject_list, log_file_list))
    if value_type in ['folder_name', 'subject_id']:
        option_list = option_list + list(zip(subject_list, subject_list))
    if value_type == 'file_name':
        option_list = option_list + list(zip(subject_list, [log_file.name for log_file in log_file_list]))

    widget = ipywidgets.Dropdown(options=option_list, description="Subject ID")
    if callback:
        widget.observe(callback, names='value')
    return widget


def action_dropdown_widget(log_data: Union[LogData, pd.DataFrame],
                           callback: Optional[Callable] = None) -> ipywidgets.Dropdown:
    options = [("Select Action", None)]

    if isinstance(log_data, LogData):
        df = log_data.df
    else:
        df = log_data

    avail_actions = list(df['action'].unique())
    options = options + list(zip(avail_actions, avail_actions))

    widget = ipywidgets.Dropdown(options=options, description="Action")
    if callback:
        widget.observe(callback, names='value')
    return widget


def day_dropdown_widget(log_data: Union[LogData, pd.DataFrame],
                        callback: Optional[Callable] = None) -> ipywidgets.Dropdown:
    options = [("Select Day", None)]

    if isinstance(log_data, LogData):
        df = log_data.df
    else:
        df = log_data

    dates = df.index.normalize().unique()
    dates = [str(date.date()) for date in dates]

    options = options + list(zip(dates, dates))
    widget = ipywidgets.Dropdown(options=options, description="Day")
    if callback:
        widget.observe(callback, names='value')
    return widget
