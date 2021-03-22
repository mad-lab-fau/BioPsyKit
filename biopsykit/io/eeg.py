from typing import Tuple

import pandas as pd

from biopsykit._types import path_t


def load_eeg_muse(file_path: path_t) -> Tuple[pd.DataFrame, int]:
    data = pd.read_csv(file_path)
    sampling_rate = 250
    # convert timestamps to datetime object, set as dataframe index and
    # the timestamp from UTC into the correct time zone
    data['timestamps'] = pd.to_datetime(data['timestamps'], unit='s')
    data.set_index('timestamps', inplace=True)
    data = data.tz_localize("UTC").tz_convert("Europe/Berlin")
    if "Right AUX" in data.columns:
        # drop the AUX column
        data.drop(columns="Right AUX", inplace=True)
    return data, sampling_rate


def write_frequency_bands(data: pd.DataFrame, file_path: path_t):
    data.to_csv(file_path)
