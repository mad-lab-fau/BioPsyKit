# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
from pathlib import Path
from typing import TypeVar, Sequence, Optional, Dict, Union
import pytz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from NilsPodLib import Dataset

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')

tz = pytz.timezone('Europe/Berlin')
utc = pytz.timezone('UTC')

cmap_fau = sns.color_palette(["#003865", "#c99313", "#8d1429", "#00b1eb", "#009b77", "#98a4ae"])
_keys_fau = ['fau', 'phil', 'wiso', 'med', 'nat', 'tech']


def cmap_fau_blue(cmap_type: Union[str, None]) -> Sequence[str]:
    palette_fau = sns.color_palette(
        ["#001628", "#001F38", "#002747", "#003056", "#003865",
         "#26567C", "#4D7493", "#7392AA", "#99AFC1", "#BFCDD9",
         "#E6EBF0"]
    )
    if cmap_type is '3':
        return palette_fau[1::3]
    elif cmap_type is '2':
        return palette_fau[5::4]
    elif cmap_type is '2_lp':
        return palette_fau[2::5]
    else:
        return palette_fau


def fau_color(key: str) -> str:
    return cmap_fau[_keys_fau.index(key)] or cmap_fau['fau']


def adjust_color(key: str, amount: Optional[float] = 1.5) -> str:
    import colorsys
    import matplotlib.colors as mc
    c = colorsys.rgb_to_hls(*mc.to_rgb(fau_color(key)))
    return mc.to_hex(colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))


def split_data(time_info: Union[pd.Series, Dict[str, Sequence[str]]], dataset: Optional[Dataset] = None,
               df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    data_dict: Dict[str, pd.DataFrame] = {}
    if dataset is None and df is None:
        raise ValueError("Either 'dataset' or 'df' must be specified as parameter!")
    if dataset:
        df = dataset.data_as_df("ecg", index="utc_datetime").tz_localize(tz=utc).tz_convert(tz=tz)
    if isinstance(time_info, pd.Series):
        for name, start, end in zip(time_info.index, np.pad(time_info, (0, 1)), time_info[1:]):
            data_dict[name] = df.between_time(start, end)
    else:
        data_dict = {name: df.between_time(*start_end) for name, start_end in time_info.items()}
    return data_dict


def write_hr_to_excel(ep: 'EcgProcessor', folder: path_t, filename: path_t):
    # ensure pathlib
    folder = Path(folder)
    filename = Path(filename)

    writer = pd.ExcelWriter(folder.joinpath(filename), engine='xlsxwriter')
    for label, df_hr in ep.heart_rate.items():
        df_hr.tz_localize(None).to_excel(writer, sheet_name=label)
    writer.save()


def load_hr_excel(filename: path_t) -> Dict[str, pd.DataFrame]:
    dict_hr = pd.read_excel(filename, index_col="date", sheet_name=None)
    dict_hr = {k: v.tz_localize(tz) for k, v in dict_hr.items()}
    return dict_hr


def write_result_dict_to_csv(result_dict: Dict, filename: path_t):
    df_hrv_concat = pd.concat(result_dict, names=["Subject_ID"])
    if filename.exists():
        df_hrv_old = pd.read_csv(filename, index_col=["Subject_ID", "Phase", "Subphase"])
        df_hrv_concat = df_hrv_concat.combine_first(df_hrv_old).sort_index(level=0)
    df_hrv_concat.reset_index().to_csv(filename, index=False)


def export_figure(fig: plt.Figure, filename: str, base_dir: path_t, use_subfolder: Optional[bool] = True,
                  formats: Sequence[str] = None):
    if formats is None:
        formats = ['pdf']

    # ensure pathlib
    base_dir = Path(base_dir)
    subfolder = [base_dir] * len(formats)

    if use_subfolder:
        subfolder = [base_dir.joinpath(f) for f in formats]
        for folder in subfolder:
            folder.mkdir(exist_ok=True, parents=True)

    for f, subfold in zip(formats, subfolder):
        fig.savefig(subfold.joinpath(filename + '.' + f), transparent=(f is 'pdf'), format=f)
