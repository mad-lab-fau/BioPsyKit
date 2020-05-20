# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
from pathlib import Path
from typing import TypeVar, Sequence, Optional
import pytz

import seaborn as sns

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')

tz = pytz.timezone('Europe/Berlin')
utc = pytz.timezone('UTC')

cmap_fau = sns.color_palette(["#003865", "#c99313", "#8d1429", "#00b1eb", "#009b77", "#98a4ae"])
_keys_fau = ['fau', 'phil', 'wiso', 'med', 'nat', 'tech']


def cmap_fau_blue(cmap_type: str) -> Sequence[str]:
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
