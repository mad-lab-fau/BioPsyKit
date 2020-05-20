# -*- coding: utf-8 -*-
"""Set of helper functions used throughout the library.

@author: Robert Richer, Arne Küderle
"""
from pathlib import Path
from typing import TypeVar

import pytz

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')

tz = pytz.timezone('Europe/Berlin')
utc = pytz.timezone('UTC')
