"""Some custom helper types to make type hints and type checking easier.

For user facing type declarations, please see :py:func:`biopsykit.utils.datatype_helper`.
"""

from pathlib import Path
from typing import TypeVar

path_t = TypeVar("path_t", str, Path)  # noqa: invalid-name
T = TypeVar("T")
