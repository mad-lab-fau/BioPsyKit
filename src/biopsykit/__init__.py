"""A Python package for the analysis of biopsychological data."""
import platform

import matplotlib
import neurokit2
import numpy as np
import pandas as pd
import pingouin
import scipy

import biopsykit.carwatch_logs  # pylint: disable=unused-import
import biopsykit.classification  # pylint: disable=unused-import
import biopsykit.example_data  # pylint: disable=unused-import
import biopsykit.io  # pylint: disable=unused-import
import biopsykit.metadata  # pylint: disable=unused-import
import biopsykit.plotting  # pylint: disable=unused-import
import biopsykit.protocols  # pylint: disable=unused-import
import biopsykit.questionnaires  # pylint: disable=unused-import
import biopsykit.saliva  # pylint: disable=unused-import
import biopsykit.signals  # pylint: disable=unused-import
import biopsykit.sleep  # pylint: disable=unused-import
import biopsykit.stats  # pylint: disable=unused-import
import biopsykit.utils  # pylint: disable=unused-import

__all__ = [
    "carwatch_logs",
    "classification",
    "example_data",
    "io",
    "metadata",
    "plotting",
    "protocols",
    "questionnaires",
    "saliva",
    "signals",
    "sleep",
    "stats",
    "utils",
]

__version__ = "0.8.0"


def version() -> None:
    """Get the version of BioPsyKit and its core dependencies.

    Examples
    --------
    >>> import biopsykit as bp
    >>>
    >>> bp.version()

    """
    print(
        f"Operating System: {platform.system()} ({platform.architecture()[1]} {platform.architecture()[0]})\n",
        f"- Python: {platform.python_version()}\n",
        f"- BioPsyKit: {__version__}\n\n",
        f"- NumPy: {np.__version__}\n",
        f"- Pandas: {pd.__version__}\n",
        f"- SciPy: {scipy.__version__}\n",
        f"- matplotlib: {matplotlib.__version__}\n",
        f"- NeuroKit2: {neurokit2.__version__}\n",
        f"- pingouin: {pingouin.__version__}\n",
    )
