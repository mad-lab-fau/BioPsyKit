"""A Python package for the analysis of biopsychological data."""
import platform

import matplotlib
import neurokit2
import numpy as np
import pandas as pd
import pingouin
import scipy

import biopsykit.carwatch_logs  # noqa: F401
import biopsykit.classification  # noqa: F401
import biopsykit.example_data  # noqa: F401
import biopsykit.io  # noqa: F401
import biopsykit.metadata  # noqa: F401
import biopsykit.plotting  # noqa: F401
import biopsykit.protocols  # noqa: F401
import biopsykit.questionnaires  # noqa: F401
import biopsykit.saliva  # noqa: F401
import biopsykit.signals  # noqa: F401
import biopsykit.sleep  # noqa: F401
import biopsykit.stats  # noqa: F401
import biopsykit.utils  # noqa: F401

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

__version__ = "0.5.1"


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
