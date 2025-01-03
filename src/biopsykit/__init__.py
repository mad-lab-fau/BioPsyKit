"""A Python package for the analysis of biopsychological data."""

from biopsykit import (
    classification,
    example_data,
    io,
    metadata,
    plotting,
    protocols,
    questionnaires,
    saliva,
    signals,
    sleep,
    stats,
    utils,
)

__all__ = [
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

__version__ = "0.11.0"


def version() -> None:
    """Get the version of BioPsyKit and its core dependencies.

    Examples
    --------
    >>> import biopsykit as bp
    >>>
    >>> bp.version()

    """
    import platform

    import matplotlib
    import neurokit2
    import numpy as np
    import pandas as pd
    import pingouin
    import scipy

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
