"""A Python package for the analysis of biopsychological data."""

from biopsykit import example_data

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

__version__ = "0.14.0"


def __getattr__(name):
    if name in __all__:
        import importlib  # noqa: PLC0415

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def version() -> None:
    """Get the version of BioPsyKit and its core dependencies.

    Examples
    --------
    >>> import biopsykit as bp
    >>>
    >>> bp.version()

    """
    import platform  # noqa: PLC0415

    import matplotlib  # noqa: PLC0415
    import neurokit2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import pingouin  # noqa: PLC0415
    import scipy  # noqa: PLC0415

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
