"""Module for sleep/wake detection base class."""
from biopsykit.utils._types import arr_t
from biopsykit.utils.datatype_helper import SleepWakeDataFrame


class _SleepWakeBase:
    """Base class for sleep/wake detection algorithms."""

    def __init__(self, **kwargs):
        pass

    def fit(self, data: arr_t, **kwargs):
        """Fit sleep/wake detection algorithm to input data.

        .. note::
            Algorithms that do not have to (re)fit a ML model before sleep/wake prediction, such as rule-based
            algorithms, will internally bypass this method as the ``fit`` step is not needed.

        Parameters
        ----------
        data : array_like
            input data

        """
        raise NotImplementedError("Needs to be implemented by child class.")

    def predict(self, data: arr_t, **kwargs) -> SleepWakeDataFrame:
        """Apply sleep/wake prediction algorithm on input data.

        Parameters
        ----------
        data : array_like
            input data

        """
        raise NotImplementedError("Needs to be implemented by child class.")
