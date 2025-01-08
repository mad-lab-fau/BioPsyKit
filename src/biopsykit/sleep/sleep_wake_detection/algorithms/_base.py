"""Module for sleep/wake detection base class."""
from biopsykit.utils._types_internal import arr_t
from biopsykit.utils.dtypes import SleepWakeDataFrame


class _SleepWakeBase:
    """Base class for sleep/wake detection algorithms."""

    epoch_length: int = None
    """Epoch length in seconds."""

    def __init__(self, **kwargs):
        self.epoch_length: int = kwargs.get("epoch_length", 60)

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
