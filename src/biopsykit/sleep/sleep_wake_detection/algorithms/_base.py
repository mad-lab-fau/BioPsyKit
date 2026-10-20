"""Module for sleep/wake detection base class."""

from biopsykit.sleep._epoch import EPOCH_LENGTH_ATTR, resolve_epoch_length, validate_epoch_length
from biopsykit.utils._types_internal import arr_t
from biopsykit.utils.dtypes import SleepWakeDataFrame, _SleepWakeDataFrame


class _SleepWakeBase:
    """Base class for sleep/wake detection algorithms."""

    epoch_length: float | None = None
    """Epoch length in seconds."""

    default_epoch_length = 60

    def __init__(self, **kwargs):
        self.epoch_length = validate_epoch_length(kwargs.get("epoch_length", self.default_epoch_length))

    def _prediction_frame(self, scores, index, epoch_length):
        result = _SleepWakeDataFrame(scores, index=index, columns=["sleep_wake"])
        result.attrs[EPOCH_LENGTH_ATTR] = resolve_epoch_length(result, epoch_length)
        return result

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
