"""Sleep/Wake detection using the *Perez/Pozuelo Algorithm*."""
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
from biopsykit.utils.array_handling import sanitize_input_1d, sliding_window
from biopsykit.utils._types import arr_t
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class PerezPozuelo(_SleepWakeBase):
    """Class representing the *Perez/Pozuelo Algorithm* for sleep/wake detection based on heart rate data."""

    def __init__(self, **kwargs):
        """Class representing the *Perez/Pozuelo Algorithm* for sleep/wake detection based on heart rate data.

        Parameters
        ----------
        **kwargs
            additional arguments to be passed to the algorithm, such as:

            * ``quantile`` (``float``)
              add documentation. Default: 0.85
            * ``minimum_length`` (``int``)
              add documentation. Default: 30
            * ``gap`` (``int``)
              add documentation. Default: 60

        References
        ----------
        add reference

        """
        quantile = kwargs.get("quantile", 0.85)
        self.quantile: float = quantile
        """add documentation"""

        minimum_length = kwargs.get("minimum_length", 30)
        self.minimum_length: int = minimum_length
        """add documentation"""

        gap = kwargs.get("gap", 60)
        self.gap: int = gap
        """add documentation"""

        super().__init__(**kwargs)

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
        return

    def predict(self, data: arr_t, **kwargs) -> SleepWakeDataFrame:
        """Apply sleep/wake prediction algorithm on input data.

        Parameters
        ----------
        data : array_like
            array with heart rate values

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

        """
        index = None
        if isinstance(data, pd.DataFrame):
            index = data.index
            data = sanitize_input_1d(data)

        dist = ECDF(data)

        heart_rate = dist.x[np.searchsorted(dist.y, self.quantile, side="left")]

        preprocessed = data
        preprocessed[data <= heart_rate] = 0  # wake = 0
        preprocessed[data > heart_rate] = 1  # sleep = 1

        scores = sliding_window(preprocessed, window_samples=10, overlap_samples=9)

        if index is not None:
            scores = pd.DataFrame(scores, index=index, columns=["sleep_wake"])

        return _SleepWakeDataFrame(scores)
