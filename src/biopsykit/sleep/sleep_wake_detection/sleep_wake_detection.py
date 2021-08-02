"""General class for sleep/wake detection."""
from typing import Optional

from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke import ColeKripke
from biopsykit.utils._types import arr_t
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class SleepWakeDetection:
    """General class for sleep/wake detection.

    This class provides a generalized interface for sleep/wake detection independent of the used algorithm.
    When initializing a new instance the algorithm type can be specified.

    """

    sleep_wake_algo = None

    def __init__(self, algorithm_type: Optional[str] = None, **kwargs):
        """Initialize new ``SleepWakeDetection`` instance.

        Parameters
        ----------
        algorithm_type : str, optional
            name of sleep/wake detection algorithm to internally use for sleep/wake detection or ``None`` to use
            default algorithm (Cole/Kripke Algorithm, see
            :class:`~biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke.ColeKripke` for further information)
        kwargs
            optional parameters to be passed to the sleep/wake detection algorithm. The possible parameters depend on
            the selected sleep/wake detection algorithm and are passed to the respective class.

        """
        available_sleep_wake_algorithms = {"cole_kripke": ColeKripke}

        if algorithm_type is None:
            algorithm_type = "cole_kripke"

        if algorithm_type not in available_sleep_wake_algorithms:
            raise ValueError(
                "Invalid algorithm type for sleep/wake detection! Must be one of {}, got {}.".format(
                    available_sleep_wake_algorithms, algorithm_type
                )
            )

        sleep_wake_cls = available_sleep_wake_algorithms[algorithm_type]
        if sleep_wake_cls is ColeKripke:
            self.sleep_wake_algo = sleep_wake_cls(scale_factor=kwargs.get("scale_factor", None))

    def predict(self, data: arr_t) -> SleepWakeDataFrame:
        """Apply sleep/wake prediction on input data.

        Parameters
        ----------
        data : array_like
            input data

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

        """
        return _SleepWakeDataFrame(getattr(self.sleep_wake_algo, "predict")(data))
