"""General class for sleep/wake detection."""
from typing import Optional

from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke import ColeKripke
from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke_alternative import ColeKripkeAlternative
from biopsykit.sleep.sleep_wake_detection.algorithms.sadeh import Sadeh
from biopsykit.sleep.sleep_wake_detection.algorithms.sazonov import Sazonov
from biopsykit.sleep.sleep_wake_detection.algorithms.scripps_clinic import ScrippsClinic
from biopsykit.sleep.sleep_wake_detection.algorithms.webster import Webster
from biopsykit.utils._types import arr_t
from biopsykit.utils.datatype_helper import SleepWakeDataFrame, _SleepWakeDataFrame


class SleepWakeDetection:
    """General class for sleep/wake detection."""

    sleep_wake_algo = None

    def __init__(self, algorithm_type: Optional[str] = None, **kwargs):
        """General class for sleep/wake detection.

        This class provides a generalized interface for sleep/wake detection independent of the used algorithm.
        When initializing a new instance the algorithm type can be specified.>

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
        available_sleep_wake_algorithms = {
            "cole_kripke": ColeKripke,
            "sadeh": Sadeh,
            "cole_kripke_alternative": ColeKripkeAlternative,
            "webster": Webster,
            "scripps_clinic": ScrippsClinic,
            "sazonov": Sazonov,
        }

        if algorithm_type is None:
            algorithm_type = "cole_kripke"

        if algorithm_type not in available_sleep_wake_algorithms:
            raise ValueError(
                "Invalid algorithm type for sleep/wake detection! Must be one of {}, got {}.".format(
                    available_sleep_wake_algorithms, algorithm_type
                )
            )

        sleep_wake_cls = available_sleep_wake_algorithms[algorithm_type]

        if (
            sleep_wake_cls is ColeKripke
            or sleep_wake_cls is ColeKripkeAlternative
            or sleep_wake_cls is Webster
            or sleep_wake_cls is ScrippsClinic
        ):
            if "scale_factor" in kwargs:
                self.sleep_wake_algo = sleep_wake_cls(scale_factor=kwargs["scale_factor"])
            else:
                self.sleep_wake_algo = sleep_wake_cls()

        else:
            self.sleep_wake_algo = sleep_wake_cls()

    def predict(self, data: arr_t, **kwargs) -> SleepWakeDataFrame:
        """Apply sleep/wake prediction on input data.

        Parameters
        ----------
        data : array_like
            input data
        **kwargs :
            additional arguments to be passed to the sleep/wake detection algorithm.
            The possible arguments depend on the individual algorithm.

        Returns
        -------
        :obj:`~biopsykit.utils.datatype_helper.SleepWakeDataFrame`
            dataframe with sleep/wake predictions

        """
        return _SleepWakeDataFrame(getattr(self.sleep_wake_algo, "predict")(data, **kwargs))
