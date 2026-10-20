"""General class for sleep/wake detection."""

from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke import ColeKripke
from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke_old import ColeKripkeOld
from biopsykit.sleep.sleep_wake_detection.algorithms.sadeh import Sadeh
from biopsykit.sleep.sleep_wake_detection.algorithms.sazonov import Sazonov
from biopsykit.sleep.sleep_wake_detection.algorithms.scripps_clinic import ScrippsClinic
from biopsykit.sleep.sleep_wake_detection.algorithms.webster import Webster
from biopsykit.utils._types_internal import arr_t
from biopsykit.utils.dtypes import SleepWakeDataFrame


class SleepWakeDetection:
    """General class for sleep/wake detection."""

    sleep_wake_algo = None

    def __init__(self, algorithm_type: str | None = None, **kwargs):
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
            the selected sleep/wake detection algorithm and are passed to the respective class and prediction call.
            ``epoch_length`` is in seconds. Classifier-specific epoch/kernel restrictions still apply.

        """
        available_sleep_wake_algorithms = {
            "cole_kripke": ColeKripke,
            "sadeh": Sadeh,
            "cole_kripke_old": ColeKripkeOld,
            "webster": Webster,
            "scripps_clinic": ScrippsClinic,
            "sazonov": Sazonov,
        }

        if algorithm_type is None:
            algorithm_type = "cole_kripke"

        if algorithm_type not in available_sleep_wake_algorithms:
            raise ValueError(
                f"Invalid algorithm type for sleep/wake detection! "
                f"Must be one of {available_sleep_wake_algorithms}, got {algorithm_type}."
            )

        sleep_wake_cls = available_sleep_wake_algorithms[algorithm_type]

        self.sleep_wake_algo = sleep_wake_cls(**kwargs)
        self._prediction_kwargs = dict(kwargs)

    @property
    def epoch_length(self) -> float:
        """Configured epoch duration in seconds, also stored in prediction attrs['epoch_length_s']."""
        return self.sleep_wake_algo.epoch_length

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
        :obj:`~biopsykit.utils.dtypes.SleepWakeDataFrame`
            dataframe with sleep/wake predictions. Each timestamp is the start of an epoch; duration in seconds
            is stored in ``attrs['epoch_length_s']``. Integer indices count epochs. Keyword overrides apply only
            to this prediction and do not modify the configured epoch length.

        """
        return self.sleep_wake_algo.predict(data, **(self._prediction_kwargs | kwargs))
