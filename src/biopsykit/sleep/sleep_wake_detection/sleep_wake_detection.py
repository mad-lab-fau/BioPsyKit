from typing import Optional, Union

import pandas as pd
import numpy as np

from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke import ColeKripke
from biopsykit.sleep.sleep_wake_detection.algorithms.sadeh import Sadeh
from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke_alternative import ColeKripkeAlternative
from biopsykit.sleep.sleep_wake_detection.algorithms.webster import Webster
from biopsykit.sleep.sleep_wake_detection.algorithms.scripps_clinic import ScrippsClinic
from biopsykit.sleep.sleep_wake_detection.algorithms.perez_pozuelo import PerezPozuelo
from biopsykit.sleep.sleep_wake_detection.algorithms.sazonov import Sazonov




class SleepWakeDetection:
    sleep_wake_algo = None

    def __init__(self, algorithm_type: Optional[str] = "default", **kwargs):
        available_sleep_wake_algorithms = {"cole_kripke": ColeKripke, "sadeh": Sadeh, "cole_kripke_alternative": ColeKripkeAlternative,
                                           "webster": Webster, "scripps_clinic": ScrippsClinic, "perez_pozuelo": PerezPozuelo,
                                           "sazonov": Sazonov}

        if algorithm_type == "default":
            algorithm_type = "cole_kripke"

        if algorithm_type not in available_sleep_wake_algorithms:
            raise ValueError(
                "Invalid algorithm type for sleep/wake detection! Must be one of {}, got {}.".format(
                    available_sleep_wake_algorithms, algorithm_type
                )
            )

        sleep_wake_cls = available_sleep_wake_algorithms[algorithm_type]

        if sleep_wake_cls is ColeKripke or ColeKripkeAlternative or Webster or ScrippsClinic:
            if "scale_factor" in kwargs:
                self.sleep_wake_algo = sleep_wake_cls(scale_factor=kwargs["scale_factor"])
            else:
                self.sleep_wake_algo = sleep_wake_cls()

        else:
            self.sleep_wake_algo = sleep_wake_cls()





    def predict(self, data: Union[pd.DataFrame, np.array]) -> Union[np.array, pd.DataFrame]:
        return getattr(self.sleep_wake_algo, "predict")(data)
