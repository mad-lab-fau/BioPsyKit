from typing import Optional, Union

import pandas as pd
import numpy as np

from biopsykit.sleep.sleep_wake_detection.algorithms.cole_kripke import ColeKripke
from biopsykit.sleep.sleep_wake_detection.algorithms.sadeh import Sadeh

class SleepWakeDetection:
    sleep_wake_algo = None

    def __init__(self, algorithm_type: Optional[str] = "default", **kw_args):
        available_sleep_wake_algorithms = {"cole_kripke": ColeKripke, "sadeh": Sadeh}

        if algorithm_type == "default":
            algorithm_type = "cole_kripke"

        if algorithm_type not in available_sleep_wake_algorithms:
            raise ValueError(
                "Invalid algorithm type for sleep/wake detection! Must be one of {}, got {}.".format(
                    available_sleep_wake_algorithms, algorithm_type
                )
            )

        sleep_wake_cls = available_sleep_wake_algorithms[algorithm_type]
        if sleep_wake_cls is ColeKripke:
            if "scale_factor" in kw_args:
                self.sleep_wake_algo = sleep_wake_cls(kw_args["scale_factor"])
            else:
                self.sleep_wake_algo = sleep_wake_cls()
        elif sleep_wake_cls is Sadeh:
            self.sleep_wake_algo = sleep_wake_cls()

    def predict(self, data: Union[pd.DataFrame, np.array]) -> Union[np.array, pd.DataFrame]:
        return getattr(self.sleep_wake_algo, "predict")(data)
