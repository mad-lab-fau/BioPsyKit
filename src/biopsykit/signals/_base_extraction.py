from typing import Literal, get_args

import pandas as pd
from tpcp import Algorithm

HANDLE_MISSING_EVENTS = Literal["raise", "warn", "ignore"]


class CanHandleMissingEventsMixin(Algorithm):
    def __init__(self, handle_missing_events: HANDLE_MISSING_EVENTS = "warn"):
        """Mixin class to handle missing events in the input dataframes.

        Parameters
        ----------
        handle_missing_events : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. If "warn", a warning is raised if missing data is found.
            If "raise", an exception is raised if missing data is found. If "ignore", missing data is ignored.
            Default: "warn"

        """
        self.handle_missing_events = handle_missing_events

    def _check_valid_missing_handling(self):
        if self.handle_missing_events not in get_args(HANDLE_MISSING_EVENTS):
            raise ValueError(
                f"Invalid value '{self.handle_missing_events}' for 'handle_missing_events'. "
                f"Must be one of {get_args(HANDLE_MISSING_EVENTS)}."
            )


class BaseExtraction(Algorithm):
    """Base class which defines the interface for all fiducial point extraction algorithms.

    Results:
        points_ : saves positions of extracted points in pd.DataFrame
    """

    _action_methods = "extract"

    # results
    points_: pd.DataFrame
