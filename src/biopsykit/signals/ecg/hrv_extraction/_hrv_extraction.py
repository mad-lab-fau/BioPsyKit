from collections.abc import Callable, Sequence
from typing import ClassVar

import neurokit2 as nk
import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self


class HrvExtraction(Algorithm):
    _action_methods = "extract"

    HRV_TYPES: ClassVar[list[str]] = ["hrv_time", "hrv_frequency", "hrv_nonlinear"]

    HRV_METHODS: ClassVar[dict[str, Callable]] = {
        "hrv_time": nk.hrv_time,
        "hrv_nonlinear": nk.hrv_nonlinear,
        "hrv_frequency": nk.hrv_frequency,
    }

    hrv_types: Sequence[str]

    hrv_extracted_: pd.DataFrame

    def __init__(self, hrv_types: str | Sequence[str] = "default"):
        """Initialize new ``HrvExtraction`` algorithm instance.

        This algorithm extracts heart rate variability (HRV) features from the R-peak data using the NeuroKit2 library.

        Parameters
        ----------
        hrv_types : str or sequence of str, optional
            The types of HRV features to extract. Options are:
                * "default": Extracts the default HRV features (time- and frequency-domain).
                * "all": Extracts all available HRV features (time, frequency, and nonlinear).
                * "time" or "hrv_time": Extracts time-domain HRV features.
                * "frequency" or "hrv_frequency": Extracts frequency-domain HRV features.
                * "nonlinear" or "hrv_nonlinear": Extracts nonlinear HRV features.
            Default: "default"

        """
        self.hrv_types = hrv_types
        super().__init__()

    def _sanitize_hrv_types(self):
        """Sanitize the hrv_types parameter to ensure it is a list of valid types."""
        if isinstance(self.hrv_types, str):
            if self.hrv_types == "default":
                self.hrv_types = ["hrv_time", "hrv_frequency"]
            elif self.hrv_types == "all":
                self.hrv_types = self.HRV_TYPES
            else:
                self.hrv_types = [self.hrv_types]
        elif not isinstance(self.hrv_types, Sequence):
            raise TypeError("hrv_types must be a string or a sequence of strings.")

        self.hrv_types = [
            f"hrv_{hrv_type}" if not hrv_type.startswith("hrv_") else hrv_type for hrv_type in self.hrv_types
        ]
        for hrv_type in self.hrv_types:
            if hrv_type not in self.HRV_TYPES:
                raise TypeError(f"Invalid hrv_type: {hrv_type}. Must be one of {self.HRV_TYPES}.")

    def extract(self, *, rpeaks: pd.DataFrame, sampling_rate_hz: float) -> Self:
        """Extract heart rate variability (HRV) features from the R-peak data.

        Parameters
        ----------
        rpeaks : :class:`~pandas.DataFrame`
            The R-peak data. The DataFrame contains the R-peak locations, with the index
            representing the heartbeat IDs and the column "r_peak_sample" containing the R-peak samples.
        sampling_rate_hz : float
            The sampling rate of the ECG signal in Hz.

        """
        self._sanitize_hrv_types()

        hrv_methods = {key: self.HRV_METHODS[key] for key in self.hrv_types if key in self.HRV_METHODS}

        # compute HRV parameters
        hrv_data = [hrv_methods[key](rpeaks["r_peak_sample"], sampling_rate=sampling_rate_hz) for key in hrv_methods]
        hrv_data = pd.concat(hrv_data, axis=1)

        self.hrv_extracted_ = hrv_data
        return self
