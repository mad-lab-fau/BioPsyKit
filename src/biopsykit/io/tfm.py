from typing import Dict, Optional

import pandas as pd


class TFMDataset:
    """Class for loading and processing Task Force Monitor (TFM) data."""

    CHANNEL_MAPPING = {}

    _tz: str

    def __init__(
        self, data_dict: Dict[str, pd.DataFrame], sampling_rate_dict: Dict[str, float], tz: Optional[str] = None
    ):
        """Initialize a TFM dataset.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing TFM data. Keys are channel names, values are dataframes with the TFM data.
        tz : str, optional
            Timezone of the data. Default: None

        """
        self._data = data_dict
        self._sampling_rate = sampling_rate_dict
        for name, data in data_dict.items():
            setattr(self, name, data)
        for name, sampling_rate in sampling_rate_dict.items():
            setattr(self, f"sampling_rate_hz_{name}", sampling_rate)
        setattr(self, "channels", list(self._data.keys()))

        self._tz = tz
