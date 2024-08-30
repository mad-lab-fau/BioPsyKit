"""Module for loading and processing Task Force Monitor (TFM) data."""
from typing import Dict, Optional

import pandas as pd
from biopsykit.utils._types import path_t
from scipy.io import loadmat


class TFMDataset:
    """Class for loading and processing Task Force Monitor (TFM) data."""

    CHANNEL_MAPPING = {"ecg_1": "rawECG1", "ecg_2": "rawECG2", "icg_der": "rawICG"}
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

    @classmethod
    def from_mat_file(
        cls,
        path: path_t,
        # channel_mapping: Optional[Dict[str, str]] = None,
        tz: Optional[str] = "Europe/Berlin",
    ):
        """Load a TFM dataset from a .mat file.

        Parameters
        ----------
        path : str or :class:`~pathlib.Path`
            Path to the .mat file.
        tz : str, optional
            Timezone of the data. Default: "Europe/Berlin"

        """
        data = loadmat(path, struct_as_record=False, squeeze_me=True)
        data_raw = data["RAW_SIGNALS"]
        # keys = [s for s in dir(data_raw) if not s.startswith("_")]
        # print(keys)
        data_dict = {key: getattr(data_raw, value) for key, value in cls.CHANNEL_MAPPING.items()}
        return cls(data_dict=data_dict, tz=tz, sampling_rate_dict={})

    def data_as_df(self) -> Dict[str, pd.DataFrame]:
        """Return the TFM data as a dictionary of pandas DataFrames.

        Returns
        -------
        dict
            Dictionary containing the TFM data as pandas DataFrames. Keys are channel names, values are the dataframes.

        """
        return self._data
