"""Module for loading and processing Task Force Monitor (TFM) data."""
from typing import ClassVar, Optional

import pandas as pd
from scipy.io import loadmat

from biopsykit.utils._types import path_t


class TFMDataset:
    """Class for loading and processing Task Force Monitor (TFM) data."""

    CHANNEL_MAPPING: ClassVar[dict[str, str]] = {"ecg_1": "rawECG1", "ecg_2": "rawECG2", "icg_der": "rawICG"}
    _tz: str

    def __init__(
        self, data_dict: dict[str, pd.DataFrame], sampling_rate_dict: dict[str, float], tz: Optional[str] = None
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
        """Load TFM data from a .mat file.

        Parameters
        ----------
        path : str
            Path to the .mat file containing the TFM data.
        tz : str, optional
            Timezone of the data. Default: "Europe/Berlin"

        Returns
        -------
        :class:`~biopsykit.io.tfm.TFMDataset`
            TFM dataset object.

        """
        data = loadmat(path, struct_as_record=False, squeeze_me=True)
        data_raw = data["RAW_SIGNALS"]
        # keys = [s for s in dir(data_raw) if not s.startswith("_")]
        # print(keys)
        data_dict = {key: getattr(data_raw, value) for key, value in cls.CHANNEL_MAPPING.items()}
        return cls(data_dict=data_dict, tz=tz, sampling_rate_dict={})

    def data_as_df(self):
        """Return data as a single pandas DataFrame.

        Returns
        -------
        :class:`~pandas.DataFrame`
            Data as a single pandas DataFrame.

        """
        return self._data
