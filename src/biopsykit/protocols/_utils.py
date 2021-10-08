"""Utility functions for the ``biopsykit.protocols`` module."""
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pandas._libs.lib import is_timedelta_or_timedelta64_array

from biopsykit.utils._datatype_validation_helper import _assert_len_list
from biopsykit.utils.datatype_helper import SalivaMeanSeDataFrame
from biopsykit.utils.exceptions import ValidationError


def _get_sample_times(
    saliva_data: Union[Dict[str, SalivaMeanSeDataFrame], SalivaMeanSeDataFrame],
    sample_times: Union[Sequence[int], Dict[str, Sequence[int]]],
    test_times: Sequence[int],
    sample_times_absolute: Optional[bool] = False,
) -> Union[Sequence[int], Dict[str, Sequence[int]]]:

    if isinstance(sample_times, dict):
        for key in sample_times:
            sample_times[key] = _get_sample_times(saliva_data[key], sample_times[key], test_times)
        return sample_times

    if isinstance(saliva_data, dict):
        sample_times = {}
        for key in saliva_data:
            sample_times[key] = _get_sample_times(saliva_data[key], sample_times, test_times)
        return sample_times

    if sample_times is None:
        sample_times = _get_sample_times_extract(saliva_data)

    _assert_len_list(test_times, 2)
    # ensure numpy
    sample_times = np.array(sample_times)

    if not sample_times_absolute:
        sample_times = _get_sample_times_absolute(sample_times, test_times)
    return list(sample_times)


def _get_sample_times_absolute(sample_times: np.ndarray, test_times: Sequence[int]):
    if is_timedelta_or_timedelta64_array(sample_times.flatten()):
        # convert into minutes
        sample_times_idx = sample_times.astype(float) / (1e9 * 60)
    else:
        sample_times_idx = sample_times
    index_post = np.where(sample_times_idx >= test_times[0])[0]
    sample_times[index_post] = sample_times[index_post] + (test_times[1] - test_times[0])
    return sample_times


def _get_sample_times_extract(saliva_data: pd.DataFrame):
    if "time" in saliva_data.reset_index().columns:
        return saliva_data.reset_index()["time"].unique()
    raise ValueError(
        "No sample times specified! Sample times must either be specified by passing them to the "
        "'sample_times' parameter or to by including them in 'saliva_data' ('time' column)."
    )


def _check_sample_times_match(data: SalivaMeanSeDataFrame, sample_times: Sequence[int]) -> None:
    if not any(len(sample_times) == s for s in [len(data.index.get_level_values("sample").unique()), len(data)]):
        raise ValidationError(
            "Number of saliva samples in data does not match number of samples in 'sample_times'. "
            "Expected {}, got {}.".format(len(data), len(sample_times))
        )
