from typing import Sequence, Union, Dict

import numpy as np
from biopsykit.utils._datatype_validation_helper import _assert_len_list
from biopsykit.utils.datatype_helper import SalivaMeanSeDataFrame
from biopsykit.utils.exceptions import ValidationError


def _get_sample_times(
    saliva_data: Union[Dict[str, SalivaMeanSeDataFrame], SalivaMeanSeDataFrame],
    sample_times: Union[Sequence[int], Dict[str, Sequence[int]]],
    test_times: Sequence[int],
) -> Union[Sequence[int], Dict[str, Sequence[int]]]:
    if isinstance(sample_times, dict):
        for key in sample_times:
            sample_times[key] = _get_sample_times(saliva_data[key], sample_times[key], test_times)
        return sample_times

    if sample_times is None:
        if "time" in saliva_data.index.names:
            sample_times = saliva_data.index.get_level_values("time").unique()
        else:
            raise ValueError(
                "No sample times specified! Sample times must either be specified by passing them to the "
                "'sample_times' parameter or to by including them in 'saliva_data' ('time' index level)."
            )

    _assert_len_list(test_times, 2)
    # ensure numpy
    sample_times = np.array(sample_times)

    index_post = np.where(sample_times >= 0)[0]
    sample_times[index_post] = sample_times[index_post] + test_times[1]
    return list(sample_times)


def _check_sample_times_match(data: SalivaMeanSeDataFrame, sample_times: Sequence[int]) -> None:
    if len(data.index.get_level_values("sample").unique()) != len(sample_times):
        raise ValidationError(
            "Number of saliva samples in data does not match number of samples in 'sample_times'. "
            "Expected {}, got {}.".format(len(data), len(sample_times))
        )
