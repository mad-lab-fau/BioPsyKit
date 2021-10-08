"""Module providing helper functions to convert old BioPsyKit export files to new export formats."""
from typing import Optional

from biopsykit.io.ecg import load_hr_phase_dict, write_hr_phase_dict
from biopsykit.utils._types import path_t

__all__ = ["legacy_convert_hr_phase_dict"]

from biopsykit.utils.datatype_helper import HeartRatePhaseDict


def legacy_convert_hr_phase_dict(file_path: path_t, export: Optional[bool] = True) -> Optional[HeartRatePhaseDict]:
    """Legacy conversion for :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`.

    Legacy conversion includes changing the column name from "ECG_Rate" into "Heart_Rate".


    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    export : bool, optional
        ``True`` to directly export the converted
        :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict` under the same file name,
        ``False`` to return the converted :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`


    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
        The converted :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`, if ``export`` is ``False``

    """
    hr_phase_dict = load_hr_phase_dict(file_path, assert_format=False)
    for key, df in hr_phase_dict.items():
        df.columns = ["Heart_Rate"]
        hr_phase_dict[key] = df
    if export:
        write_hr_phase_dict(hr_phase_dict, file_path)
        return None
    return hr_phase_dict
