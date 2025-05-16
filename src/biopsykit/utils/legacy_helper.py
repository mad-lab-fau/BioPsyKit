"""Module providing helper functions to convert old BioPsyKit export files to new export formats."""

from biopsykit.io.ecg import load_hr_phase_dict, write_hr_phase_dict
from biopsykit.utils._types_internal import path_t

__all__ = ["legacy_convert_hr_phase_dict"]

from biopsykit.utils.dtypes import HeartRatePhaseDict


def legacy_convert_hr_phase_dict(file_path: path_t, export: bool | None = True) -> HeartRatePhaseDict | None:
    """Legacy conversion for :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict`.

    Legacy conversion includes changing the column name from "ECG_Rate" into "Heart_Rate".


    Parameters
    ----------
    file_path : :class:`~pathlib.Path` or str
        path to file
    export : bool, optional
        ``True`` to directly export the converted
        :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict` under the same file name,
        ``False`` to return the converted :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict`


    Returns
    -------
    :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict`
        The converted :obj:`~biopsykit.utils.dtypes.HeartRatePhaseDict`, if ``export`` is ``False``

    """
    hr_phase_dict = load_hr_phase_dict(file_path, assert_format=False)
    for key, df in hr_phase_dict.items():
        df.columns = ["Heart_Rate"]
        hr_phase_dict[key] = df
    if export:
        write_hr_phase_dict(hr_phase_dict, file_path)
        return None
    return hr_phase_dict
