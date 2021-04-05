"""This module provides helper functions to convert old BioPsyKit export files to new export formats."""
from typing import Optional

from biopsykit.utils._types import path_t

from biopsykit.io.ecg import load_hr_subject_dict, write_hr_subject_dict

__all__ = ["legacy_convert_hr_subject_dict"]

from biopsykit.utils.datatype_helper import HeartRateSubjectDict


def legacy_convert_hr_subject_dict(file_path: path_t, export: Optional[bool] = True) -> Optional[HeartRateSubjectDict]:
    """Legacy conversion for :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`.

    Legacy conversion includes changing the column name from "ECG_Rate" into "Heart_Rate".

    Parameters
    ----------
    file_path : :any:`pathlib.Path` or str
        path to file
    export : bool, optional
        ``True`` to directly export the converted
        :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict` under the same file name,
        ``False`` to return the converted :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`

    Returns
    -------
    :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`
        The converted :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`, if ``export`` is ``False``

    """
    hr_subject_dict = load_hr_subject_dict(file_path, assert_format=False)
    for key, df in hr_subject_dict.items():
        df.columns = ["Heart_Rate"]
        hr_subject_dict[key] = df
    if export:
        write_hr_subject_dict(hr_subject_dict, file_path)
        return None
    return hr_subject_dict
