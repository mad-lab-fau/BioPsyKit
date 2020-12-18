from typing import Union, Dict, Sequence, Optional, Tuple

import pandas as pd

from biopsykit.protocols import base


class GenericProtocol(base.BaseProtocol):

    def __init__(self, name: str):
        super().__init__(name)

    def concat_phase_dict(
            self,
            dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
            phases: Sequence[str]
    ) -> Dict[str, pd.DataFrame]:
        return super()._concat_phase_dict(dict_hr_subject, phases)

    def split_subphases(
            self,
            data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
            subphase_names: Sequence[str], subphase_times: Sequence[Tuple[int, int]],
            is_group_dict: Optional[bool] = False
    ) -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
        return super()._split_subphases(data, subphase_names, subphase_times, is_group_dict)

    def split_groups(
            self,
            phase_dict: Dict[str, pd.DataFrame],
            condition_dict: Dict[str, Sequence[str]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        return super()._split_groups(phase_dict, condition_dict)

    def mean_se_subphases(
            self,
            data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
            subphases: Sequence[str],
            is_group_dict: Optional[bool] = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return super()._mean_se_subphases(data, subphases, is_group_dict)
