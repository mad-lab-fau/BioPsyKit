from typing import Dict, Sequence, Union, Tuple, Optional

import pandas as pd
import numpy as np


def concat_phase_dict(dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
                      phases: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """
    Rearranges the 'HR subject dict' (see `utils.load_hr_excel_all_subjects`) into 'Phase dict', i.e. a dictionary with
    one dataframe per Stress Test phase where each dataframe contains column-wise HR data for all subjects.

    The **output** format will be the following:

    { <"Stress_Phase"> : hr_dataframe, 1 subject per column }

    Parameters
    ----------
    dict_hr_subject : dict
        'HR subject dict', i.e. a nested dict with heart rate data per Stress Test phase and subject
    phases : list
        list of Stress Test phases. E.g. for MIST this would be the three MIST phases ['MIST1', 'MIST2', 'MIST3'],
        for TSST this would be ['Preparation', 'Speaking', 'ArithmeticTask']

    Returns
    -------
    dict
        'Phase dict', i.e. a dict with heart rate data of all subjects per Stress Test phase

    """

    dict_phase: Dict[str, pd.DataFrame] = {key: pd.DataFrame(columns=list(dict_hr_subject.keys())) for key in phases}
    for subj in dict_hr_subject:
        dict_bl = dict_hr_subject[subj]
        for phase in phases:
            dict_phase[phase][subj] = dict_bl[phase]['ECG_Rate']

    return dict_phase


def split_subphases(data: Union[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]],
                    subphase_names: Sequence[str], subphase_times: Sequence[Tuple[int, int]],
                    is_group_dict: Optional[bool] = False) \
        -> Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """
    Splits a `Phase dict` (or a dict of such, in case of multiple groups, see ``bp.protocols.utils.concat_dicts``)
    into a `Subphase dict` (see below for further explanation).

    The **input** is a `Phase dict`, i.e. a dictionary with heart rate data per Stress Test phase
    in the following format:

    { <"Phase"> : HR_dataframe, 1 subject per column }

    If multiple groups are present, then the expected input is nested, i.e. a dict of 'Phase dicts',
    with one entry per group.

    The **output** is a `Subphase dict`, i.e. a nested dictionary with heart rate data per Subphase in the
    following format:

    { <"Phase"> : { <"Subphase"> : HR_dataframe, 1 subject per column } }

    If multiple groups are present, then the output is nested, i.e. a dict of 'Subphase dicts',
    with one entry per group.


    Parameters
    ----------
    data : dict
        'Phase dict' or nested dict of 'Phase dicts' if `is_group_dict` is ``True``
    subphase_names : list
        List with names of subphases
    subphase_times : list
        List with start and end times of each subphase in seconds
    is_group_dict : bool, optional
        ``True`` if group dict was passed, ``False`` otherwise. Default: ``False``

    Returns
    -------
    dict
        'Subphase dict' with course of HR data per Stress Test phase, subphase and subject, respectively or
        nested dict of 'Subphase dicts' if `is_group_dict` is ``True``

    """
    if is_group_dict:
        # recursively call this function for each group
        return {group: split_subphases(dict_group) for group, dict_group in data.items()}
    else:
        phase_dict = {}
        # split data into subphases for each MIST phase
        for phase, df in data.items():
            phase_dict[phase] = {subph: df[start:end] for subph, (start, end) in zip(subphase_names, subphase_times)}
        return phase_dict


def split_groups(phase_dict: Dict[str, pd.DataFrame],
                 condition_dict: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Splits 'Phase dict' into group dict, i.e. one 'Phase dict' per group.

    Parameters
    ----------
    phase_dict : dict
        'Phase dict' to be split in groups. See ``utils.concat_phase_dict`` for further information
    condition_dict : dict
        dictionary of group membership. Keys are the different groups, values are lists of subject IDs that belong
        to the respective group

    Returns
    -------
    dict
        nested group dict with one 'Phase dict' per group

    """
    return {
        condition: {key: df[condition_dict[condition]] for key, df in phase_dict.items()} for condition
        in condition_dict.keys()
    }


def hr_course(data: Union[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]],
              subphases: Sequence[str],
              is_group_dict: Optional[bool] = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Computes the heart rate mean and standard error per subphase over all subjects.

    As input either 1. a 'Subphase dict' (for only one group) or 2. a dict of 'Subphase dict', one dict per
    group (for multiple groups, if `is_group_dict` is ``True``) can be passed
    (see ``utils.split_subphases`` for more explanation). Both dictionaries are outputs from
    `utils.split_subphases``.

    The output is a 'mse dataframe' (or a dict of such, in case of multiple groups), a pandas dataframe with:
        * columns: ['mean', 'se'] for mean and standard error
        * rows: MultiIndex with level 0 = Phases and level 1 = Subphases.

    The dict structure should like the following:
        (a) { "<Phase>" : { "<Subphase>" : heart rate dataframe, 1 subject per column } }
        (b) { "<Group>" : { <"Phase"> : { "<Subphase>" : heart rate dataframe, 1 subject per column } } }

    Parameters
    ----------
    data : dict
        nested dictionary containing heart rate data.
    subphases : list, optional
        list of subphase names or ``None`` to use default subphase names. Default: ``None``
    is_group_dict : boolean, optional
        ``True`` if `data` is a group dict, i.e. contains dictionaries for multiple groups, ``False`` otherwise.
        Default: ``False``

    Returns
    -------
    dict or pd.DataFrame
        'mse dataframe' or dict of 'mse dataframes', one dataframe per group, if `group_dict` is ``True``.
    """

    if is_group_dict:
        return {group: hr_course(dict_group, subphases) for group, dict_group in data.items()}
    else:
        mean_hr = {phase: pd.DataFrame({subph: df[subph].mean() for subph in subphases}) for phase, df in data.items()}
        df_mist = pd.concat(mean_hr.values(), axis=1, keys=mean_hr.keys())
        return pd.concat([df_mist.mean(), df_mist.std() / np.sqrt(df_mist.shape[0])], axis=1, keys=["mean", "se"])
