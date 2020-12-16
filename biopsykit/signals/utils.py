from typing import Union, Tuple, Optional, Dict, Sequence

import pandas as pd
import numpy as np
import neurokit2 as nk
from numpy.lib.stride_tricks import as_strided

from biopsykit.utils import sanitize_input_1d


def interpolate_sec(data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Interpolates input data to a frequency of 1 Hz.

    *Note*: This function requires the index of the dataframe or series to be a datetime index.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        data to interpolate. Index of data needs to be 'pd.DateTimeIndex'

    Returns
    -------
    pd.DataFrame
        dataframe with data interpolated to seconds


    Raises
    ------
    ValueError
        if no dataframe or series is passed, or if the dataframe/series has no datetime index

    """

    from scipy import interpolate
    if isinstance(data, pd.DataFrame):
        column_name = data.columns
    elif isinstance(data, pd.Series):
        column_name = [data.name]
    else:
        raise ValueError("Only 'pd.DataFrame' or 'pd.Series' allowed as input!")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index of data needs to be 'pd.DateTimeIndex'!")

    x_old = np.array((data.index - data.index[0]).total_seconds())
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    data = sanitize_input_1d(data)
    interpol_f = interpolate.interp1d(x=x_old, y=data, fill_value="extrapolate")
    x_new = pd.Index(x_new, name="Time")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=column_name)


def interpolate_dict_sec(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Interpolates all data in the dictionary to 1Hz data (see `interpolate_sec` for further information).

    Parameters
    ----------
    data_dict : dict
        nested data dict with heart rate data

    Returns
    -------
    dict
        nested data dict with heart rate data interpolated to seconds
    """

    return {
        subject_id: {
            phase: interpolate_sec(df_hr) for phase, df_hr in dict_hr.items()
        } for subject_id, dict_hr in data_dict.items()
    }


def interpolate_and_cut(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_dict = interpolate_dict_sec(data_dict)
    durations = np.array([[len(df) for phase, df in dict_hr.items()] for dict_hr in data_dict.values()])
    phase_names = np.array([list(val.keys()) for key, val in data_dict.items()])
    if (phase_names[0] == phase_names).all():
        phase_names = phase_names[0]
    else:
        raise ValueError("Phases are not the same for all subjects!")

    # minimal duration of each Phase
    min_dur = {phase: dur for phase, dur in zip(phase_names, np.min(durations, axis=0))}

    for subject_id, dict_hr in data_dict.items():
        dict_cut = {}
        for phase in phase_names:
            dict_cut[phase] = dict_hr[phase][0:min_dur[phase]]
        data_dict[subject_id] = dict_cut

    return data_dict


def concat_phase_dict(dict_hr_subject: Dict[str, Dict[str, pd.DataFrame]],
                      phases: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """
    Rearranges a 'HR subject dict' (see ``utils.load_hr_excel_all_subjects``) into a 'Phase dict', i.e. a dictionary
    with one dataframe per Phase where each dataframe contains column-wise HR data for all subjects.

    The **output** format will be the following:

    { <"Phase"> : hr_dataframe, 1 subject per column }

    E.g., see ``biopsykit.protocols.mist.concat_phase_dict()`` for further information.

    Parameters
    ----------
    dict_hr_subject : dict
        'HR subject dict', i.e. a nested dict with heart rate data per phase and subject
    phases : list
        list of phase names

    Returns
    -------
    dict
        'Phase dict', i.e. a dict with heart rate data of all subjects per phase

    """

    dict_phase: Dict[str, pd.DataFrame] = {key: pd.DataFrame(columns=list(dict_hr_subject.keys())) for key in
                                           phases}
    for subj in dict_hr_subject:
        dict_bl = dict_hr_subject[subj]
        for phase in phases:
            dict_phase[phase][subj] = dict_bl[phase]['ECG_Rate']

    return dict_phase


def find_extrema_in_radius(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                           indices: Union[pd.DataFrame, pd.Series, np.ndarray], radius: Union[int, Tuple[int, int]],
                           extrema_type: Optional[str] = "min") -> np.ndarray:
    """
    Finds extrema values (min or max) within a given radius.

    Parameters
    ----------
    data : array_like
        input data
    indices : array_like
        array with indices for which to search for extrema values around
    radius: int or tuple of int
        radius around `indices` to search for extrema. If `radius` is an ``int`` then search for extrema equally
        in both directions in the interval [index - radius, index + radius].
        If `radius` is a ``tuple`` then search for extrema in the interval [ index - radius[0], index + radius[1] ]
    extrema_type : {'min', 'max'}, optional
        extrema type to be searched for. Default: 'min'

    Returns
    -------
    array_like
        array containing the indices of the found extrema values in the given radius around `indices`.
        Has the same length as `indices`.

    Examples
    --------
    >>> import biopsykit.signal as signal
    >>> data = pd.read_csv("data.csv")
    >>> indices = np.array([16, 25, 40, 57, 86, 100])
    >>>
    >>> radius = 4
    >>> # search minima in 'data' in a 4 sample 'radius' around each entry of 'indices'
    >>> signal.find_extrema_in_radius(data, indices, radius)
    >>>
    >>> radius = (5, 0)
    >>> # search maxima in 'data' in a 5 samples before each entry of 'indices'
    >>> signal.find_extrema_in_radius(data, indices, radius, extrema_type='max')
    """
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}

    if extrema_type not in extrema_funcs:
        raise ValueError("`extrema_type` must be one of {}, not {}".format(list(extrema_funcs.keys()), extrema_type))
    extrema_func = extrema_funcs[extrema_type]

    # ensure numpy
    data = sanitize_input_1d(data)
    indices = sanitize_input_1d(indices)
    indices = indices.astype(int)
    # possible start offset if beginning of array needs to be padded to ensure radius
    start_padding = 0

    # determine upper and lower limit
    if isinstance(radius, tuple):
        lower_limit = radius[0]
    else:
        lower_limit = radius
    if isinstance(radius, tuple):
        upper_limit = radius[-1]
    else:
        upper_limit = radius

    # round up and make sure it's an integer
    lower_limit = np.ceil(lower_limit).astype(int)
    upper_limit = np.ceil(upper_limit).astype(int)

    # pad end/start of array if last_index+radius/first_index-radius is longer/shorter than array
    if len(data) - np.max(indices) <= upper_limit:
        data = np.pad(data, (0, upper_limit), constant_values=np.nan)
    if np.min(indices) < lower_limit:
        start_padding = lower_limit
        data = np.pad(data, (lower_limit, 0), constant_values=np.nan)

    # initialize window array
    windows = np.zeros(shape=(len(indices), lower_limit + upper_limit + 1))
    for i, index in enumerate(indices):
        # get windows around index
        windows[i] = data[index - lower_limit + start_padding:index + upper_limit + start_padding + 1]

    return extrema_func(windows, axis=1) + indices - lower_limit


def remove_outlier_and_interpolate(data: np.ndarray, outlier_mask: np.ndarray, x_old: Optional[np.ndarray] = None,
                                   desired_length: Optional[int] = None) -> np.ndarray:
    """
    Sets all detected outlier to nan, imputes them by linear interpolation their neighbors and interpolates
    the resulting values to a desired length.

    Parameters
    ----------
    data : array_like
        input data
    outlier_mask: array_like
        outlier mask. has to be the same length like `data`. ``True`` entries indicate outliers
    x_old : array_like, optional
        x values of the input data to interpolate or ``None`` if no interpolation should be performed. Default: ``None``
    desired_length : int, optional
        desired length of the output signal or ``None`` to keep input length. Default: ``None``

    Returns
    -------
    np.array
        Outlier-removed and interpolated data

    Raises
    ------
    ValueError
        if `data` and `outlier_mask` don't have the same length or if `x_old` is ``None`` when `desired_length`
        is passed as parameter

    """
    # ensure numpy
    data = np.array(data)
    outlier_mask = np.array(outlier_mask)

    if len(data) != len(outlier_mask):
        raise ValueError("`data` and `outlier_mask` need to have same length!")
    if x_old is None and desired_length:
        raise ValueError("`x_old` must also be passed when `desired_length` is passed!")
    x_old = np.array(x_old)

    # remove outlier
    data[outlier_mask] = np.nan
    # fill outlier by linear interpolation of neighbors
    data = pd.Series(data).interpolate(limit_direction='both').values
    # interpolate signal
    x_new = np.linspace(x_old[0], x_old[-1], desired_length)
    return nk.signal_interpolate(x_old, data, x_new, method='linear')


def sliding_window(
        data: Union[np.array, pd.Series, pd.DataFrame],
        sampling_rate: Union[int, float],
        window_s: int,
        overlap_percent: Optional[float] = 0,
        overlap_samples: Optional[int] = 0
):
    data = sanitize_input_1d(data)
    window = int(sampling_rate * window_s)

    if overlap_samples == 0:
        window_step = window - int(overlap_percent * window)
    else:
        window_step = overlap_samples

    data = np.pad(data, (0, window - data.shape[0] % window), 'constant', constant_values=0)
    new_shape = data.shape[:-1] + ((data.shape[-1] - int(overlap_percent * window)) // window_step,
                                   window)
    new_strides = (data.strides[:-1] + (window_step * data.strides[-1],) +
                   data.strides[-1:])
    arr_new = as_strided(data, shape=new_shape, strides=new_strides)
    return arr_new
