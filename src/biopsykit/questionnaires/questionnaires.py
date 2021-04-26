"""Module containing implementations for various psychological questionnaires.

Each function at least expects a dataframe containing the required columns in a specified order
(see function documentations for specifics) to be passed to the ``data`` argument.

If ``data`` is a dataframe that contains more than the required two columns, e.g., if the complete questionnaire
dataframe is passed, the required columns can be sliced by specifying them in the ``columns`` parameter.
Also, if the columns in the dataframe dataframe columns are not in the correct order, the order can be specified
using the ``columns`` parameter.

Some questionnaire functions also allow the possibility to only compute certain subscales. To do this, a dictionary
with subscale names as keys and the corresponding column names (as list of str) or column indices
(as list of ints) can be passed to the ``subscales`` parameter.

.. warning::
    Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
    questionnaire item columns, which typically also start with index 1!

"""
from typing import Optional, Sequence, Union, Dict, Literal

import numpy as np
import pandas as pd

from biopsykit.questionnaires.utils import (
    invert,
    bin_scale,
    to_idx,
    _compute_questionnaire_subscales,
    _invert_subscales,
)
from biopsykit.utils._datatype_validation_helper import _assert_value_range, _assert_num_columns


def psqi(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **Pittsburgh Sleep Quality Index (PSQI)**.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.

    Returns
    -------
    :class:`~pandas.DataFrame`
        PSQI score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    """
    score_name = "PSQI"
    score_range = [0, 3]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # Subjective Sleep Quality
    ssq = data.filter(regex="06").iloc[:, 0]

    # Sleep Latency
    sl = data.filter(regex="02").iloc[:, 0]
    bin_scale(sl, bins=[0, 15, 30, 60], last_max=True, inplace=True)

    # Sleep Duration
    sd = data.filter(regex="04").iloc[:, 0]

    # Sleep Disturbances
    sdist = data.filter(regex="05").iloc[:, :]

    # Use of Sleep Medication
    sm = data.filter(regex="07").iloc[:, 0]

    # Daytime Dysfunction
    dd = data.filter(regex="0[89]").sum(axis=1)
    dd = bin_scale(dd, bins=[-1, 0, 2, 4], inplace=False, last_max=True)

    sl = sl + data.filter(regex="05a").iloc[:, 0]

    # Habitual Sleep Efficiency
    hse = (sd / data["HoursBed"]) * 100.0

    sdist = sdist.drop([sdist.columns[0], sdist.columns[-2]], axis="columns")

    sd = invert(bin_scale(sd, bins=[0, 4.9, 6, 7], last_max=True), score_range=score_range)
    hse = invert(bin_scale(hse, bins=[0, 64, 74, 84], last_max=True), score_range=score_range)
    sdist = sdist.sum(axis=1)
    sdist = bin_scale(sdist, bins=[-1, 0, 9, 18, 27])

    psqi_data = {
        score_name + "_SubjectiveSleepQuality": ssq,
        score_name + "_SleepLatency": sl,
        score_name + "_SleepDuration": sd,
        score_name + "_HabitualSleepEfficiency": hse,
        score_name + "_SleepDisturbances": sdist,
        score_name + "_UseSleepMedication": sm,
        score_name + "_DaytimeDysfunction": dd,
    }

    data = pd.DataFrame(psqi_data, index=data.index)
    data[score_name + "_TotalIndex"] = data.sum(axis=1)
    return data


def mves(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **Maastricht Vital Exhaustion Scale (MVES)**.

    The MVES uses 23 items to assess the concept of Vital Exhaustion (VE), which is characterized by feelings of
    excessive fatigue, lack of energy, irritability, and feelings of demoralization. Higher scores indicate greater
    vital exhaustion.

    .. note::
        This implementation assumes a score range of [0, 2].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        MVES score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Appels, A., Höppener, P., & Mulder, P. (1987). A questionnaire to assess premonitory symptoms of myocardial
    infarction. *International Journal of Cardiology*, 17(1), 15–24. https://doi.org/10.1016/0167-5273(87)90029-5

    """
    score_name = "MVES"
    score_range = [0, 2]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 23)
    _assert_value_range(data, score_range)

    # Reverse scores 9, 14
    data = invert(data, cols=to_idx([9, 14]), score_range=score_range)
    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def tics_s(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[Union[str, int]]]] = None,
) -> pd.DataFrame:
    """Compute the **Trier Inventory for Chronic Stress (Short Version) (TICS-S)**.

    The TICS assesses frequency of various types of stressful experiences in the past 3 months.

    It consists of the subscales (the name in the brackets indicate the name in the returned dataframe),
    with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Work Overload``: [1, 3, 21]
        * ``Social Overload``: [11, 18, 28]
        * ``Excessive Demands at Work``: [12, 16, 27]
        * ``Lack of Social Recognition``: [2, 20, 23]
        * ``Work Discontent``: [8, 13, 24]
        * ``Social Tension``: [4, 9, 26]
        * ``Performance Pressure at Work``: [5, 14, 29]
        * ``Performance Pressure in Social Interactions``: [6, 15, 22]
        * ``Social Isolation``: [19, 25, 30]
        * ``Worry Propensity``: [7, 10, 17]

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        TICS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    Examples
    --------
    >>> from biopsykit.questionnaires import tics_s
    >>> # compute only a subset of subscales; questionnaire items additionally have custom indices
    >>> subscales = {
    >>>     'WorkOverload': [1, 2, 3],
    >>>     'SocialOverload': [4, 5, 6],
    >>> }
    >>> tics_s_result = tics_s(data, subscales=subscales)

    References
    ----------
    Schulz, P., Schlotz, W., & Becker, P. (2004). Trierer Inventar zum chronischen Stress: TICS. *Hogrefe*.

    """
    score_name = "TICS_S"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 30)
        subscales = {
            "WorkOverload": [1, 3, 21],
            "SocialOverload": [11, 18, 28],
            "PressureToPerform": [5, 14, 29],
            "WorkDiscontent": [8, 13, 24],
            "DemandsWork": [12, 16, 27],
            "PressureSocial": [6, 15, 22],
            "LackSocialRec": [2, 20, 23],
            "SocialTension": [4, 9, 26],
            "SocialIsolation": [19, 25, 30],
            "ChronicWorry": [7, 10, 17],
        }

    tics_data = _compute_questionnaire_subscales(data, score_name, subscales)

    if len(data.columns) == 30:
        # compute total score if all columns are present
        tics_data[score_name] = data.sum(axis=1)

    return pd.DataFrame(tics_data, index=data.index)


def tics_l(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[Union[str, int]]]] = None,
) -> pd.DataFrame:
    """Compute the **Trier Inventory for Chronic Stress (Long Version) (TICS-L)**.

    The TICS assesses frequency of various types of stressful experiences in the past 3 months.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * Work Overload: [50, 38, 44, 54, 17, 4, 27, 1]
        * Social Overload: [39, 28, 49, 19, 7, 57]
        * Excessive Demands at Work: [55, 24, 20, 35, 47, 3]
        * Lack of Social Recognition: [31, 18, 46, 2]
        * Work Discontent: [21, 53, 10, 48, 41, 13, 37, 5]
        * Social Tension: [26, 15, 45, 52, 6, 33]
        * Performance Pressure at Work: [23, 43, 32, 22, 12, 14, 8, 40, 30]
        * Performance Pressure in Social Interactions: [6, 15, 22]
        * Social Isolation: [42, 51, 34, 56, 11, 29]
        * Worry Propensity: [36, 25, 16, 9]

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        TICS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    Examples
    --------
    >>> from biopsykit.questionnaires import tics_s
    >>> # compute only a subset of subscales; questionnaire items additionally have custom indices
    >>> subscales = {
    >>>     'WorkOverload': [1, 2, 3],
    >>>     'SocialOverload': [4, 5, 6],
    >>> }
    >>> tics_s_result = tics_s(data, subscales=subscales)

    References
    ----------
    Schulz, P., Schlotz, W., & Becker, P. (2004). Trierer Inventar zum chronischen Stress: TICS. *Hogrefe*.

    """
    score_name = "TICS_L"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 57)
        subscales = {
            "WorkOverload": [50, 38, 44, 54, 17, 4, 27, 1],  # Arbeitsüberlastung
            "SocialOverload": [39, 28, 49, 19, 7, 57],  # Soziale Überlastung
            "PressureToPerform": [23, 43, 32, 22, 12, 14, 8, 40, 30],  # Erfolgsdruck
            "WorkDiscontent": [
                21,
                53,
                10,
                48,
                41,
                13,
                37,
                5,
            ],  # Unzufriedenheit mit der Arbeit
            "DemandsWork": [55, 24, 20, 35, 47, 3],  # Überforderung bei der Arbeit
            "LackSocialRec": [31, 18, 46, 2],  # Mangel an sozialer Anerkennung
            "SocialTension": [26, 15, 45, 52, 6, 33],  # Soziale Spannungen
            "SocialIsolation": [42, 51, 34, 56, 11, 29],  # Soziale Isolation
            "ChronicWorry": [36, 25, 16, 9],  # Chronische Besorgnis
        }

    tics_data = _compute_questionnaire_subscales(data, score_name, subscales)

    if len(data.columns) == 57:
        # compute total score if all columns are present
        tics_data[score_name] = data.sum(axis=1)

    return pd.DataFrame(tics_data, index=data.index)


def pss(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **Perceived Stress Scale (PSS)**.

    The PSS is a widely used self-report questionnaire with adequate reliability and validity asking
    about how stressful a person has found his/her life during the previous month.

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        PSS score

    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Cohen, S., Kamarck, T., & Mermelstein, R. (1983). A Global Measure of Perceived Stress.
    *Journal of Health and Social Behavior*, 24(4), 385. https://doi.org/10.2307/2136404

    """
    score_name = "PSS"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 10)
    _assert_value_range(data, score_range)

    # Reverse scores 4, 5, 7, 8
    data = invert(data, cols=to_idx([4, 5, 7, 8]), score_range=score_range)

    return pd.DataFrame(data.sum(axis=1, skipna=False), columns=[score_name])


def cesd(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **Center for Epidemiological Studies Depression Scale (CES-D)**.

    The CES-D asks about depressive symptoms experienced over the past week.
    Higher scores indicate greater depressive symptoms.

    .. note::
        This implementation assumes a score range of [0, 3].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        CES-D score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Radloff, L. S. (1977). The CES-D Scale: A Self-Report Depression Scale for Research in the General Population.
    Applied Psychological Measurement, 1(3), 385–401. https://doi.org/10.1177/014662167700100306

    """
    score_name = "CESD"
    score_range = [0, 3]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 20)
    _assert_value_range(data, score_range)

    # Reverse scores 4, 8, 12, 16
    data = invert(data, cols=to_idx([4, 8, 12, 16]), score_range=score_range)
    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def ghq(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **General Health Questionnaire (GHQ)**.

    The GHQ-12 is a widely used tool for detecting psychological and mental health and as a screening tool for
    excluding psychological and psychiatric morbidity. Higher scores indicate *lower* health.
    A summed score above 4 is considered an indicator of psychological morbidity.

    .. note::
        This implementation assumes a score range of [0, 3].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.

    Returns
    -------
    :class:`~pandas.DataFrame`
        GHQ score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Goldberg, D. P. (1972). The detection of psychiatric illness by questionnaire. *Maudsley monograph*, 21.

    """
    score_name = "GHQ"
    score_range = [0, 3]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 12)
    _assert_value_range(data, score_range)

    # Reverse scores 1, 3, 4, 7, 8, 12
    data = invert(data, cols=to_idx([1, 3, 4, 7, 8, 12]), score_range=score_range)
    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def hads(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[Union[str, int]]]] = None,
) -> pd.DataFrame:
    """Compute the **Hospital Anxiety and Depression Scale (HADS)**.

    The HADS is a brief and widely used instrument to measure psychological distress in patients
    and in the general population. It has two subscales: anxiety and depression.
    Higher scores indicate greater distress.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Anxiety``: [1, 3, 5, 7, 9, 11, 13]
        * ``Depression``: [2, 4, 6, 8, 10, 12, 14]

    .. note::
        This implementation assumes a score range of [0, 3].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        HADS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns do not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Zigmond, A. S., & Snaith, R. P. (1983). The hospital anxiety and depression scale.
    *Acta psychiatrica scandinavica*, 67(6), 361-370.

    """
    score_name = "HADS"
    score_range = [0, 3]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 14)
        subscales = {
            "Anxiety": [1, 3, 5, 7, 9, 11, 13],
            "Depression": [2, 4, 6, 8, 10, 12, 14],
        }

    # Reverse scores 2, 4, 6, 7, 12, 14
    data = _invert_subscales(
        data, subscales=subscales, idx_dict={"Anxiety": [7], "Depression": [2, 4, 6, 12, 14]}, score_range=score_range
    )

    hads_data = _compute_questionnaire_subscales(data, score_name, subscales)

    if len(data.columns) == 14:
        # compute total score if all columns are present
        hads_data[score_name] = data.sum(axis=1)
    return pd.DataFrame(hads_data, index=data.index)


def type_d_scale(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Compute **Type D Personality Scale**.

    Type D personality is a personality trait characterized by negative affectivity (NA) and social
    inhibition (SI). Individuals who are high in both NA and SI have a *distressed* or Type D personality.

    It consists of the subscales, with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Negative Affect``: [2, 4, 5, 7, 9, 12, 13]
        * ``Social Inhibition``: [1, 3, 6, 8, 10, 11, 14]


    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        DS Type-D score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Denollet, J. (2005). DS14: standard assessment of negative affectivity, social inhibition, and Type D personality.
    *Psychosomatic medicine*, 67(1), 89-97.

    """
    score_name = "DS"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 14)
        subscales = {
            "NegativeAffect": [2, 4, 5, 7, 9, 12, 13],
            "SocialInhibition": [1, 3, 6, 8, 10, 11, 14],
        }

    # Reverse scores 1, 3
    data = _invert_subscales(data, subscales=subscales, idx_dict={"SocialInhibition": [1, 3]}, score_range=score_range)

    ds_data = _compute_questionnaire_subscales(data, score_name, subscales)

    if len(data.columns) == 14:
        # compute total score if all columns are present
        ds_data[score_name] = data.sum(axis=1)

    return pd.DataFrame(ds_data, index=data.index)


def rse(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute **Rosenberg Self-Esteem Inventory**.

    The RSE is the most frequently used measure of global self-esteem. Higher scores indicate greater self-esteem.

    .. note::
        This implementation assumes a score range of [0, 3].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        RSE score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Rosenberg, M. (1965). Society and the Adolescent Self-Image. *Princeton University Press*, Princeton, NJ.

    """
    score_name = "RSE"
    score_range = [0, 3]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 10)
    _assert_value_range(data, score_range)

    # Reverse scores 2, 5, 6, 8, 9
    data = invert(data, cols=to_idx([2, 5, 6, 8, 9]), score_range=score_range)
    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def scs(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Compute **Self-Compassion Scale (SCS)**.

    The Self-Compassion Scale measures the tendency to be compassionate rather than critical
    toward the self in difficult times. It is typically assessed as a composite but can be broken down
    into subscales. Higher scores indicate greater self-compassion.

    It consists of the subscales, with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``SelfKindness``: [5, 12, 19, 23, 26]
        * ``SelfJudgment``: [1, 8, 11, 16, 21]
        * ``CommonHumanity``: [3, 7, 10, 15]
        * ``Isolation``: [4, 13, 18, 25]
        * ``Mindfulness``: [9, 14, 17, 22]
        * ``OverIdentified`` [2, 6, 20, 24]

    .. note::
        This implementation assumes a score range of [1, 5].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        SCS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Neff, K. D. (2003). The development and validation of a scale to measure self-compassion.
    *Self and identity*, 2(3), 223-250.
    https://www.academia.edu/2040459

    """
    score_name = "SCS"
    score_range = [1, 5]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 26)
        subscales = {
            "SelfKindness": [5, 12, 19, 23, 26],
            "SelfJudgment": [1, 8, 11, 16, 21],
            "CommonHumanity": [3, 7, 10, 15],
            "Isolation": [4, 13, 18, 25],
            "Mindfulness": [9, 14, 17, 22],
            "OverIdentified": [2, 6, 20, 24],
        }
    # Reverse scores 1, 2, 4, 6, 8, 11, 13, 16, 18, 20, 21, 24, 25
    data = _invert_subscales(
        data,
        subscales=subscales,
        idx_dict={"SelfJudgment": [1, 8, 11, 16, 21], "Isolation": [4, 13, 18, 25], "OverIdentified": [2, 6, 20, 24]},
        score_range=score_range,
    )

    # SCS is a mean, not a sum score!
    scs_data = _compute_questionnaire_subscales(data, score_name, subscales, agg_type="mean")

    if len(data.columns) == 26:
        # compute total score if all columns are present
        scs_data[score_name] = data.mean(axis=1)

    return pd.DataFrame(scs_data, index=data.index)


def midi(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute **Midlife Development Inventory (MIDI) Sense of Control Scale**.

    The Midlife Development Inventory (MIDI) sense of control scale assesses perceived control,
    that is, how much an individual perceives to be in control of his or her environment. Higher scores indicate
    greater sense of control.

    .. note::
        This implementation assumes a score range of [1, 7].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        MIDI score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Lachman, M. E., & Weaver, S. L. (1998). The sense of control as a moderator of social class differences in
    health and well-being. *Journal of personality and social psychology*, 74(3), 763.

    """
    score_name = "MIDI"
    score_range = [1, 7]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 12)
    _assert_value_range(data, score_range)

    # Reverse scores 1, 2, 4, 5, 7, 9, 10, 11
    data = invert(data, cols=to_idx([1, 2, 4, 5, 7, 9, 10, 11]), score_range=score_range)

    # MIDI is a mean, not a sum score!
    return pd.DataFrame(data.mean(axis=1), columns=[score_name])


def tsgs(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[Union[str, int]]]] = None,
) -> pd.DataFrame:
    """Compute **Trait Shame and Guilt Scale**.

    The TSGS assesses the experience of shame, guilt, and pride over the past few months with three separate subscales.
    Shame and guilt are considered distinct emotions, with shame being a global negative feeling about the self,
    and guilt being a negative feeling about a specific event rather than the self. Higher scores on each subscale
    indicate higher shame, guilt, or pride.

    It consists of the subscales,
    with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Shame``: [2, 5, 8, 11, 14]
        * ``Guilt``: [3, 6, 9, 12, 15]
        * ``Pride``: [1, 4, 7, 10, 13]

    .. note::
        This implementation assumes a score range of [1, 5].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.

    Returns
    -------
    :class:`~pandas.DataFrame`
        TSGS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Rohleder, N., Chen, E., Wolf, J. M., & Miller, G. E. (2008). The psychobiology of trait shame in young women:
    Extending the social self preservation theory. *Health Psychology*, 27(5), 523.

    """
    score_name = "TSGS"
    score_range = [1, 5]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 15)
        subscales = {
            "Shame": [2, 5, 8, 11, 14],
            "Guilt": [3, 6, 9, 12, 15],
            "Pride": [1, 4, 7, 10, 13],
        }

    tsgs_data = _compute_questionnaire_subscales(data, score_name, subscales)
    return pd.DataFrame(tsgs_data, index=data.index)


def rmidi(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[Union[str, int]]]] = None,
) -> pd.DataFrame:
    """Compute **Revised Midlife Development Inventory (MIDI) Personality Scale**.

    The Midlife Development Inventory (MIDI) includes 6 personality trait scales: Neuroticism,
    Extraversion, Openness to Experience, Conscientiousness, Agreeableness, and Agency.  Higher scores
    indicate higher endorsement of each personality trait.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Neuroticism``: [3, 8, 13, 19]
        * ``Extraversion``: [1, 6, 11, 23, 27]
        * ``Openness``: [14, 17, 21, 22, 25, 28, 29]
        * ``Conscientiousness``: [4, 9, 16, 24, 31]
        * ``Agreeableness``: [2, 7, 12, 18, 26]
        * ``Agency``: [5, 10, 15, 20, 30]

    .. note::
        This implementation assumes a score range of [1, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        RMIDI score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Prenda, K. M., & Lachman, M. E. (2001). Planning for the future: a life management strategy for increasing control
    and life satisfaction in adulthood. *Psychology and aging*, 16(2), 206.

    """
    score_name = "RMIDI"
    score_range = [1, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # "most items need to be reverse scored before subscales are computed => reverse all"
    data = invert(data, score_range=score_range)

    if subscales is None:
        _assert_num_columns(data, 31)
        subscales = {
            "Neuroticism": [3, 8, 13, 19],
            "Extraversion": [1, 6, 11, 23, 27],
            "Openness": [14, 17, 21, 22, 25, 28, 29],
            "Conscientiousness": [4, 9, 16, 24, 31],
            "Agreeableness": [2, 7, 12, 18, 26],
            "Agency": [5, 10, 15, 20, 30],
        }

    # Re-reverse scores 19, 24
    data = _invert_subscales(
        data, subscales=subscales, idx_dict={"Neuroticism": [19], "Conscientiousness": [24]}, score_range=score_range
    )
    rmidi_data = _compute_questionnaire_subscales(data, score_name, subscales)

    return pd.DataFrame(rmidi_data, index=data.index)


def lsq(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute **Life Stress Questionnaire**.

    The LSQ asks participants about stressful life events that they and their close relatives have experienced
    throughout their entire life, what age they were when the event occurred, and how much it impacted them.
    Higher scores indicate more stress.

    It consists of the subscales:
        * ``PartnerStress``: columns with suffix ``_Partner``
        * ``ParentStress``: columns with suffix ``_Parent``
        * ``ChildStress``: columns with suffix ``_Child``

    .. note::
        This implementation assumes a score range of [0, 1].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : list of str, optional
        List of subscales (``Partner``, ``Parent``, ``Child``) to compute or ``None`` to compute all subscales.
        Default: ``None``


    Returns
    -------
    :class:`~pandas.DataFrame`
        LSQ score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Prenda, K. M., & Lachman, M. E. (2001). Planning for the future: a life management strategy for increasing control
    and life satisfaction in adulthood. *Psychology and aging*, 16(2), 206.

    """
    score_name = "LSQ"
    score_range = [0, 1]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 30)
        subscales = ["Partner", "Parent", "Child"]

    lsq_data = {"{}_{}".format(score_name, subscale): data.filter(like=subscale).sum(axis=1) for subscale in subscales}

    return pd.DataFrame(lsq_data, index=data.index)


def ctq(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Compute **Childhood Trauma Questionnaire (CTQ)**.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``PhysicalAbuse``: [9, 11, 12, 15, 17]
        * ``SexualAbuse``: [20, 21, 23, 24, 27]
        * ``EmotionalNeglect``: [5, 7, 13, 19, 28]
        * ``PhysicalNeglect``: [1, 2, 4, 6, 26]
        * ``EmotionalAbuse``: [3, 8, 14, 18, 25]

    Additionally, three items assess the validity of the responses (high scores on these items could be grounds for
    exclusion of a given participants’ responses):
        * ``Validity``: [10, 16, 22]

    .. note::
        This implementation assumes a score range of [1, 5].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        CTQ score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Bernstein, D. P., Fink, L., Handelsman, L., Foote, J., Lovejoy, M., Wenzel, K., ... & Ruggiero, J. (1994).
    Initial reliability and validity of a new retrospective measure of child abuse and neglect.
    *The American journal of psychiatry*.

    """
    score_name = "CTQ"
    score_range = [1, 5]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 28)
        subscales = {
            "PhysicalAbuse": [9, 11, 12, 15, 17],
            "SexualAbuse": [20, 21, 23, 24, 27],
            "EmotionalNeglect": [5, 7, 13, 19, 28],
            "PhysicalNeglect": [1, 2, 4, 6, 26],
            "EmotionalAbuse": [3, 8, 14, 18, 25],
            "Validity": [10, 16, 22],
        }

    # reverse scores 2, 5, 7, 13, 19, 26, 28
    data = _invert_subscales(
        data,
        subscales=subscales,
        idx_dict={
            "PhysicalNeglect": [2, 26],
            "EmotionalNeglect": [5, 7, 13, 19, 28],
        },
        score_range=score_range,
    )
    ctq_data = _compute_questionnaire_subscales(data, score_name, subscales)

    return pd.DataFrame(ctq_data, index=data.index)


def peat(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute **Pittsburgh Enjoyable Activities Test (PEAT)**.

    The PEAT is a self-report measure of engagement in leisure activities. It asks participants to report how often
    over the last month they have engaged in each of the activities. Higher scores indicate more time spent in
    leisure activities.

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        PEAT score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Pressman, S. D., Matthews, K. A., Cohen, S., Martire, L. M., Scheier, M., Baum, A., & Schulz, R. (2009).
    Association of enjoyable leisure activities with psychological and physical well-being.
    *Psychosomatic medicine*, 71(7), 725.

    """
    score_name = "PEAT"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 10)
    _assert_value_range(data, score_range)

    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def purpose_life(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute **Purpose in Life** questionnaire.

    Purpose in life refers to the psychological tendency to derive meaning from life’s experiences
    and to possess a sense of intentionality and goal directedness that guides behavior.
    Higher scores indicate greater purpose in life.

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        TICS score

    References
    ----------
    Boyle, P. A., Barnes, L. L., Buchman, A. S., & Bennett, D. A. (2009). Purpose in life is associated with
    mortality among community-dwelling older persons. *Psychosomatic medicine*, 71(5), 574.

    """
    score_name = "PurposeLife"
    score_range = [1, 5]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 10)
    _assert_value_range(data, score_range)

    # reverse scores 2, 3, 5, 6, 10
    data = invert(data, cols=to_idx([2, 3, 5, 6, 10]), score_range=score_range)

    # Purpose in Life is a mean, not a sum score!
    return pd.DataFrame(data.mean(axis=1), columns=[score_name])


def trait_rumination(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute **Trait Rumination**.

    Higher scores indicate greater rumination.

    .. note::
        This implementation assumes a score range of [0, 1], where 0 = no rumination, 1 = rumination.
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        TraitRumination score


    References
    ----------
    Nolen-Hoeksema, S., Morrow, J., & Fredrickson, B. L. (1993). Response styles and the duration of episodes of
    depressed mood. *Journal of abnormal psychology*, 102(1), 20.

    """
    score_name = "TraitRumination"
    score_range = [0, 1]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def besaa(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[Union[int, str]]]] = None,
) -> pd.DataFrame:
    """Compute **Body-Esteem Scale for Adolescents and Adults (BESAA)**.

    Body Esteem refers to self-evaluations of one’s body or appearance. The BESAA is based on
    the idea that feelings about one’s weight can be differentiated from feelings about one’s general appearance,
    and that one’s own opinions may be differentiated from the opinions attributed to others.
    Higher scores indicate higher body esteem.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Appearance``: [1, 6, 9, 7, 11, 13, 15, 17, 21, 23]
        * ``Weight``: [3, 4, 8, 10, 16, 18, 19, 22]
        * ``Attribution``: [2, 5, 12, 14, 20]

    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        BESAA score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Mendelson, B. K., Mendelson, M. J., & White, D. R. (2001). Body-esteem scale for adolescents and adults.
    *Journal of personality assessment*, 76(1), 90-106.

    """
    score_name = "BESAA"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 23)
        subscales = {
            "Appearance": [1, 6, 9, 7, 11, 13, 15, 17, 21, 23],
            "Weight": [3, 4, 8, 10, 16, 18, 19, 22],
            "Attribution": [2, 5, 12, 14, 20],
        }

    # reverse scores 4, 7, 9, 11, 13, 17, 18, 19, 21
    data = _invert_subscales(
        data,
        subscales=subscales,
        idx_dict={"Appearance": [7, 17, 21], "Weight": [4, 18, 19], "Attribution": [9, 11, 13]},
        score_range=score_range,
    )

    # BESAA is a mean, not a sum score!
    besaa_data = _compute_questionnaire_subscales(data, score_name, subscales, agg_type="mean")

    return pd.DataFrame(besaa_data, index=data.index)


def fscrs(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Compute **Forms of Self-Criticizing/Attacking and Self-Reassuring Scale (FSCRS)**.

    Self-criticism describes the internal relationship with the self in which part of the self shames
    and puts down, while the other part of the self responds and submits to such attacks.
    Self-reassurance refers to the opposing idea that many individuals focus on positive aspects of self and defend
    against self-criticism. The FSCRS exemplifies some of the self-statements made by either those who are
    self-critical or by those who self-reassure.
    The scale measures these two traits on a continuum with self-criticism at one end and
    self-reassurance at the other. Higher scores on each subscale indicate higher self-criticizing ("Inadequate Self"),
    self-attacking ("Hated Self"), and self-reassuring "Reassuring Self", respectively.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``InadequateSelf``: [1, 2, 4, 6, 7, 14, 17, 18, 20]
        * ``HatedSelf``: [9, 10, 12, 15, 22]
        * ``ReassuringSelf``: [3, 5, 8, 11, 13, 16, 19, 21]

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        FSCRS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Gilbert, P., Clarke, M., Hempel, S., Miles, J. N., & Irons, C. (2004). Criticizing and reassuring oneself:
    An exploration of forms, styles and reasons in female students. *British Journal of Clinical Psychology*,
    43(1), 31-50.

    """
    score_name = "FSCRS"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 22)
        subscales = {
            "InadequateSelf": [1, 2, 4, 6, 7, 14, 17, 18, 20],
            "HatedSelf": [9, 10, 12, 15, 22],
            "ReassuringSelf": [3, 5, 8, 11, 13, 16, 19, 21],
        }

    fscrs_data = _compute_questionnaire_subscales(data, score_name, subscales)

    return pd.DataFrame(fscrs_data, index=data.index)


def pasa(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Compute the **Primary Appraisal Secondary Appraisal Scale (PASA)**.

    The PASA assesses each of the four cognitive appraisal processes relevant for acute stress protocols,
    such as the TSST: primary stress appraisal (threat and challenge) and secondary stress appraisal
    (self-concept of own abilities and control expectancy). Higher scores indicate greater appraisals for each sub-type.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Threat``: [1, 9, 5, 13]
        * ``Challenge``: [6, 10, 2, 14]
        * ``SelfConcept``: [7, 3, 11, 15]
        * ``ControlExp``: [4, 8, 12, 16]

    .. note::
        This implementation assumes a score range of [1, 6].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        PASA score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range


    References
    ----------
    Gaab, J., Rohleder, N., Nater, U. M., & Ehlert, U. (2005). Psychological determinants of the cortisol stress
    response: the role of anticipatory cognitive appraisal. *Psychoneuroendocrinology*, 30(6), 599-610.

    """
    score_name = "PASA"
    score_range = [1, 6]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 16)
        subscales = {
            "Threat": [1, 9, 5, 13],
            "Challenge": [6, 10, 2, 14],
            "SelfConcept": [7, 3, 11, 15],
            "ControlExp": [4, 8, 12, 16],
        }

    # reverse scores 1, 6, 7, 9, 10
    data = _invert_subscales(
        data,
        subscales=subscales,
        idx_dict={"Threat": [1, 9], "Challenge": [6, 10], "SelfConcept": [7]},
        score_range=score_range,
    )

    pasa_data = _compute_questionnaire_subscales(data, score_name, subscales)

    if all(s in subscales for s in ["Threat", "Challenge"]):
        pasa_data[score_name + "_Primary"] = (
            pasa_data[score_name + "_Threat"] + pasa_data[score_name + "_Challenge"]
        ) / 2

    if all(s in subscales for s in ["SelfConcept", "ControlExp"]):
        pasa_data[score_name + "_Secondary"] = (
            pasa_data[score_name + "_SelfConcept"] + pasa_data[score_name + "_ControlExp"]
        ) / 2

    if all(s in subscales for s in ["PASA_Primary", "PASA_Secondary"]):
        pasa_data[score_name + "_StressComposite"] = (
            pasa_data[score_name + "_Primary"] - pasa_data[score_name + "_Secondary"]
        )

    return pd.DataFrame(pasa_data, index=data.index)


def ssgs(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    subscales: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Compute the **State Shame and Guilt Scale (SSGS)**.

    The SSGS assesses the experience of shame, guilt, and pride experienced during an acute stress protocol with three
    separate subscales. Shame and guilt are considered distinct emotions, with shame being a global negative feeling
    about the self, and guilt being a negative feeling about a specific event rather than the self.
    This scale is a modified version from the State Shame and Guilt scale by Marschall et al. (1994).
    Higher scores on each subscale indicate higher shame, guilt, or pride.

    It consists of the subscales with the item indices (count-by-one, i.e., the first question has the index 1!):
        * ``Pride``: [1, 4, 7, 10, 13]
        * ``Shame``: [2, 5, 8, 11, 14]
        * ``Guilt``: [3, 6, 9, 12, 15]

    .. note::
        This implementation assumes a score range of [1, 5].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    subscales : dict, optional
        A dictionary with subscale names (keys) and column names or column indices (count-by-1) (values)
        if only specific subscales should be computed.


    Returns
    -------
    :class:`~pandas.DataFrame`
        SSGS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Rohleder, N., Chen, E., Wolf, J. M., & Miller, G. E. (2008). The psychobiology of trait shame in young women:
    Extending the social self preservation theory. *Health Psychology*, 27(5), 523.

    Marschall, D., Sanftner, J., & Tangney, J. P. (1994). The state shame and guilt scale.
    *Fairfax, VA: George Mason University*.

    """
    score_name = "SSGS"
    score_range = [1, 5]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if subscales is None:
        _assert_num_columns(data, 15)
        subscales = {
            "Pride": [1, 4, 7, 10, 13],
            "Shame": [2, 5, 8, 11, 14],
            "Guilt": [3, 6, 9, 12, 15],
        }

    ssgs_data = _compute_questionnaire_subscales(data, score_name, subscales)

    return pd.DataFrame(ssgs_data, index=data.index)


def panas(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    questionnaire_version: Optional[Literal["english", "german"]] = "english",
) -> pd.DataFrame:
    """Compute the **Positive and Negative Affect Schedule (PANAS)**.

    The PANAS assesses *positive affect* (interested, excited, strong, enthusiastic, proud, alert, inspired,
    determined, attentive, and active) and *negative affect* (distressed, upset, guilty, scared, hostile, irritable,
    ashamed, nervous, jittery, and afraid).
    Higher scores on each subscale indicate greater positive or negative affect.

    .. note::
        This implementation assumes a score range of [1, 5].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.

    .. warning::
        Column indices in ``subscales`` are assumed to start at 1 (instead of 0) to avoid confusion with
        questionnaire item columns, which typically also start with index 1!


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.
    questionnaire_version : "english" or "german", optional
        Language of the questionnaire used since index items differ between the german and the english version.
        Default: ``english``


    Returns
    -------
    :class:`~pandas.DataFrame`
        PANAS score


    Raises
    ------
    ValueError
        if ``subscales`` is supplied and dict values are something else than a list of strings or a list of ints
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Watson, D., Clark, L. A., & Tellegen, A. (1988). Development and validation of brief measures of positive and
    negative affect: the PANAS scales. *Journal of personality and social psychology*, 54(6), 1063.

    """
    score_name = "PANAS"
    score_range = [1, 5]
    supported_versions = ["english", "german"]

    # create copy of data
    data = data.copy()

    if questionnaire_version not in supported_versions:
        raise AttributeError(
            "questionnaire_version must be one of {}, not {}.".format(supported_versions, questionnaire_version)
        )

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_num_columns(data, 20)
    _assert_value_range(data, score_range)

    if questionnaire_version == "german":
        # German Version has other item indices
        neg_affect = [2, 5, 7, 8, 9, 12, 14, 16, 19, 20]
    else:
        neg_affect = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]

    panas_data = {score_name + "_NegativeAffect": data.iloc[:, to_idx(neg_affect)].sum(axis=1)}
    panas_data[score_name + "_PositiveAffect"] = data.sum(axis=1) - panas_data[score_name + "_NegativeAffect"]
    panas_data[score_name + "_Total"] = panas_data[score_name + "_PositiveAffect"] + invert(
        data.iloc[:, to_idx(neg_affect)], score_range=score_range
    ).sum(axis=1)

    return pd.DataFrame(panas_data, index=data.index)


def state_rumination(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Compute the **State Rumination** scale.

    Rumination is the tendency to dwell on negative thoughts and emotions.
    Higher scores indicate greater rumination.

    .. note::
        This implementation assumes a score range of [0, 4].
        Use :func:`~biopsykit.questionnaires.utils.convert_scale()` to convert the items into the correct range
        beforehand.


    Parameters
    ----------
    data : :class:`~pandas.DataFrame`
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if ``columns`` parameter is supplied
    columns : list of str or :class:`pandas.Index`, optional
        list with column names in correct order.
        This can be used if columns in the dataframe are not in the correct order or if a complete dataframe is
        passed as ``data``.


    Returns
    -------
    :class:`~pandas.DataFrame`
        State Rumination score


    Raises
    ------
    `biopsykit.exceptions.ValidationError`
        if number of columns does not match
    `biopsykit.exceptions.ValueRangeError`
        if values are not within the required score range

    References
    ----------
    Roger, D., & Najarian, B. (1998). The relationship between emotional rumination and cortisol secretion
    under stress. *Personality and Individual Differences*, 24(4), 531-538.

    """
    score_name = "StateRumination"
    score_range = [0, 4]

    # create copy of data
    data = data.copy()

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)
    _assert_num_columns(data, 27)

    # reverse scores 1, 6, 9, 12, 15, 17, 18, 20, 27
    data = invert(data, cols=to_idx([1, 6, 9, 12, 15, 17, 18, 20, 27]), score_range=score_range)

    state_rum = {score_name: data.sum(axis=1)}

    return pd.DataFrame(state_rum, index=data.index)


# HABIT DATASET


def abi(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Angstbewältigungsinventar"""

    score_name = "ABI"
    score_range = [1, 2]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # split into 8 subitems, consisting of 10 questions each
    items = np.split(data, 8, axis=1)
    abi_raw = pd.concat(items, keys=np.arange(1, len(items) + 1), axis=1)
    idx_kov = {
        # ABI-P
        2: [2, 3, 7, 8, 9],
        4: [1, 4, 5, 8, 10],
        6: [2, 3, 5, 6, 7],
        8: [2, 4, 6, 8, 10],
        # ABI-E
        1: [2, 3, 6, 8, 10],
        3: [2, 4, 5, 7, 9],
        5: [3, 4, 5, 9, 10],
        7: [1, 5, 6, 7, 9],
    }
    idx_kov = {key: np.array(idx_kov[key]) for key in idx_kov}
    idx_vig = {key: np.setdiff1d(np.arange(1, 11), np.array(idx_kov[key]), assume_unique=True) for key in idx_kov}
    abi_kov, abi_vig = [
        pd.concat(
            [abi_raw.loc[:, key].iloc[:, idx[key] - 1] for key in idx],
            axis=1,
            keys=abi_raw.columns.unique(level=0),
        )
        for idx in [idx_kov, idx_vig]
    ]

    abi_data = {
        score_name + "_KOV-T": abi_kov.sum(axis=1),
        score_name + "_VIG-T": abi_vig.sum(axis=1),
        score_name + "_KOV-P": abi_kov.loc[:, [2, 4, 6, 8]].sum(axis=1),
        score_name + "_VIG-P": abi_vig.loc[:, [2, 4, 6, 8]].sum(axis=1),
        score_name + "_KOV-E": abi_kov.loc[:, [1, 3, 5, 7]].sum(axis=1),
        score_name + "_VIG-E": abi_vig.loc[:, [1, 3, 5, 7]].sum(axis=1),
    }

    return pd.DataFrame(abi_data, index=data.index)


def stadi(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """State-Trait-Angst-Depressions-Inventar"""

    score_name = "STADI"
    score_range = [1, 4]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if len(data.columns) == 10:
        # only STADI-State-Anxiety
        stadi_data = dict()
        for subsc, idx in zip(["AU", "BE"], [[1, 3, 5, 7, 9], [2, 4, 6, 9, 10]]):
            stadi_data["{}_State_{}".format(score_name, subsc)] = data.iloc[:, to_idx(idx)].sum(axis=1)
        df_stadi = pd.DataFrame(stadi_data, index=data.index)
        df_stadi["{}_State_Anxiety".format(score_name)] = (
            stadi_data["{}_State_AU".format(score_name)] + stadi_data["{}_State_BE".format(score_name)]
        )
        return df_stadi
    elif len(data.columns) == 20:
        st = ["State"]
    else:
        st = ["State", "Trait"]

    # split into n subitems (either State or State and Trait)
    items = np.split(data, len(st), axis=1)
    data = pd.concat(items, keys=st, axis=1)

    idx_stadi = {
        "AU": [1, 5, 9, 13, 17],
        "BE": [2, 6, 10, 14, 18],
        "EU": [3, 7, 11, 15, 19],
        "DY": [4, 8, 12, 16, 20],
    }

    stadi_data = dict()
    for s in st:
        for key in idx_stadi:
            stadi_data["{}_{}_{}".format(score_name, s, key)] = data[s].iloc[:, to_idx(idx_stadi[key])].sum(axis=1)

    df_stadi = pd.DataFrame(stadi_data, index=data.index)

    dict_meta = {
        "{}_{}_Anxiety".format(score_name, sub): stadi_data["{}_{}_AU".format(score_name, sub)]
        + stadi_data["{}_{}_BE".format(score_name, sub)]
        for sub in st
    }

    dep = {
        "{}_{}_Depression".format(score_name, sub): stadi_data["{}_{}_EU".format(score_name, sub)]
        + stadi_data["{}_{}_DY".format(score_name, sub)]
        for sub in st
    }
    dict_meta.update(dep)

    total = {
        "{}_{}_Total".format(score_name, sub): dict_meta["{}_{}_Anxiety".format(score_name, sub)]
        + dict_meta["{}_{}_Depression".format(score_name, sub)]
        for sub in st
    }
    dict_meta.update(total)

    df_meta = pd.DataFrame(dict_meta, index=data.index)
    df_meta = df_meta.reindex(sorted(df_meta.columns), axis="columns")

    # join dataframe of subscores and meta scores
    return df_stadi.join(df_meta)


def svf_120(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Stressverarbeitungsfragebogen - 120 items

    NOTE: This implementation expects a score range of [1, 5].
    """

    score_name = "SVF120"
    score_range = [1, 5]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if idxs is None:
        idxs = {
            "Bag": [10, 31, 50, 67, 88, 106],  # Bagatellisierung
            "Her": [17, 38, 52, 77, 97, 113],  # Herunterspielen
            "Schab": [5, 30, 43, 65, 104, 119],  # Schuldabwehr
            "Abl": [1, 20, 45, 86, 101, 111],  # Ablenkung
            "Ers": [22, 36, 64, 74, 80, 103],  # Ersatzbefriedigung
            "Sebest": [34, 47, 59, 78, 95, 115],  # Selbstbestätigung
            "Entsp": [12, 28, 58, 81, 99, 114],  # Entspannung
            "Sitkon": [11, 18, 39, 66, 91, 116],  # Situationskontrolle
            "Rekon": [2, 26, 54, 68, 85, 109],  # Reaktionskontrolle
            "Posi": [15, 37, 56, 71, 83, 96],  # Positive Selbstinstruktion
            "Sozube": [3, 21, 42, 63, 84, 102],  # Soziales Unterstützungsbedürfnis
            "Verm": [8, 29, 48, 69, 98, 118],  # Vermeidung
            "Flu": [14, 24, 40, 62, 73, 120],  # Flucht
            "Soza": [6, 27, 49, 76, 92, 107],  # Soziale Abkapselung
            "Gedw": [16, 23, 55, 72, 100, 110],  # Gedankliche Weiterbeschäftigung
            "Res": [4, 32, 46, 60, 89, 105],  # Resignation
            "Selmit": [13, 41, 51, 79, 94, 117],  # Selbstbemitleidung
            "Sesch": [9, 25, 35, 57, 75, 87],  # Selbstbeschuldigung
            "Agg": [33, 44, 61, 82, 93, 112],  # Aggression
            "Pha": [7, 19, 53, 70, 90, 108],  # Pharmakaeinnahme
        }
    svf = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}

    svf = pd.DataFrame(svf, index=data.index)

    names = ["Pos1", "Pos2", "Pos3", "Pos_Gesamt", "Neg_Gesamt"]
    subscales = [
        ("Bag", "Her", "Schab"),
        ("Abl", "Ers", "Sebest", "Entsp"),
        ("Sitkon", "Rekon", "Posi"),
        (
            "Bag",
            "Her",
            "Schab",
            "Abl",
            "Ers",
            "Sebest",
            "Entsp",
            "Sitkon",
            "Rekon",
            "Posi",
        ),
        ("Flu", "Soza", "Gedw", "Res", "Selmit", "Sesch"),
    ]

    for n, subsc in zip(names, subscales):
        svf["{}_{}".format(score_name, n)] = svf[["{}_{}".format(score_name, s) for s in subsc]].mean(axis=1)

    return svf


def svf_42(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Stressverarbeitungsfragebogen - 42 items"""

    score_name = "SVF42"
    score_range = [1, 5]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if idxs is None:
        idxs = {
            "Bag": [7, 22],  # Bagatellisierung
            "Her": [11, 35],  # Herunterspielen
            "Schab": [2, 34],  # Schuldabwehr
            "Abl": [1, 32],  # Ablenkung
            "Ers": [12, 42],  # Ersatzbefriedigung
            "Sebest": [19, 37],  # Selbstbestätigung
            "Entsp": [13, 26],  # Entspannung
            "Sitkon": [4, 23],  # Situationskontrolle
            "Rekon": [17, 33],  # Reaktionskontrolle
            "Posi": [9, 24],  # Positive Selbstinstruktion
            "Sozube": [14, 27],  # Soziales Unterstützungsbedürfnis
            "Verm": [6, 30],  # Vermeidung
            "Flu": [16, 40],  # Flucht
            "Soza": [20, 29],  # Soziale Abkapselung
            "Gedw": [10, 25],  # Gedankliche Weiterbeschäftigung
            "Res": [38, 15],  # Resignation
            "Hilf": [18, 28],  # Hilflosigkeit
            "Selmit": [8, 31],  # Selbstbemitleidung
            "Sesch": [21, 36],  # Selbstbeschuldigung
            "Agg": [3, 39],  # Aggression
            "Pha": [5, 41],  # Pharmakaeinnahme
        }
    svf = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}

    svf = pd.DataFrame(svf, index=data.index)

    names = ["Denial", "Distraction", "Stressordevaluation"]
    subscales = [
        ("Flu", "Verm", "Soza"),
        ("Ers", "Entsp", "Sozube"),
        ("Bag", "Her", "Posi"),
    ]

    for n, subsc in zip(names, subscales):
        svf["{}_{}".format(score_name, n)] = svf[["{}_{}".format(score_name, s) for s in subsc]].mean(axis=1)

    return svf


def brief_cope(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Brief-COPE - 28 items"""

    score_name = "BriefCope"
    score_range = [1, 4]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if idxs is None:
        idxs = {
            "Self_Distraction": [1, 19],  # Ablenkung
            "Active_Coping": [2, 7],  # Aktive Bewältigung
            "Denial": [3, 8],  # Verleugnung
            "Substance_Use": [4, 11],  # Alkohol/Drogen
            "Emotional_Support": [5, 15],  # Emotionale Unterstützung
            "Instrumental_Support": [10, 23],  # Instrumentelle Unterstützung
            "Behavioral_Disengagement": [6, 16],  # Verhaltensrückzug
            "Venting": [9, 21],  # Ausleben von Emotionen
            "Pos_Reframing": [12, 17],  # Positive Umdeutung
            "Planning": [14, 25],  # Planung
            "Humor": [18, 28],  # Humor
            "Acceptance": [20, 24],  # Akzeptanz
            "Religion": [22, 27],  # Religion
            "Self_Blame": [13, 26],  # Selbstbeschuldigung
        }
    cope = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}

    return pd.DataFrame(cope, index=data.index)


def bfi_k(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Big Five Inventory - Kurzversion"""

    score_name = "BFI-K"
    score_range = [1, 5]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert items 1, 2, 8, 9, 11, 12, 17, 21
    data = invert(data, cols=to_idx([1, 2, 8, 9, 11, 12, 17, 21]), score_range=score_range)

    if idxs is None:
        idxs = {
            "E": [1, 6, 11, 16],  # Extraversion
            "V": [2, 7, 12, 17],  # Verträglichkeit
            "G": [3, 8, 13, 18],  # Gewissenhaftigkeit
            "N": [4, 9, 14, 19],  # Neurotizismus
            "O": [5, 10, 15, 20, 21],  # Offenheit für neue Erfahrungen
        }

    bfik = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].mean(axis=1) for key in idxs}

    return pd.DataFrame(bfik, index=data.index)


def rsq(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Response Styles Questionnaire"""

    score_name = "RSQ"
    score_range = [1, 4]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if idxs is None:
        idxs = {
            "SympRum": [2, 3, 4, 8, 11, 12, 13, 25],  # Symptomfokussierte Rumination
            "SelbstRum": [1, 19, 26, 28, 30, 31, 32],  # Selbstfokussierte Rumination
            "Distract": [5, 6, 7, 9, 14, 16, 18, 20],  # Distraktion
        }

    rsq_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].mean(axis=1) for key in idxs}
    rsq_data = pd.DataFrame(rsq_data, index=data.index)

    # invert items 5, 6, 7, 9, 14, 16, 18, 20 to add "Distract" subscale to total score
    rsq_data["{}_{}".format(score_name, "Distract")] = (
        invert(data, cols=to_idx(idxs["Distract"]), score_range=score_range, inplace=False)
        .iloc[:, to_idx(idxs["Distract"])]
        .mean(axis=1)
    )
    rsq_data[score_name] = pd.DataFrame(rsq_data, index=data.index).mean(axis=1)
    return rsq_data


def sss(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Subjektiver Sozialer Status"""

    score_name = "SSS"
    score_range = [1, 10]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def fkk(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Fragebogen zur Kompetenz- und Kontrollüberzeugungen"""

    score_name = "FKK"
    score_range = [1, 6]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert items 4, 8, 12, 24
    data = invert(data, cols=to_idx([4, 8, 12, 24]), score_range=score_range)

    if idxs is None:
        # Primärskalenwerte
        idxs = {
            "SK": [4, 8, 12, 24, 16, 20, 28, 32],
            "I": [1, 5, 6, 11, 23, 25, 27, 30],
            "P": [3, 10, 14, 17, 19, 22, 26, 29],
            "C": [2, 7, 9, 13, 15, 18, 21, 31],
        }
    fkk_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}
    fkk_data = pd.DataFrame(fkk_data, index=data.index)

    # Sekundärskalenwerte
    fkk_data[score_name + "_SKI"] = fkk_data[score_name + "_SK"] + fkk_data[score_name + "_I"]
    fkk_data[score_name + "_PC"] = fkk_data[score_name + "_P"] + fkk_data[score_name + "_C"]
    # Tertiärskalenwerte
    fkk_data[score_name + "_SKI_PC"] = fkk_data[score_name + "_SKI"] - fkk_data[score_name + "_PC"]

    return fkk_data


def bidr(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Balanced Inventory of Desirable Responding"""

    score_name = "BIDR"
    score_range = [1, 7]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert items 2, 4, 5, 7, 9, 10, 11, 12, 14, 15, 17, 18, 20 => invert all and re-invert the others
    data = invert(data, score_range=score_range)
    data = invert(data, cols=to_idx([1, 3, 6, 8, 13, 16, 19]), score_range=score_range)

    if idxs is None:
        idxs = {
            "ST": np.arange(1, 11),  # Selbsttäuschung
            "FT": np.arange(11, 21),  # Fremdtäuschung
        }

    bidr_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}
    return pd.DataFrame(bidr_data, index=data.index)


def kkg(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Kontrollüberzeugungen zu Krankheit und Gesundheit"""

    score_name = "KKG"
    score_range = [1, 6]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    if idxs is None:
        idxs = {
            "I": [1, 5, 8, 16, 17, 18, 21],
            "P": [2, 4, 6, 10, 12, 14, 20],
            "C": [3, 7, 9, 11, 13, 15, 19],
        }

    kkg_data = {score_name + "_" + key: data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}
    return pd.DataFrame(kkg_data, index=data.index)


def thoughts_questionnaire(
    data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None
) -> pd.DataFrame:
    """Thoughts Questionnaire"""

    score_name = "Thoughts"
    score_range = [1, 5]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert items 1, 6, 9, 12, 15, 17, 18, 20, 27
    data = invert(data, cols=to_idx([1, 6, 9, 12, 15, 17, 18, 20, 27]), score_range=score_range)
    return pd.DataFrame(data.sum(axis=1), columns=[score_name])


def fee(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    questionnaire_version: Optional[str] = "german",
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Fragebogen zum erinnerten elterlichen Erziehungsverhalten"""

    score_name = "FEE"
    score_range = [1, 4]
    supported_versions = ["english", "german"]

    if questionnaire_version not in supported_versions:
        raise AttributeError(
            "questionnaire_version must be one of {}, not {}.".format(supported_versions, questionnaire_version)
        )

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    df_mother = pd.DataFrame()
    df_father = pd.DataFrame()
    if questionnaire_version == "german":
        df_mother = data.filter(like="Mutter").copy()
        df_father = data.filter(like="Vater").copy()
    elif questionnaire_version == "english":
        df_mother = data.filter(like="Mother").copy()
        df_father = data.filter(like="Father").copy()

    if idxs is None:
        idxs = {
            "Ablehnung_Strafe": [1, 3, 6, 8, 16, 18, 20, 22],
            "Emot_Waerme": [2, 7, 9, 12, 14, 15, 17, 24],
            "Kontrolle": [4, 5, 10, 11, 13, 19, 21, 23],
        }

    fee_mother = {
        "{}_{}_Mother".format(score_name, key): df_mother.iloc[:, to_idx(idxs[key])].mean(axis=1) for key in idxs
    }
    fee_father = {
        "{}_{}_Father".format(score_name, key): df_father.iloc[:, to_idx(idxs[key])].mean(axis=1) for key in idxs
    }
    fee_mother.update(fee_father)

    return pd.DataFrame(fee_mother, index=data.index)


def mbi(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Maslach Burnout Inventory"""

    score_name = "MBI"
    score_range = [1, 6]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    mbi_type = data.iloc[:, 0] - 1
    mbi_type.name = "{}_Type".format(score_name)
    data.drop(axis=1, labels=data.columns[0], inplace=True)
    # MBI in HABIT was assessed in the regular and Student form,
    # depending on the subject => 2 questionnaires, split into 2 dataframes
    items = np.split(data, 2, axis=1)
    for i in [0, 1]:
        items[i] = items[i][mbi_type == i]
        items[i].columns = items[0].columns
    data = pd.concat(items).sort_index()

    if idxs is None:
        idxs = {
            "EmotErsch": [1, 2, 3, 4, 5],
            "PersErf": [6, 7, 8, 11, 12, 16],
            "Deperson": [9, 10, 13, 14, 15],
        }

    mbi_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].mean(axis=1) for key in idxs}

    data = pd.DataFrame(mbi_data, index=data.index)
    data[mbi_type.name] = mbi_type
    return data


def mlq(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Meaning in Life Questionnaire"""

    score_name = "MLQ"
    score_range = [1, 7]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert item 9
    data = invert(data, cols=to_idx([9]), score_range=score_range)

    if idxs is None:
        idxs = {
            "PresenceMeaning": [1, 4, 5, 6, 9],
            "SearchMeaning": [2, 3, 7, 8, 10],
        }

    mlq_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].mean(axis=1) for key in idxs}
    return pd.DataFrame(mlq_data, index=data.index)


def ceca(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """Childhood Experiences of Care and Abuse"""

    score_name = "CECA"

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    ceca_data = [
        data.filter(like="Q3_05"),
        data.filter(like="Q3_07"),
        data.filter(like="Q3_09"),
        data.filter(like="Q3_12").iloc[:, to_idx([5, 6])],
        data.filter(like="Q3_13"),
        data.filter(like="Q3_16").iloc[:, to_idx([5, 6])],
    ]

    ceca_data = pd.concat(ceca_data, axis=1).sum(axis=1)
    return pd.DataFrame(ceca_data, index=data.index, columns=[score_name])


def pfb(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Partnerschaftsfragebogen"""

    score_name = "PFB"
    score_range = [1, 4]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert item 19
    data = invert(data, cols=to_idx([19]), score_range=score_range)

    if idxs is None:
        idxs = {
            "Zaertlichkeit": [2, 3, 5, 9, 13, 14, 16, 23, 27, 28],
            "Streitverhalten": [1, 6, 8, 17, 18, 19, 21, 22, 24, 26],
            "Gemeinsamkeit": [4, 7, 10, 11, 12, 15, 20, 25, 29, 30],
            "Glueck": [31],
        }

    pfb_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}

    pfb_data[score_name] = data.iloc[:, 0:30].sum(axis=1)
    return pd.DataFrame(pfb_data, index=data.index)


def asq(data: pd.DataFrame, columns: Optional[Union[Sequence[str], pd.Index]] = None) -> pd.DataFrame:
    """
    **Anticipatory Stress Questionnaire (ASQ)**

    The ASQ measures anticipation of stress on the upcoming day.

    NOTE: This implementation assumes a score range of [1, 11]. Use ``bp.questionnaires.utils.convert_scale()`` to
    convert the items into the correct range.


    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if `columns` parameter is supplied
    columns : list of string, optional
        list with column names to use for computing this score if a complete dataframe is supplied.
        See ``bp.questionnaires.utils.convert_scale()``

    Returns
    -------
    pd.DataFrame
        ASQ score


    References
    ------------
    Powell, D. J., & Schlotz, W. (2012). Daily Life Stress and the Cortisol Awakening Response:
    Testing the Anticipation Hypothesis. *PLoS ONE*, 7(12), e52067. https://doi.org/10.1371/journal.pone.0052067
    """

    score_name = "ASQ"
    score_range = [1, 11]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert items 2,3
    data = invert(data, cols=to_idx([2, 3]), score_range=score_range)

    return pd.DataFrame(data.mean(axis=1), columns=[score_name])


def mdbf(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """
    **Multidimensionaler Befindlichkeitsfragebogen (MDBF)**

    The MDBF measures different bipolar dimensions of current mood and psychological
    wellbeing.

    It consists of three subscales:

    * Good-Bad mood (`GoodBad`)
    * Awake-Tired (`AwakeTired`)
    * Calm-Nervous (`CalmNervous`)

    NOTE: This implementation assumes a score range of [1, 5]. Use ``bp.questionnaires.utils.convert_scale()`` to
    convert the items into the correct range.


    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if `columns` parameter is supplied
    columns : list of string, optional
        list with column names to use for computing this score if a complete dataframe is supplied.
        See ``bp.questionnaires.utils.convert_scale()``

    Returns
    -------
    pd.DataFrame
        MDBF score

    References
    ------------
    Steyer, R., Schwenkmezger, P., Notz, P., & Eid, M. (1997). Der Mehrdimensionale Befindlichkeitsfragebogen MDBF
    [Multidimensional mood questionnaire]. *Göttingen, Germany: Hogrefe*.

    """

    score_name = "MDBF"
    score_range = [1, 5]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    _assert_value_range(data, score_range)

    # invert items 3, 4, 5, 7, 9, 11, 13, 16, 18, 19, 22, 23
    data = invert(
        data,
        cols=to_idx([3, 4, 5, 7, 9, 11, 13, 16, 18, 19, 22, 23]),
        score_range=score_range,
    )

    if idxs is None:
        idxs = {
            "GoodBad": [1, 4, 8, 11, 14, 16, 18, 21],
            "AwakeTired": [2, 5, 7, 10, 13, 17, 20, 23],
            "CalmNervous": [3, 6, 9, 12, 15, 19, 22, 24],
        }

    mdbf_data = {"{}_{}".format(score_name, key): data.iloc[:, to_idx(idxs[key])].sum(axis=1) for key in idxs}

    mdbf_data[score_name] = data.sum(axis=1)

    return pd.DataFrame(mdbf_data, index=data.index)


def meq(
    data: pd.DataFrame,
    columns: Optional[Union[Sequence[str], pd.Index]] = None,
    idxs: Optional[Dict[str, Sequence[int]]] = None,
) -> pd.DataFrame:
    """
    **Morningness Eveningness Questionnaire (MEQ)**

    The MDBF measures different bipolar dimensions of current mood and psychological
    wellbeing.

    It consists of three subscales:

    * Good-Bad mood (`GoodBad`)
    * Awake-Tired (`AwakeTired`)
    * Calm-Nervous (`CalmNervous`)

    NOTE: This implementation assumes a score range of [1, 4], with some items having a score range of [1, 5].
    Use ``bp.questionnaires.utils.convert_scale()`` to convert the items into the correct range.


    Parameters
    ----------
    data : pd.DataFrame
        dataframe containing questionnaire data. Can either be only the relevant columns for computing this score or
        a complete dataframe if `columns` parameter is supplied
    columns : list of string, optional
        list with column names to use for computing this score if a complete dataframe is supplied.
        See ``bp.questionnaires.utils.convert_scale()``

    Returns
    -------
    pd.DataFrame
        MEQ score and Chronotype Classification in two levels:
        * 5 levels ('Chronotype_Fine'):
            - 0: definite evening type (MEQ score 14-30)
            - 1: moderate evening type (MEQ score 31-41)
            - 2: intermediate type (MEQ score 42-58)
            - 3: moderate morning type (MEQ score 59-69)
            - 4: definite morning type (MEQ score 70-86)
        * 3 levels ('Chronotype_Coarse'):
            - 0: evening type (MEQ score 14-41)
            - 1: intermediate type (MEQ score 42-58)
            - 2: morning type (MEQ score 59-86)

    References
    ------------
    Horne, J. A., & Östberg, O. (1976). A self-assessment questionnaire to determine morningness-eveningness in
    human circadian rhythms. International journal of chronobiology.

    """

    score_name = "MEQ"
    score_range = [1, 4]

    if columns is not None:
        # if columns parameter is supplied: slice columns from dataframe
        data = data.loc[:, columns]

    # some columns have scores from 1-5 => check them separately
    col_idx = to_idx([1, 2, 10, 17, 18])
    try:
        _assert_value_range(data.iloc[:, col_idx], [1, 5])
        col_mask = np.arange(0, len(data.columns))
        col_mask = col_mask[~np.isin(col_mask, to_idx([1, 2, 10, 17, 18]))]
        _assert_value_range(data.iloc[:, col_mask], score_range)
    except ValueError:
        raise ValueError(
            "Attention! This implementation of MEQ expects all values in the range {}, except the columns {}, "
            "which are expected to be in the range {}! "
            "Please consider converting to the correct range using "
            "`biopsykit.questionnaire.utils.convert_scale`.".format(score_range, col_idx, [1, 5])
        )

    # invert items 1, 2, 10, 17, 18 (score range [1,5])
    invert(data, cols=to_idx([1, 2, 10, 17, 18]), score_range=[1, 5], inplace=True)
    # invert items 3, 8, 9, 10, 11, 13, 15, 19 (score range [1,4])
    invert(
        data,
        cols=to_idx([3, 8, 9, 11, 13, 15, 19]),
        score_range=score_range,
        inplace=True,
    )

    # recode items 11, 12, 19
    data.iloc[:, to_idx(11)].replace({1: 0, 2: 2, 3: 4, 4: 6}, inplace=True)
    data.iloc[:, to_idx(12)].replace({1: 0, 2: 2, 3: 3, 4: 5}, inplace=True)
    data.iloc[:, to_idx(19)].replace({1: 0, 2: 2, 3: 4, 4: 6}, inplace=True)

    meq_data = pd.DataFrame(np.sum(data, axis=1), columns=[score_name])
    meq_data["Chronotype_Fine"] = bin_scale(meq_data[score_name], bins=[0, 30, 41, 58, 69, 86])
    meq_data["Chronotype_Coarse"] = bin_scale(meq_data[score_name], bins=[0, 41, 58, 86])

    return meq_data
