"""Module implementing a base class to represent psychological protocols."""
from typing import Dict, Sequence, Union, Tuple, Optional, Any, Iterable

import pandas as pd
import matplotlib.pyplot as plt

import biopsykit.protocols.plotting as plot
from biopsykit.protocols.utils import _check_sample_times_match, _get_sample_times
from biopsykit.utils.data_processing import (
    resample_dict_sec,
    select_dict_phases,
    normalize_to_phase,
    add_subject_conditions,
    cut_phases_to_shortest,
    mean_per_subject_dict,
    rearrange_subject_data_dict,
    split_dict_into_subphases,
    merge_study_data_dict,
    mean_se_per_phase,
    split_subject_conditions,
)
from biopsykit.utils.datatype_helper import (
    HeartRateSubjectDataDict,
    is_study_data_dict,
    SalivaRawDataFrame,
    is_saliva_raw_dataframe,
)
from biopsykit.utils.exceptions import ValidationError


class BaseProtocol:
    """Base class representing a psychological protocol and data collected within a study.

    The general structure of the protocol can be specified by passing a ``structure`` dict to the constructor of
    ``BaseProtocol``.

    Up to three nested structure levels are supported:
        * 1st level: ``study part``: Different parts of the study, such as: "Preface", "Test",
          and "Questionnaires"
        * 2nd level: ``phase``: Different phases of the psychological protocol that belong to the same *study
          part*, such as: "Preparation", "Stress", "Recovery"
        * 3rd level: ``subphase``: Different subphases that belong to the same *phase*, such as:
          "Baseline", "Arithmetic Task", "Feedback"

    .. note::
        Duration of phases and/or subphases are expected in **seconds**.


    Examples
    --------
    >>> from biopsykit.protocols import BaseProtocol
    >>> # Example 1: study with three parts, no finer division into phases
    >>> structure = {
    >>>     "Preface": None,
    >>>     "Test": None,
    >>>     "Questionnaires": None
    >>> }
    >>> BaseProtocol(name="Base", structure=structure)
    >>> # Example 2: study with three parts, all parts have different phases with specific durations
    >>> structure = {
    >>>     "Preface": {"Questionnaires": 240, "Baseline": 60},
    >>>     "Test": {"Preparation": 120, "Test": 240, "Recovery": 120},
    >>>     "Recovery": {"Part1": 240, "Part2": 240}
    >>> }
    >>> BaseProtocol(name="Base", structure=structure)
    >>> # Example 3: only certain study parts have different phases (example: TSST)
    >>> structure = {
    >>>     "Before": None,
    >>>     "TSST": {"Preparation": 300, "Talk": 300, "Math": 300},
    >>>     "After": None
    >>> }
    >>> BaseProtocol(name="Base", structure=structure)
    >>> # Example 4: study with phases and subphases, only certain study parts have different phases (example: MIST)
    >>> structure = {
    >>>     "Before": None,
    >>>     "MIST": {
    >>>         "MIST1": {"BL": 60, "AT": 240, "FB": 120},
    >>>         "MIST2": {"BL": 60, "AT": 240, "FB": 120},
    >>>         "MIST3": {"BL": 60, "AT": 240, "FB": 120}
    >>>     },
    >>>     "After": None
    >>> }
    >>> BaseProtocol(name="Base", structure=structure)

    """

    def __init__(
        self,
        name: str,
        structure: Optional[Dict[str, Any]] = None,
        test_times: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Initialize a new ``BaseProtocol`` instance.

        Parameters
        ----------
        name : str
            name of protocol
        structure : dict, optional
            nested dictionary specifying the structure of the protocol.
            Up to three nested structure levels are supported:
                * 1st level: ``study_part``: Different parts of the study, such as: "Preface", "Test",
                  and "Questionnaires"
                * 2nd level: ``phase``: Different phases of the psychological protocol that belong to the same *study
                  part*, such as: "Preparation", "Stress", "Recovery"
                * 3rd level: ``subphase``: Different subphases that belong to the same *phase*, such as:
                  "Baseline", "Arithmetic Task", "Feedback"

            If a study part has no division into finer phases (or a phase has no division into finer subphases) the
            dictionary value can be set to ``None``. If the whole study has no division into different parts, the
            ``structure`` dict can be set to ``None``. Default: ``None``
        test_times : list, optional
            start and end time of psychological test (in minutes) or ``None`` if the protocol has no particular test.
            ``test_times`` is then internally set to [0, 0]. Default: ``None``
        **kwargs
            additional parameters to be passed to ``BaseProtocol``, such as:
                * ``saliva_plot_params``: dictionary with parameters to style
                  :meth:`~biopsykit.protocols.base.BaseProtocol.saliva_plot`
                * ``hr_mean_plot_params``: dictionary with parameters to style
                  :meth:`~biopsykit.protocols.base.BaseProtocol.hr_mean_plot`
                * ``hr_ensemble_plot_params``: dictionary with parameters to style
                  :meth:`~biopsykit.protocols.base.BaseProtocol.hr_ensemble_plot`

        """
        self.name: str = name
        """Study or protocol name"""

        self.structure: Dict[str, Any] = structure
        """Structure of protocol, i.e., whether protocol if divided into different parts, phases, or subphases.

        If protocol is not divided into different parts ``protocol_structure`` is set to ``None``.
        """

        self.saliva_types: Sequence[str] = []
        """List of saliva data types present in the study."""

        if test_times is None:
            test_times = [0, 0]
        self.test_times: Sequence[int] = test_times
        """Start and end time of psychological test (in minutes).

        If no psychological test was performed in the protocol ``test_times`` is set to [0, 0].
        """

        self.sample_times: Dict[str, Sequence[int]] = {}
        """Dictionary with sample times of saliva samples (in minutes).

        Sample times are either provided explicitly using the ``sample_times`` parameter in
        :meth:`~biopsykit.protocols.base.BaseProtocol.add_saliva_data` or by extracting it from the saliva data
        (if a ``time`` column is present).
        """

        self.saliva_data: Dict[str, SalivaRawDataFrame] = {}
        """Dictionary with saliva data collected during the study.

        Data in :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame` format can be added using
        :meth:`~biopsykit.protocols.base.BaseProtocol.add_saliva_data`.
        """

        self.hr_data: Dict[str, HeartRateSubjectDataDict] = {}
        """Dictionary with heart rate data collected during the study.
        If the study consists of multiple study parts each part has its own ``HeartRateSubjectDataDict``.
        If the study has no individual study parts (only different phases), the name of the one and only study part
        defaults to ``Study`` (to ensure consistent dictionary structure).

        Data in :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict` format can be added using
        :meth:`~biopsykit.protocols.base.BaseProtocol.add_hr_data`.
        """

        self.hr_results: Dict[str, pd.DataFrame] = {}
        """Dictionary with heart rate results.

        Dict keys are the identifiers that are specified when computing results from ``hr_data`` using
        :meth:`~biopsykit.protocols.base.BaseProtocol.compute_hr_results`.
        """

        self.hr_ensemble: Dict[str, Dict[str, pd.DataFrame]] = {}
        """Dictionary with merged heart rate data for heart rate ensemble plot.


        Dict keys are the identifiers that are specified when computing ensemble HR data from ``hr_data`` using
        :meth:`~biopsykit.protocols.base.BaseProtocol.compute_hr_ensemble`.

        See Also
        --------
        :meth:`~biopsykit.protocols.base.BaseProtocol.hr_ensemble_plot`
            heart rate ensemble plot
        """

        self.saliva_plot_params: Dict[str, Any] = kwargs.get("saliva_plot_params", {})
        """Plot parameters for customizing the general `saliva plot` for a specific psychological protocol.

        See Also
        --------
        :meth:`~biopsykit.protocols.base.BaseProtocol.saliva_plot`
            saliva plot
        """

        self.hr_mean_plot_params: Dict[str, Any] = kwargs.get("hr_mean_plot_params", {})
        """Plot parameters for customizing the general `HR mean plot` for a specific psychological protocol.

        See Also
        --------
        :meth:`~biopsykit.protocols.base.BaseProtocol.hr_mean_plot`
            HR mean plot
        """

        self.hr_ensemble_plot_params: Dict[str, Any] = kwargs.get("hr_ensemble_plot_params", {})
        """Plot parameters for customizing the general `HR ensemble plot` for a specific psychological protocol.

        See Also
        --------
        :meth:`~biopsykit.protocols.base.BaseProtocol.hr_ensemble_plot`
            HR ensemble plot
        """

    def __repr__(self) -> str:
        """Return string representation of Protocol instance.

        Returns
        -------
        str
            string representation of Protocol instance

        """
        return self.__str__()

    def __str__(self) -> str:
        """Return string representation of Protocol instance.

        Returns
        -------
        str
            string representation of Protocol instance

        """
        if len(self.saliva_data) > 0:
            return """{}
            Saliva Type(s): {}
            Saliva Sample Times: {}
            Structure: {}
            """.format(
                self.name, self.saliva_types, self.sample_times, self.structure
            )
        return """{}
        Structure: {}""".format(
            self.name, self.structure
        )

    def add_saliva_data(
        self,
        saliva_type: Union[str, Sequence[str]],
        saliva_data: Union[SalivaRawDataFrame, Dict[str, SalivaRawDataFrame]],
        sample_times: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
        test_times: Optional[Sequence[int]] = None,
    ):
        """Add saliva data collected during psychological protocol to ``Protocol`` instance.

        Parameters
        ----------
        saliva_type : str or list of str
            saliva type (or list of such) of saliva data
        saliva_data : :obj:`~biopsykit.utils.datatype_helper.SalivaRawDataFrame` or dict
            saliva data (or dict of such) to be added to this protocol.
        sample_times : list of int or dict, optional
            list of sample times in minutes. Sample times are expected to be provided *relative* to the psychological
            test in the protocol (if present). Per convention, a sample collected **directly before** was collected at
            time point $t = -1$, a sample collected **directly after** the test was collected at time point $t = 0$.
        test_times : list of int, optional
            list with start and end time of psychological test in minutes. Per convention, the start of the test
            should be at time point $t = 0$. ``test_times`` is also used to compute the **absolute** sample times

        """
        if isinstance(saliva_type, str):
            saliva_type = [saliva_type]
        self.saliva_types = saliva_type

        if test_times is not None:
            self.test_times = test_times

        if saliva_data is not None:
            if not isinstance(sample_times, dict):
                sample_times = {key: sample_times for key in self.saliva_types}
            if not isinstance(saliva_data, dict):
                saliva_data = {key: saliva_data for key in self.saliva_types}
            self.sample_times.update(_get_sample_times(saliva_data, sample_times, self.test_times))
            self.saliva_data.update(self._add_saliva_data(saliva_data, self.saliva_types, self.sample_times))

    def _add_saliva_data(
        self,
        data: Union[SalivaRawDataFrame, Dict[str, SalivaRawDataFrame]],
        saliva_type: Union[str, Sequence[str]],
        sample_times: Union[Sequence[int], Dict[str, Sequence[int]]],
    ) -> Union[SalivaRawDataFrame, Dict[str, SalivaRawDataFrame]]:
        saliva_data = {}
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    saliva_data[key] = self._add_saliva_data(value, key, sample_times[key])
                return saliva_data
            is_saliva_raw_dataframe(data, saliva_type)
            _check_sample_times_match(data, sample_times)
            return data
        except ValidationError as e:
            raise ValidationError(
                "'data' is not a 'SalivaRawDataFrame' (or a dict of such). "
                "The validation raised the following error:\n\n{}".format(str(e))
            ) from e

    def add_hr_data(self, data: HeartRateSubjectDataDict, study_part: Optional[str] = None):
        """Add time-series heart rate data collected during psychological protocol to ``Protocol`` instance.

        Parameters
        ----------
        data : :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDataDict`
            dictionary heart rate data of all subjects collected during the protocol
        study_part : str, optional
            string indicating to which study part data belongs to or ``None`` if data has no individual study parts.
            Default: ``None``

        """
        is_study_data_dict(data)
        if study_part is None:
            study_part = "Study"
        self.hr_data[study_part] = data

    def compute_hr_results(
        self,
        result_id: str,
        study_part: Optional[str] = None,
        resample_sec: Optional[bool] = True,
        normalize_to: Optional[bool] = False,
        select_phases: Optional[bool] = False,
        split_into_subphases: Optional[bool] = False,
        mean_per_subject: Optional[bool] = True,
        add_conditions: Optional[bool] = False,
        params: Dict[str, Any] = None,
    ):
        """Compute heart rate data results from one study part.

        The different processing steps can be enabled or disabled by setting the function parameters to
        ``True`` or ``False``, respectively. Parameters that are required for a specific processing step can be
        provided in the ``params`` dict. The dict key must match the name of the processing step.

        Parameters
        ----------
        result_id : str
            Result ID, a descriptive name of the results that were computed.
            This ID will also be used as key to store the computed results in the ``hr_results`` dictionary
        study_part : str, optional
            study part the data which should be processed belongs to or ``None`` if data has no
            individual study parts.
            Default: ``None``
        resample_sec : bool, optional
            ``True`` to apply resampling. Instantaneous heart rate data will then be resampled to 1 Hz.
            Default: ``True``
        normalize_to : bool, optional
            ``True`` to normalize heart rate data per subject. Data will then be the heart rate increase relative to
            the average heart rate in the phase. The name of the phase (or a dataframe containing heart rate
            data to normalize to) is specified in the ``params`` dictionary (key: ``normalize_to``).
            Default: ``False``
        select_phases : bool, optional
            ``True`` to only select specific phases for further processing, ``False`` to use all data from
            ``study_part``. The phases to be selected are specified in the ``params`` dictionary
            (key: ``select_phases``).
            Default: ``False``
        split_into_subphases : bool, optional
            ``True`` to further split phases into subphases, ``False`` otherwise. The subphases are provided as
            dictionary (keys: subphase names, values: subphase durations in seconds) in the ``params`` dictionary
            (key: ``split_into_subphases``).
            Default: ``False``
        mean_per_subject : bool, optional
            ``True`` to compute the mean heart rate per phase (and subphase, if present) for each subject and
            combine results into one dataframe. The resulting index level names of the dataframe can be provided as list
            in the ``params`` dictionary (key: ``mean_per_subject``). By default, the index level names are
            ["subject", "phase"].
            Default: ``True``
        add_conditions : bool, optional
            ``True`` to add subject conditions to dataframe data. Information on which subject belongs to which
            condition can be provided as :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame` or
            :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDict` in the ``params`` dictionary
            (key: ``add_conditions``).
            Default: ``False``
        params : dict, optional
            dictionary with parameters provided to the different processing steps

        """
        if study_part is None:
            study_part = "Study"
        data_dict = self.hr_data[study_part].copy()

        if params is None:
            params = {}

        if resample_sec:
            data_dict = resample_dict_sec(data_dict)

        if normalize_to:
            param = params.get("normalize_to", None)
            data_dict = normalize_to_phase(data_dict, param)

        if select_phases:
            param = params.get("select_phases", None)
            data_dict = select_dict_phases(data_dict, param)

        if split_into_subphases:
            param = params.get("split_into_subphases", None)
            data_dict = split_dict_into_subphases(data_dict, param)

        if mean_per_subject:
            param = params.get("mean_per_subject", ["subject", "phase"])
            data_dict = mean_per_subject_dict(data_dict, param, "Heart_Rate")

        if add_conditions:
            param = params.get("add_conditions", None)
            data_dict = add_subject_conditions(data_dict, param)

        self.hr_results[result_id] = data_dict

    def compute_hr_ensemble(
        self,
        ensemble_id: str,
        study_part: Optional[str] = None,
        resample_sec: Optional[bool] = True,
        normalize_to: Optional[bool] = True,
        cut_phases: Optional[bool] = True,
        merge_dict: Optional[bool] = True,
        add_conditions: Optional[bool] = False,
        params: Dict[str, Any] = None,
    ):
        """Compute heart rate ensemble data from one study part.

        Heart rate ensemble data are time-series data where data from all subjects within one phase
        have the same length and can thus be overlaid as mean ± standard error in a plot.

        The different processing steps can be enabled or disabled by setting the function parameters to
        ``True`` or ``False``, respectively. Parameters that are required for a specific processing step can be
        provided in the ``params`` dict. The dict key must match the name of the processing step.


        Parameters
        ----------
        ensemble_id : str
            ensemble identifier, a descriptive name of the ensemble data that were computed.
            This ID will also be used as key to store the computed ensemble data in the ``hr_ensemble`` dictionary
        study_part : str, optional
            study part the data which should be processed belongs to or ``None`` if data has no
            individual study parts.
            Default: ``None``
        resample_sec : bool, optional
            ``True`` to apply resampling. Instantaneous heart rate data will then be resampled to 1 Hz.
            Default: ``True``
        normalize_to : bool, optional
            ``True`` to normalize heart rate data per subject. Data will then be the heart rate increase relative to
            the average heart rate in the phase. The name of the phase (or a dataframe containing heart rate
            data to normalize to) is specified in the ``params`` dictionary (key: ``normalize_to``).
            Default: ``False``
        cut_phases : bool, optional
            ``True`` to cut time-series data to shortest duration of a subject in each phase, ``False`` otherwise.
            Default: ``True``
        merge_dict : bool, optional
            ``True`` to convert ``StudyDataDict`` into ``MergedStudyDataDict``, i.e., merge dictionary data from
            individual subjects into one dataframe for each phase.
            Default: ``True``
        add_conditions : bool, optional
            ``True`` to add subject conditions to dataframe data. Information on which subject belongs to which
            condition can be provided as :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDataFrame` or
            :obj:`~biopsykit.utils.datatype_helper.SubjectConditionDict` in the ``params`` dictionary
            (key: ``add_conditions``).
            Default: ``False``
        params : dict, optional
            dictionary with parameters provided to the different processing steps

        See Also
        --------
        `~biopsykit.protocols.plotting.hr_ensemble_plot`
            Heart rate ensemble plot

        """
        if study_part is None:
            study_part = "Study"
        data_dict = self.hr_data[study_part].copy()

        if resample_sec:
            data_dict = resample_dict_sec(data_dict)

        if normalize_to:
            param = params.get("normalize_to", None)
            data_dict = normalize_to_phase(data_dict, param)

        data_dict = rearrange_subject_data_dict(data_dict)

        if cut_phases:
            data_dict = cut_phases_to_shortest(data_dict)

        if merge_dict:
            data_dict = merge_study_data_dict(data_dict)

        if add_conditions:
            param = params.get("add_conditions", None)
            data_dict = split_subject_conditions(data_dict, param)

        self.hr_ensemble[ensemble_id] = data_dict

    def get_hr_results(self, result_id: str) -> pd.DataFrame:
        """Return heart rate processing results.

        Heart rate results can be computed by calling :meth:`~biopsykit.protocols.base.BaseProtocol.compute_hr_results`.

        Parameters
        ----------
        result_id : str
            identifier of result parameters specified when computing results via
            :meth:`~biopsykit.protocols.base.BaseProtocol.compute_hr_results`

        Returns
        -------
        :class:`~pandas.DataFrame`
            heart rate processing results

        """
        return self.hr_results.get(result_id, None)

    def get_hr_ensemble(self, ensemble_id: str):
        """Return heart rate ensemble data.

        Parameters
        ----------
        ensemble_id : str
            identifier of ensemble parameters specified when computing ensemble parameters via
            :meth:`~biopsykit.protocols.base.BaseProtocol.compute_hr_ensemble`

        Returns
        -------
        :class:`~pandas.DataFrame`
            heart rate ensemble results

        """
        return self.hr_ensemble.get(ensemble_id, None)

    def saliva_plot(
        self,
        saliva_type: Optional[Union[str, Sequence[str]]] = "cortisol",
        **kwargs,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot saliva data during psychological protocol as mean ± standard error.

        Parameters
        ----------
        saliva_type : {"cortisol", "amylase", "il6"}, optional
            saliva type to be plotted. If a dict is passed and ``saliva_type`` is ``None``
            the saliva types are inferred from dict keys. Default: ``cortisol``
        kwargs
            additional parameters to be passed to :func:`biopsykit.protocols.plotting.saliva_plot`.


        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        ax : :class:`matplotlib.axes.Axes`
            axes object

        See Also
        --------
        `biopsykit.protocols.plotting.saliva_plot`
            Plot saliva data during a psychological protocol

        """
        if len(self.saliva_types) == 0:
            raise ValueError("No saliva data to plot!")

        kwargs.update(self.saliva_plot_params)

        if isinstance(saliva_type, str):
            saliva_type = [saliva_type]

        data = {key: self.saliva_data[key] for key in saliva_type}
        sample_times = {key: self.sample_times[key] for key in saliva_type}

        return plot.saliva_plot(
            data=data,
            saliva_type=None,
            sample_times=sample_times,
            test_times=self.test_times,
            sample_times_absolute=True,
            **kwargs,
        )

    @staticmethod
    def saliva_plot_combine_legend(fig: plt.Figure, ax: plt.Axes, saliva_types: Sequence[str], **kwargs):
        """Combine multiple legends of ``saliva_plot`` into one joint legend outside of plot.

        If data from multiple saliva types are combined into one plot (e.g., by calling
        :func:`~biopsykit.protocols.plotting.saliva_plot` on the same plot twice) then two separate legend are created.
        This function can be used to combine the two legends into one.


        Parameters
        ----------
        fig : :class:`~matplotlib.figure.Figure`
            figure object
        ax : :class:`~matplotlib.axes.Axes`
            axes object
        saliva_types : list
            list of saliva types in plot
        kwargs
            additional arguments to customize plot that are passed to
            :func:`~biopsykit.protocols.plotting.saliva_plot_combine_legend`

        """
        return plot.saliva_plot_combine_legend(fig=fig, ax=ax, saliva_types=saliva_types, **kwargs)

    def saliva_feature_boxplot(
        self,
        x: str,
        saliva_type: str,
        feature: Optional[str] = None,
        stats_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Draw a boxplot with significance brackets, specifically designed for saliva features.

        This is a wrapper of :func:`~biopsykit.protocols.plotting.saliva_feature_boxplot` that directly uses the
        saliva data added to this ``Protocol`` instance.

        Parameters
        ----------
        x : str
            column of x axis in ``data``
        saliva_type : str
            type of saliva data to plot
        feature : str, optional
            name of feature to plot or ``None``
        stats_kwargs : dict, optional
            dictionary with arguments for significance brackets
        kwargs
            additional arguments that are passed to :func:`~biopsykit.protocols.plotting.saliva_feature_boxplot`


        See Also
        --------
        `biopsykit.protocols.plotting.saliva_feature_plot`
            plot saliva features as boxplot without ``Protocol`` instance
        `biopsykit.plotting.feature_boxplot`
            plot features as boxplot


        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        ax : :class:`matplotlib.axes.Axes`
            axes object

        """
        return plot.saliva_feature_boxplot(
            self.saliva_data[saliva_type], x, saliva_type, feature, stats_kwargs, **kwargs
        )

    def saliva_multi_feature_boxplot(
        self,
        saliva_type: str,
        features: Union[Sequence[str], Dict[str, Union[str, Sequence[str]]]],
        hue: Optional[str] = None,
        stats_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Iterable[plt.Axes]]:
        """Draw multiple features as boxplots with significance brackets, specifically designed for saliva features.

        This is a wrapper of :func:`~biopsykit.protocols.plotting.saliva_multi_feature_boxplot` that directly uses the
        saliva data added to this ``Protocol`` instance.

        Parameters
        ----------
        saliva_type : str
            type of saliva data to plot
        hue : str, optional
            column name of grouping variable. Default: ``None``
        features : list of str or dict of str
            features to plot. If ``features`` is a list, each entry must correspond to one feature category in the
            index level specified by ``group``. A separate subplot will be created for each feature.
            If similar features (i.e., different `slope` or `AUC` parameters) should be combined into one subplot,
            ``features`` can be provided as dictionary.
            Then, the dict keys specify the feature category (a separate subplot will be created for each category)
            and the dict values specify the feature (or list of features) that are combined into the subplots.
        stats_kwargs : dict, optional
            nested dictionary with arguments for significance brackets.
            See :func:`~biopsykit.plotting.feature_boxplot` for further information


        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        axs : list of :class:`matplotlib.axes.Axes`
            list of subplot axes objects


        See Also
        --------
        `biopsykit.plotting.saliva_multi_feature_boxplot`
            plot multiple saliva features as boxplots without instantiating a ``Protocol`` instance
            `biopsykit.stats.StatsPipeline`
                class to create statistical analysis pipelines and get parameter for plotting significance brackets

        """
        return plot.saliva_multi_feature_boxplot(
            self.saliva_data[saliva_type], saliva_type, features, hue, stats_kwargs, **kwargs
        )

    def hr_ensemble_plot(
        self, ensemble_id: str, subphases: Optional[Dict[str, Dict[str, int]]] = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Draw heart rate ensemble plot.

        Parameters
        ----------
        ensemble_id : str
            identifier of the ensemble data to be plotted.
            Ensemble data needs to be computed using ``compute_hr_ensemble`` first
        subphases : dict, optional
            dictionary with phases (keys) and subphases (values - dict with subphase names and subphase durations) or
            ``None`` if no subphases are present. Default: ``None``
        kwargs : dict, optional
            optional arguments for plot configuration to be passed to
            :func:`~biopsykit.protocols.plotting.hr_ensemble_plot`


        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        ax : :class:`matplotlib.axes.Axes`
            axes object


        See Also
        --------
        `compute_hr_ensemble`
            compute heart rate ensemble data
        `~biopsykit.protocols.plotting.hr_ensemble_plot`
            Heart rate ensemble plot

        """
        data = self.hr_ensemble[ensemble_id]
        kwargs.update(self.hr_ensemble_plot_params)
        return plot.hr_ensemble_plot(data=data, subphases=subphases, **kwargs)

    def hr_mean_plot(
        self,
        result_id: str,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        r"""Plot course of heart rate as mean ± standard error over phases (and subphases) of a psychological protocol.

        The correct plot is automatically inferred from the provided data:
            * only ``phase`` index level: plot phases over x axis
            * ``phase`` and ``subphase`` index levels: plot subphases over x axis, highlight phases as vertical spans
            * additionally: ``condition`` level: plot data of different conditions individually
              (corresponds to ``hue`` parameter in :func:`biopsykit.plotting.lineplot`)


        Parameters
        ----------
        result_id : str
            identifier of the heart rate result data to be plotted
        kwargs
            additional  parameters to be passed to the plot, such as:
                * ``ax``: pre-existing axes for the plot. Otherwise, a new figure and axes object is created
                  and returned.
                * ``colormap``: colormap to plot data from different phases
                * ``figsize``: tuple specifying figure dimensions
                * ``x_offset``: offset value to move different groups along the x axis for better visualization.
                  Default: 0.05
                * ``xlabel``: label of x axis. Default: "Subphases" (if subphases are present)
                  or "Phases" (if only phases are present)
                * ``ylabel``: label of y axis. Default: "$\Delta$HR [%]"
                * ``ylims``: list to manually specify y axis limits, float to specify y axis margin
                  (see :meth:`~matplotlib.Axes.margin()` for further information), or ``None`` to automatically infer
                  y axis limits
                * ``marker``: string or list of strings to specify marker style.
                  If ``marker`` is a string, then marker of each line will have the same style.
                  If ``marker`` is a list, then marker of each line will have a different style.
                * ``linestyle``: string or list of strings to specify line style.
                  If ``linestyle`` is a string, then each line will have the same style.
                  If ``linestyle`` is a list, then each line will have a different style.


        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        ax : :class:`matplotlib.axes.Axes`
            axes object


        See Also
        --------
        `biopsykit.plotting.lineplot`
            Plot data as lineplot with mean and standard error

        """
        data = mean_se_per_phase(self.hr_results[result_id])
        kwargs.update(self.hr_mean_plot_params)
        return plot.hr_mean_plot(data=data, **kwargs)