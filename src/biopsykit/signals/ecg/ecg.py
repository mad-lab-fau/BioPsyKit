"""Module for processing ECG data."""
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.stats import iqr
from tqdm.auto import tqdm

from biopsykit.signals._base import _BaseProcessor
from biopsykit.utils.array_handling import find_extrema_in_radius, remove_outlier_and_interpolate, sanitize_input_1d
from biopsykit.utils.datatype_helper import (
    EcgRawDataFrame,
    EcgResultDataFrame,
    HeartRatePhaseDict,
    RPeakDataFrame,
    _EcgResultDataFrame,
    _RPeakDataFrame,
    is_ecg_raw_dataframe,
    is_ecg_result_dataframe,
    is_r_peak_dataframe,
)

__all__ = ["EcgProcessor"]


def _hrv_process_get_hrv_types(hrv_types: Union[str, Sequence[str]]) -> Sequence[str]:
    if hrv_types is None:
        # TODO: change default parameter to 'all'
        hrv_types = ["hrv_time", "hrv_nonlinear"]
    if hrv_types == "all":
        hrv_types = list(_hrv_methods.keys())
    if isinstance(hrv_types, str):
        hrv_types = [hrv_types]

    # check whether all supplied hrv_types are valid
    for hrv_type in hrv_types:
        if hrv_type not in _hrv_methods:
            raise ValueError(
                "Invalid 'hrv_types'. Must be in {}, but got {}".format(list(_hrv_methods.keys()), hrv_type)
            )
    return hrv_types


class EcgProcessor(_BaseProcessor):
    """Class for processing ECG data."""

    def __init__(
        self,
        data: Union[EcgRawDataFrame, Dict[str, EcgRawDataFrame]],
        sampling_rate: Optional[float] = None,
        time_intervals: Optional[Union[pd.Series, Dict[str, Sequence[str]]]] = None,
        include_start: Optional[bool] = False,
    ):
        """Initialize a new ``EcgProcessor`` instance.

        To use this class simply pass data in form of a :class:`~pandas.DataFrame` (or a dict of such).
        If the data was recorded during a study that consists of multiple phases, the ECG data can be split into single
        phases by passing time information via the ``time_intervals`` parameter.

        Each instance of ``EcgProcessor`` the following attributes:

        * ``data``: dict with raw ECG data, split into the specified phases. If data was not split the
          dictionary only has one entry, accessible by the key ``Data``
        * ``ecg_result`` : dict with ECG processing results from ``data``. Each dataframe in the dict has the
          following columns:

          * ``ECG_Raw``: Raw ECG signal
          * ``ECG_Clean``: Cleaned (filtered) ECG signal
          * ``ECG_Quality``: Quality indicator in the range of [0,1] for ECG signal quality
          * ``ECG_R_Peaks``: 1.0 where R peak was detected in the ECG signal, 0.0 else
          * ``R_Peak_Outlier``: 1.0 when a detected R peak was classified as outlier, 0.0 else
          * ``Heart_Rate``: Computed Heart rate interpolated to the length of the raw ECG signal

        * ``heart_rate``: dict with heart rate derived from ``data``. Each dataframe in the dict has the
          following columns:

          * ``Heart_Rate``: Computed heart rate for each detected R peak

        * ``rpeaks``: dict with R peak location indices derived from ``data``. Each dataframe in the dict has the
          following columns:

          * ``R_Peak_Quality``: Quality indicator in the range of [0,1] for quality of the original ECG signal
          * ``R_Peak_Idx``: Index of detected R peak in the raw ECG signal
          * ``RR_Interval``: Interval between the current and the successive R peak in seconds
          * ``R_Peak_Outlier``: 1.0 when a detected R peak was classified as outlier, 0.0 else


        You can either pass a data dictionary 'data_dict' containing ECG data or dataframe containing ECG data.
        For the latter, you can additionally supply time information via ``time_intervals`` parameter to automatically
        split the data into single phases.


        Parameters
        ----------
        data : :class:`~biopsykit.utils.datatype_helper.EcgRawDataFrame` or dict
            dataframe (or dict of such) with ECG data
        sampling_rate : float, optional
            sampling rate of recorded data in Hz
        time_intervals : dict or :class:`~pandas.Series`, optional
            time intervals indicating how ``data`` should be split.
            Can either be a :class:`~pandas.Series` with the `start` times of the single phases
            (the phase names are then derived from the index) or a dictionary with tuples indicating
            `start` and `end` times of phases (the phase names are then derived from the dict keys).
            Default: ``None`` (data is not split further)
        include_start : bool, optional
            ``True`` to include the data from the beginning of the recording to the first time interval as the
            first phase (then named ``Start``), ``False`` otherwise. Default: ``False``


        Examples
        --------
        >>> # Example using NilsPod Dataset
        >>> from biopsykit.io.nilspod import load_dataset_nilspod
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>>
        >>> # path to file
        >>> file_path = "./NilsPod_TestData.bin"
        >>> # time zone of the recording (optional)
        >>> timezone = "Europe/Berlin"
        >>>
        >>> # define time intervals of the different recording phases
        >>> time_intervals = {"Part1": ("09:00", "09:30"), "Part2": ("09:30", "09:45"), "Part3": ("09:45", "10:00")}
        >>>
        >>> # load data from binary file
        >>> data, sampling_rate = load_dataset_nilspod(file_path=file_path, datastreams=['ecg'], timezone=timezone)
        >>> ecg_processor = EcgProcessor(data=data, sampling_rate=sampling_rate, time_intervals=time_intervals)

        """
        if sampling_rate is None:
            sampling_rate = 256.0
        super().__init__(
            data=data, sampling_rate=sampling_rate, time_intervals=time_intervals, include_start=include_start
        )
        for df in self.data.values():
            # make sure all data has the correct format
            is_ecg_raw_dataframe(df)

        self.ecg_result: Dict[str, EcgResultDataFrame] = {}
        """Dictionary with ECG processing result dataframes, split into different phases.

        Each dataframe is expected to be a ``EcgResultDataFrame``.

        See Also
        --------
        :obj:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`
            dataframe format

        """

        self.heart_rate: HeartRatePhaseDict = {}
        """Dictionary with time-series heart rate data, split into different phases.

        See Also
        --------
        :obj:`~biopsykit.utils.datatype_helper.HeartRatePhaseDict`
            dictionary format

        """

        self.rpeaks: Dict[str, RPeakDataFrame] = {}
        """Dictionary with R peak location indices, split into different phases.

        See Also
        --------
        :obj:`~biopsykit.utils.datatype_helper.RPeakDataFrame`
            dictionary format

        """

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        """Return ECG signal after filtering, split into different phases.

        Returns
        -------
        dict
            dictionary with filtered ECG signal per phase

        """
        return {k: pd.DataFrame(v["ECG_Clean"]) for k, v in self.ecg_result.items()}

    @property
    def hr_result(self) -> HeartRatePhaseDict:
        """Return heart rate result from ECG processing, split into different phases.

        Returns
        -------
        dict
            dictionary with time-series heart rate per phase

        """
        return self.heart_rate

    def ecg_process(
        self,
        outlier_correction: Optional[Union[str, Sequence[str]]] = "all",
        outlier_params: Optional[Dict[str, Union[float, Sequence[float]]]] = None,
        title: Optional[str] = None,
        method: Optional[str] = None,
    ) -> None:
        """Process ECG signal.

        The ECG processing pipeline consists of the following steps:

        * ``Filtering``: Uses :func:`~neurokit2.ecg.ecg_clean` to clean the ECG signal and prepare it
          for R peak detection
        * ``R-peak detection``: Uses :func:`~neurokit2.ecg.ecg_peaks` to find and extract R peaks.
        * ``Outlier correction`` (optional): Uses :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_outlier`
          to check detected R peaks for outlier and impute removed outlier by linear interpolation.


        Parameters
        ----------
        outlier_correction : list, ``all`` or ``None``, optional
            List containing outlier correction methods to be applied. Alternatively, pass ``all`` to apply all
            available outlier correction methods, or ``None`` to not apply any outlier correction.
            See :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections` to get a list of possible
            outlier correction methods. Default: ``all``
        outlier_params : dict
            Dictionary of outlier correction parameters or ``None`` for default parameters.
            See :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default` for the default parameters.
            Default: ``None``
        title : str, optional
            title of ECG processing progress bar in Jupyter Notebooks or ``None`` to leave empty. Default: ``None``
        method : {'neurokit', 'hamilton', 'pantompkins', 'elgendi', ... }, optional
            method used to clean ECG signal and perform R-peak detection as defined by the ``neurokit`` library
            (see :func:`~neurokit2.ecg.ecg_clean` and :func:`~neurokit2.ecg.ecg_peaks`) or
            ``None`` to use default method (``neurokit``).


        See Also
        --------
        :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_outlier`
            function to perform R peak outlier correction
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections`
            list of all available outlier correction methods
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default`
            dictionary with default parameters for outlier correction
        :func:`~neurokit2.ecg.ecg_clean`
            neurokit method to clean ECG signal
        :func:`~neurokit2.ecg.ecg_peaks`
            neurokit method for R-peak detection


        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # use default outlier correction pipeline
        >>> ecg_processor.ecg_process()

        >>> # don't apply any outlier correction
        >>> ecg_processor.ecg_process(outlier_correction=None)

        >>> # use custom outlier correction pipeline: only physiological and statistical outlier with custom thresholds
        >>> methods = ["physiological", "statistical"]
        >>> params = {
        >>>    'physiological': (50, 150),
        >>>    'statistical': 2.576
        >>>}
        >>> ecg_processor.ecg_process(outlier_correction=methods, outlier_params=params)

        >>> # Print available results from ECG processing
        >>> print(ecg_processor.ecg_result)
        >>> print(ecg_processor.rpeaks)
        >>> print(ecg_processor.heart_rate)

        """
        if method is None:
            method = "neurokit"

        for name, df in tqdm(self.data.items(), desc=title):
            ecg_result, rpeaks = self._ecg_process(df, method=method)
            ecg_result, rpeaks = self.correct_outlier(
                ecg_signal=ecg_result,
                rpeaks=rpeaks,
                outlier_correction=outlier_correction,
                outlier_params=outlier_params,
                sampling_rate=self.sampling_rate,
            )
            heart_rate = pd.DataFrame({"Heart_Rate": 60 / rpeaks["RR_Interval"]})
            rpeaks.loc[:, "Heart_Rate"] = heart_rate
            heart_rate_interpolated = nk.signal_interpolate(
                x_values=np.squeeze(rpeaks["R_Peak_Idx"].values),
                y_values=np.squeeze(heart_rate["Heart_Rate"].values),
                x_new=np.arange(0, len(ecg_result["ECG_Clean"])),
            )
            ecg_result["Heart_Rate"] = heart_rate_interpolated
            self.ecg_result[name] = ecg_result
            self.heart_rate[name] = heart_rate
            self.rpeaks[name] = rpeaks

    def _ecg_process(
        self, data: EcgRawDataFrame, method: Optional[str] = None
    ) -> Tuple[EcgResultDataFrame, RPeakDataFrame]:
        """Private method for ECG processing.

        Parameters
        ----------
        data : pd.DataFrame
            ECG data as pandas dataframe. Needs to have one column named 'ecg'
        method : {'neurokit', 'hamilton', 'pantompkins', 'elgendi', ... }, optional
            method for cleaning the ECG signal and R peak detection as defined by 'neurokit'.
            Default: ``None`` (corresponds to ``neurokit``)

        Returns
        -------
        ecg_result : :class:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`, optional
            Dataframe with processed ECG signal. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`
        rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
            Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`

        """
        # get numpy
        ecg_signal = data["ecg"].values
        # clean (i.e. filter) the ECG signal using the specified method
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=int(self.sampling_rate), method=method)

        # find peaks using the specified method
        # instant_peaks: array indicating where detected R peaks are in the raw ECG signal
        # rpeak_index array containing the indices of detected R peaks
        instant_peaks, rpeak_idx = nk.ecg_peaks(ecg_cleaned, sampling_rate=int(self.sampling_rate), method=method)
        rpeak_idx = rpeak_idx["ECG_R_Peaks"]
        instant_peaks = np.squeeze(instant_peaks.values)

        # compute quality indicator
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeak_idx, sampling_rate=int(self.sampling_rate))

        # construct new dataframe
        ecg_result = pd.DataFrame(
            {
                "ECG_Raw": ecg_signal,
                "ECG_Clean": ecg_cleaned,
                "ECG_Quality": quality,
                "ECG_R_Peaks": instant_peaks,
                "R_Peak_Outlier": np.zeros(len(data)),
            },
            index=data.index,
        )

        # copy new dataframe consisting of R peaks indices (and their respective quality indicator)
        rpeaks = ecg_result.loc[ecg_result["ECG_R_Peaks"] == 1.0, ["ECG_Quality"]]
        rpeaks.rename(columns={"ECG_Quality": "R_Peak_Quality"}, inplace=True)
        rpeaks.loc[:, "R_Peak_Idx"] = rpeak_idx
        # compute RR interval
        rpeaks["RR_Interval"] = np.ediff1d(rpeaks["R_Peak_Idx"], to_end=0) / self.sampling_rate
        # ensure equal length by filling the last value with the average RR interval
        rpeaks.loc[rpeaks.index[-1], "RR_Interval"] = rpeaks["RR_Interval"].mean()

        is_ecg_result_dataframe(ecg_result)
        is_r_peak_dataframe(rpeaks)

        return _EcgResultDataFrame(ecg_result), _RPeakDataFrame(rpeaks)

    @classmethod
    def outlier_corrections(cls) -> Sequence[str]:
        """Return all possible outlier correction methods.

        Currently available outlier correction methods are:

        * ``correlation``: Computes cross-correlation coefficient between every single beat and the average of
          all detected beats. Marks beats as outlier if cross-correlation coefficient is below a certain threshold.
        * ``quality``: Uses the ``ECG_Quality`` indicator from neurokit to assess signal quality. Marks beats as
          outlier if the quality indicator is below a certain threshold.
        * ``artifact``: Artifact detection based on `Berntson et al. (1990)`.
        * ``physiological``: Physiological outlier removal. Marks beats as outlier if their heart rate is above
          or below a threshold that is very unlikely to be achieved physiologically.
        * ``statistical_rr``: Statistical outlier removal based on RR intervals. Marks beats as outlier if the RR
          intervals are within the xx% highest or lowest values. Values are removed based on the z-score;
          e.g. 1.96 => 5% (2.5% highest, 2.5% lowest values); 2.576 => 1% (0.5% highest, 0.5% lowest values)
        * ``statistical_rr_diff``: Statistical outlier removal based on successive differences of RR intervals.
          Marks beats as outlier if the difference of successive RR intervals are within the xx% highest or
          lowest heart rates. Values are removed based on the z-score;
          e.g. 1.96 => 5% (2.5% highest, 2.5% lowest values); 2.576 => 1% (0.5% highest, 0.5% lowest values).

        See Also
        --------
        :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_outlier`
            function to perform R peak outlier correction
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default`
            dictionary with default parameters for outlier correction

        Returns
        -------
        list
            keys of all possible outlier correction methods

        References
        ----------
        Berntson, G. G., Quigley, K. S., Jang, J. F., & Boysen, S. T. (1990). An Approach to Artifact Identification:
        Application to Heart Period Data. *Psychophysiology*, 27(5), 586–598.
        https://doi.org/10.1111/j.1469-8986.1990.tb01982.x

        """
        return list(_outlier_correction_methods.keys())

    @classmethod
    def outlier_params_default(cls) -> Dict[str, Union[float, Sequence[float]]]:
        """Return default parameter for all outlier correction methods.

        .. note::
            The outlier correction method ``artifact`` has no threshold, but ``0.0`` is the default parameter in order
            to provide a homogenous interface

        See Also
        --------
        :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_outlier`
            function to perform R peak outlier correction
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections`
            list with available outlier correction methods

        Returns
        -------
        dict
            default parameters for outlier correction methods

        """
        return _outlier_correction_params_default

    @classmethod
    def correct_outlier(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[EcgResultDataFrame] = None,
        rpeaks: Optional[RPeakDataFrame] = None,
        outlier_correction: Optional[Union[str, None, Sequence[str]]] = "all",
        outlier_params: Optional[Dict[str, Union[float, Sequence[float]]]] = None,
        imputation_type: Optional[str] = None,
        sampling_rate: Optional[float] = 256.0,
    ) -> Tuple[EcgResultDataFrame, RPeakDataFrame]:
        """Perform outlier correction on the detected R peaks.

        Different methods for outlier detection are available (see
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections()` to get a list of possible outlier
        correction methods). All outlier methods work independently on the detected R peaks, the results will be
        combined by a logical 'or'.

        RR intervals classified as outliers will be removed and imputed either using linear interpolation
        (setting ``imputation_type`` to ``linear``) or by replacing it with the average value of the 10 preceding
        and 10 succeding RR intervals (setting ``imputation_type`` to ``moving_average``).

        To use this function, either simply pass an :class:`~biopsykit.signals.ecg.EcgProcessor` object  together with
        a ``key`` indicating which phase needs to be processed should be processed or the two dataframes ``ecg_signal``
        and ``rpeaks`` resulting from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.

        Parameters
        ----------
        ecg_processor : :class:`~biopsykit.signals.ecg.EcgProcessor`, optional
            ``EcgProcessor`` object. If this argument is supplied, the ``key`` argument needs to be supplied as well
        key : str, optional
            Dictionary key of the phase to process. Needed when ``ecg_processor`` is passed as argument
        ecg_signal : :class:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`, optional
            Dataframe with processed ECG signal. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`
        rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
            Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`
        outlier_correction : list, optional
            List containing the outlier correction methods to be applied.
            Pass ``None`` to not apply any outlier correction, ``all`` to apply all available outlier correction
            methods. See :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections` to get a list of possible
            outlier correction methods.
            Default: ``all``
        outlier_params: dict, optional
            Dict of parameters to be passed to the outlier correction methods or ``None``
            to use default parameters (see :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default`
            for more information).
            Default: ``None``
        imputation_type: str, optional
            Method for outlier imputation: ``linear`` for linear interpolation between the RR intervals before and
            after R peak outlier, or ``moving_average`` for average value of the
            10 preceding and 10 succeding RR intervals.
            Default: ``None`` (corresponds to ``moving_average``)
        sampling_rate : float, optional
            Sampling rate of recorded data in Hz. Not needed if ``ecg_processor`` is supplied as parameter.
            Default: 256

        Returns
        -------
        ecg_signal : :class:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`
            processed ECG signal in standardized format
        rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`
            extracted R peaks in standardized format


        See Also
        --------
        :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process`
            function for ECG signal processing
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_corrections`
            list of all available outlier correction methods
        :meth:`~biopsykit.signals.ecg.EcgProcessor.outlier_params_default`
            dictionary with default parameters for outlier correction


        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # Option 1: Use default outlier correction pipeline
        >>> ecg_signal, rpeaks = ecg_processor.correct_outlier(ecg_processor, key="Data")
        >>> print(ecg_signal)
        >>> print(rpeaks)
        >>> # Option 2: Use custom outlier correction pipeline: only physiological and statistical
        >>> # RR interval outlier with custom thresholds
        >>> methods = ["physiological", "statistical_rr"]
        >>> params = {
        >>>    'physiological': (50, 150),
        >>>    'statistical_rr': 2.576
        >>>}
        >>> ecg_signal, rpeaks = ecg_processor.correct_outlier(
        >>>                             ecg_processor, key="Data",
        >>>                             outlier_correction=methods,
        >>>                             outlier_params=params
        >>>                         )
        >>> print(ecg_signal)
        >>> print(rpeaks)

        """
        _assert_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor is not None:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        _check_dataframe_format(ecg_signal, rpeaks)

        outlier_correction, outlier_params, outlier_funcs = _get_outlier_params(outlier_correction, outlier_params)

        imputation_types = ["linear", "moving_average"]
        imputation_type = _get_imputation_type(imputation_type, imputation_types)

        # copy dataframe to mark removed beats later
        rpeaks_copy = rpeaks.copy()
        # get the last sample because it will get lost when computing the RR interval
        last_sample = rpeaks.iloc[-1]

        # initialize bool mask to mask outlier and add outlier column to rpeaks dataframe
        bool_mask = np.full(rpeaks.shape[0], False)
        rpeaks["R_Peak_Outlier"] = 0.0

        # TODO add source of different outlier methods for plotting/statistics
        for k in outlier_funcs:
            kwargs = {"ecg_signal": ecg_signal, "sampling_rate": sampling_rate}
            bool_mask = outlier_funcs[k](rpeaks, bool_mask, outlier_params[k], **kwargs)

        # mark all removed beats as outlier in the ECG dataframe
        rpeaks[bool_mask] = None
        removed_beats = rpeaks_copy["R_Peak_Idx"][rpeaks["R_Peak_Idx"].isna()]
        # mark all outlier with 1.0 in the column R_Peak_Outlier column
        rpeaks = rpeaks.fillna({"R_Peak_Outlier": 1.0})
        if ecg_signal is not None:
            # also mark outlier in the ECG signal dataframe
            ecg_signal.loc[removed_beats.index, "R_Peak_Outlier"] = 1.0

        # replace the last beat by average
        if "R_Peak_Quality" in rpeaks.columns:
            rpeaks.loc[last_sample.name] = [
                rpeaks["R_Peak_Quality"].mean(),
                last_sample["R_Peak_Idx"],
                rpeaks["RR_Interval"].mean(),
                0.0,
            ]

        # if imputation type is moving average: replace RR intervals by moving average before interpolating
        # the other columns
        if imputation_type == "moving_average":
            rpeaks["RR_Interval"] = rpeaks["RR_Interval"].fillna(
                rpeaks["RR_Interval"].rolling(21, center=True, min_periods=0).mean()
            )

        # interpolate all columns (except RR_Interval if imputation type is moving average)
        rpeaks = rpeaks.interpolate(method="linear", limit_direction="both")
        # drop duplicate R peaks (can happen during outlier correction at edge cases)
        rpeaks = rpeaks.drop_duplicates(subset="R_Peak_Idx")

        _check_dataframe_format(ecg_signal, rpeaks)

        return _EcgResultDataFrame(ecg_signal), _RPeakDataFrame(rpeaks)

    @classmethod
    def correct_rpeaks(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        rpeaks: Optional[RPeakDataFrame] = None,
        sampling_rate: Optional[float] = 256.0,
    ) -> pd.DataFrame:
        """Perform R peak correction algorithms to get less noisy HRV parameters.

        R peak correction comes from ``neurokit`` and is based on an algorithm by `Lipponen et al. (2019)`.

        To use this function, either simply pass an :class:`~biopsykit.signals.ecg.EcgProcessor` object together with
        a ``key`` indicating which phase needs to be processed should be processed or the dataframe ``rpeaks``
        which is a result from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.

        .. warning ::
            This algorithm might *add* additional R peaks or *remove* certain ones, so results of this function
            might **not** match with the R peaks of :meth:`~biopsykit.signals.ecg.EcgProcessor.rpeaks`. Thus, R peaks
            resulting from this function might not be used in combination with
            :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg` since R peak indices won't match.

        .. note ::
            In `BioPsyKit` this function is **not** applied to the detected R peaks during ECG signal processing but
            **only** used right before passing R peaks to :meth:`~biopsykit.signals.ecg.EcgProcessor.hrv_process()`.


        Parameters
        ----------
        ecg_processor : :class:`~biopsykit.signals.ecg.EcgProcessor`, optional
            ``EcgProcessor`` object. If this argument is supplied, the ``key`` argument needs to be supplied as well.
        key : str, optional
            Dictionary key of the phase to process. Needed when ``ecg_processor`` is passed as argument.
        rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
            Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.
        sampling_rate : float, optional
            Sampling rate of recorded data in Hz. Not needed if ``ecg_processor`` is supplied as parameter.
            Default: 256


        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe containing corrected R peak indices


        References
        ----------
        Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact
        correction using novel beat classification. *Journal of Medical Engineering and Technology*,
        43(3), 173–181. https://doi.org/10.1080/03091902.2019.1640306


        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ep = EcgProcessor(...)
        >>> # correct R peak locations
        >>> rpeaks_corrected = ep.correct_rpeaks(ecg_processor, key="Data")

        """
        _assert_rpeaks_input(ecg_processor, key, rpeaks)
        if ecg_processor:
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        is_r_peak_dataframe(rpeaks)

        # fill missing RR intervals with interpolated R Peak Locations
        rpeaks_corrected = (rpeaks["RR_Interval"].cumsum() * sampling_rate).astype(int)
        rpeaks_corrected = np.append(
            rpeaks["R_Peak_Idx"].iloc[0],
            rpeaks_corrected.iloc[:-1] + rpeaks["R_Peak_Idx"].iloc[0],
        )
        _, rpeaks_corrected = nk.signal_fixpeaks(rpeaks_corrected, int(sampling_rate), iterative=False)
        rpeaks_corrected = rpeaks_corrected.astype(int)
        rpeaks_result = pd.DataFrame(rpeaks_corrected, columns=["R_Peak_Idx"])
        return rpeaks_result

    @classmethod
    def hrv_process(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        rpeaks: Optional[RPeakDataFrame] = None,
        hrv_types: Optional[Sequence[str]] = None,
        correct_rpeaks: Optional[bool] = True,
        index: Optional[str] = None,
        index_name: Optional[str] = None,
        sampling_rate: Optional[float] = 256.0,
    ) -> pd.DataFrame:
        """Compute HRV parameters on the given data.

        By default, it applies R peak correction (see :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_rpeaks`)
        before computing HRV parameters.

        To use this function, either simply pass an :class:`~biopsykit.signals.ecg.EcgProcessor` object together with
        a ``key`` indicating which phase needs to be processed should be processed or the dataframe ``rpeaks``
        which is a result from  :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.

        Parameters
        ----------
        ecg_processor : :class:`~biopsykit.signals.ecg.EcgProcessor`, optional
            ``EcgProcessor`` object. If this argument is supplied, the ``key`` argument needs to be supplied as well.
        key : str, optional
            Dictionary key of the phase to process. Needed when ``ecg_processor`` is passed as argument.
        rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
            Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.
        hrv_types: str (or list of such), optional
            list of HRV types to be computed. Must be a subset of ["hrv_time", "hrv_nonlinear", "hrv_frequency"]
            or "all" to compute all types of HRV. Refer to :func:`neurokit2.hrv.hrv` for further information on
            the available HRV parameters. Default: ``None`` (equals to ["hrv_time", "hrv_nonlinear"])
        correct_rpeaks : bool, optional
            ``True`` to apply R peak correction (using :meth:`~biopsykit.signals.ecg.EcgProcessor.correct_rpeaks()`)
            before computing HRV parameters, ``False`` otherwise. Default: ``True``
        index: str, optional
            Index of the computed HRV parameters. Used to concatenate HRV processing results from multiple phases into
            one joint dataframe later on. Default: ``None``
        index_name : str, optional
            Index name of the output dataframe. Only used if ``index`` is also supplied. Default: ``None``
        sampling_rate : float, optional
            Sampling rate of recorded data in Hz. Not needed if ``ecg_processor`` is supplied as parameter.
            Default: 256


        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with computed HRV parameters


        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # HRV processing using default parameters (time and nonlinear), including R peak correction
        >>> hrv_output = ecg_processor.hrv_process(ecg_processor, key="Data")

        >>> # HRV processing using using all types, and without R peak correction
        >>> hrv_output = ecg_processor.hrv_process(ecg_processor, key="Data", hrv_types='all', correct_rpeaks=False)

        """
        _assert_rpeaks_input(ecg_processor, key, rpeaks)
        if ecg_processor:
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        is_r_peak_dataframe(rpeaks)

        if correct_rpeaks:
            rpeaks = cls.correct_rpeaks(rpeaks=rpeaks, sampling_rate=sampling_rate)

        hrv_types = _hrv_process_get_hrv_types(hrv_types)
        hrv_methods = {key: _hrv_methods[key] for key in hrv_types}

        # compute all HRV parameters
        list_hrv: List[pd.DataFrame] = [
            hrv_methods[key](rpeaks["R_Peak_Idx"], sampling_rate=sampling_rate) for key in hrv_methods
        ]
        # concat dataframe list
        hrv = pd.concat(list_hrv, axis=1)

        # TODO: use 'key' if index is None?
        if index is not None:
            # set index to dataframe if supplied
            hrv.index = [index]
            hrv.index.name = index_name
        return hrv

    def hrv_batch_process(self, hrv_types: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Compute HRV parameters over all phases.

        This function computes HRV parameters over all phases using
        :meth:`~biopsykit.signals.ecg.EcgProcessor.hrv_process()`.

        Parameters
        ----------
        hrv_types: str (or list of such), optional
            list of HRV types to be computed. Must be a subset of ['hrv_time', 'hrv_nonlinear', 'hrv_frequency']
            or 'all' to compute all types of HRV. Refer to :func:`neurokit2.hrv.hrv` for further information on
            the available HRV parameters. Default: ``None`` (equals to ['hrv_time', 'hrv_nonlinear'])

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with HRV parameters over all phases

        """
        return pd.concat([self.hrv_process(self, key=key, index=key, hrv_types=hrv_types) for key in self.phases])

    @classmethod
    def ecg_estimate_rsp(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[EcgResultDataFrame] = None,
        rpeaks: Optional[RPeakDataFrame] = None,
        edr_type: Optional[str] = None,
        sampling_rate: Optional[float] = 256,
    ) -> pd.DataFrame:
        """Estimate respiration signal from ECG (ECG-derived respiration, EDR).

        To use this function, either simply pass an :class:`~biopsykit.signals.ecg.EcgProcessor` object  together with
        a ``key`` indicating which phase needs to be processed should be processed or the two dataframes ``ecg_signal``
        and ``rpeaks`` resulting from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.


        Parameters
        ----------
        ecg_processor : :class:`~biopsykit.signals.ecg.EcgProcessor`, optional
            ``EcgProcessor`` object. If this argument is supplied, the ``key`` argument needs to be supplied as well.
        key : str, optional
            Dictionary key of the phase to process. Needed when ``ecg_processor`` is passed as argument.
        ecg_signal : :class:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`, optional
            Dataframe with processed ECG signal. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.
        rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
            Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`
        edr_type : {'peak_trough_mean', 'peak_trough_diff', 'peak_peak_interval'}, optional
            Method to use for estimating EDR. Must be one of 'peak_trough_mean', 'peak_trough_diff',
            or 'peak_peak_interval'. Default: 'peak_trough_mean'
        sampling_rate : float, optional
            Sampling rate of recorded data in Hz. Not needed if ``ecg_processor`` is supplied as parameter.
            Default: 256

        Returns
        -------
        :class:`~pandas.DataFrame`
            dataframe with estimated respiration signal

        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # Extract respiration signal estimated from ECG using the 'peak_trough_diff' method
        >>> rsp_signal = ecg_processor.ecg_estimate_rsp(ecg_processor, key="Data", edr_type='peak_trough_diff')

        """
        _assert_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        is_ecg_result_dataframe(ecg_signal)
        is_r_peak_dataframe(rpeaks)

        if edr_type is None:
            edr_type = "peak_trough_mean"
        if edr_type not in _edr_methods:
            raise ValueError("`edr_type` must be one of {}, not {}".format(list(_edr_methods.keys()), edr_type))
        edr_func = _edr_methods[edr_type]

        # ensure numpy
        peaks = np.squeeze(rpeaks["R_Peak_Idx"].values)

        # find troughs (minimum 0.1s before R peak)
        troughs = find_extrema_in_radius(ecg_signal["ECG_Clean"], peaks, radius=(int(0.1 * sampling_rate), 0))
        # R peak outlier should not be included into EDR estimation
        outlier_mask = rpeaks["R_Peak_Outlier"] == 1
        # estimate raw EDR signal
        edr_signal_raw = edr_func(ecg_signal["ECG_Clean"], peaks, troughs)
        # remove R peak outlier, impute missing data, and interpolate signal to length of raw ECG signal
        edr_signal = remove_outlier_and_interpolate(edr_signal_raw, outlier_mask, peaks, len(ecg_signal))
        # Preprocessing: 10-th order Butterworth bandpass filter (0.1-0.5 Hz)
        edr_signal = nk.signal_filter(edr_signal, sampling_rate=int(sampling_rate), lowcut=0.1, highcut=0.5, order=10)

        return pd.DataFrame(edr_signal, index=ecg_signal.index, columns=["ECG_Resp"])

    @classmethod
    def rsa_process(
        cls,
        ecg_signal: EcgResultDataFrame,
        rsp_signal: pd.DataFrame,
        sampling_rate: Optional[float] = 256,
    ) -> Dict[str, float]:
        """Compute respiratory sinus arrhythmia (RSA) based on ECG and respiration signal.

        RSA is computed both via Peak-to-Trough (P2T) Porges-Bohrer method using :func:`~neurokit2.hrv.hrv_rsa`.


        Parameters
        ----------
        ecg_signal : :class:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`, optional
            Dataframe with processed ECG signal. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`.
        rsp_signal : pd.DataFrame
            Dataframe with 1-D raw respiration signal. Can be a 'true' respiration signal
            (e.g. from bioimpedance or Radar) or an 'estimated' respiration signal (e.g. from ECG-derived respiration).
        sampling_rate : float, optional
            Sampling rate of recorded data in Hz.
            Default: 256

        Returns
        -------
        dict
            Dictionary containing computed RSA metrics.

        See Also
        --------
        :func:`~neurokit2.hrv.hrv_rsa`
            compute respiratory sinus arrhythmia


        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> ecg_signal = ecg_processor.ecg_result['Data']
        >>> # Extract respiration signal estimated from ECG using the 'peak_trough_diff' method
        >>> rsp_signal = ecg_processor.ecg_estimate_rsp(ecg_processor, key="Data", edr_type='peak_trough_diff')
        >>> # Compute RSA from ECG and Respiration data
        >>> rsa_output = ecg_processor.rsa_process(ecg_signal, rsp_signal)

        """
        is_ecg_result_dataframe(ecg_signal)

        # ensure numpy
        rsp_signal = sanitize_input_1d(rsp_signal)
        # Process raw respiration input
        rsp_output = nk.rsp_process(rsp_signal, int(sampling_rate))[0]
        rsp_output.index = ecg_signal.index
        # Compute RSA
        return nk.hrv_rsa(ecg_signal, rsp_output, sampling_rate=int(sampling_rate))

    # @classmethod
    # def rsp_rsa_process(
    #     cls,
    #     ecg_processor: Optional["EcgProcessor"] = None,
    #     key: Optional[str] = None,
    #     ecg_signal: Optional[pd.DataFrame] = None,
    #     rpeaks: Optional[pd.DataFrame] = None,
    #     return_mean: Optional[bool] = True,
    #     index: Optional[str] = None,
    #     index_name: Optional[str] = None,
    #     sampling_rate: Optional[int] = 256,
    # ) -> pd.DataFrame:
    #     """
    #     Computes respiration rate (RSP) and RSA metrics for the given ECG signal. Per default it estimates the
    #     respiration signal using all three available EDR methods and averages the results. Optionally, estimation
    #     results from the individual methods can be returned.s
    #
    #     To use this function, either simply pass an `EcgProcessor` object together with a `key` indicating
    #     which sub-phase should be processed or the two dataframes `ecg_signal` and `rpeaks` resulting from
    #     `EcgProcessor.ecg_process()`.
    #
    #     Parameters
    #     ----------
    #     ecg_processor : EcgProcessor, optional
    #         `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
    #     key : str, optional
    #         Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
    #     ecg_signal : pd.DataFrame, optional
    #         dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
    #     rpeaks : pd.DataFrame, optional
    #         dataframe with R peaks. Output of `EcgProcessor.ecg_process()`
    #     return_mean: bool, optional
    #         Flag indicating whether to return the averaged estimation results or individual results for each EDR
    #         estimation method.
    #         Default: ``True``
    #     index: str, optional
    #         Index value of the computed RSP and RSA features. Used to concatenate dataframes from multiple sub-phases
    #         later.
    #         Default: None
    #     index_name : str, optional
    #         Index name of the output dataframe. Only used if 'index' is also supplied
    #     sampling_rate : float
    #         Sampling rate of recorded data in Hz. Not needed if ``ecg_processor`` is supplied as parameter.
    #         Default: 256
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #         dataframe of respiration rate and RSA estimation results
    #
    #     Examples
    #     --------
    #     >>> import biopsykit as ep
    #     >>> # initialize EcgProcessor instance
    #     >>> ecg_processor = ep.EcgProcessor(...)
    #
    #     >>> # Compute respiration rate and RSA. Extract respiration signal using all available
    #     >>> # methods and average the ensemble ('return_mean' is True by default)
    #     >>> rsp_signal = ecg_processor.rsp_rsa_process(ecg_processor, key="Data")
    #
    #     >>> # Compute respiration rate and RSA. Extract respiration signal using all available
    #     >>> # methods and return the single results
    #     >>> rsp_signal = ecg_processor.rsp_rsa_process(ecg_processor, key="Data", return_mean=False)
    #     """
    #
    #     _assert_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
    #     if ecg_processor:
    #         ecg_signal = ecg_processor.ecg_result[key]
    #         rpeaks = ecg_processor.rpeaks[key]
    #         sampling_rate = ecg_processor.sampling_rate
    #
    #     # initialize dicts to store results
    #     rsp_rate = dict.fromkeys(_edr_methods.keys())
    #     rsa = dict.fromkeys(_edr_methods.keys())
    #
    #     for method in _edr_methods.keys():
    #         # estimate respiration signal, compute respiration signal and RSA using all three methods
    #         rsp_signal = cls.ecg_estimate_rsp(
    #             ecg_signal=ecg_signal,
    #             rpeaks=rpeaks,
    #             sampling_rate=sampling_rate,
    #             edr_type=method,
    #         )
    #         rsp_rate[method] = cls.rsp_compute_rate(rsp_signal, sampling_rate)
    #         rsa[method] = cls.rsa_process(ecg_signal, rsp_signal, sampling_rate)
    #
    #     if return_mean:
    #         # compute average respiration rate and RSA
    #         mean_resp_rate = np.mean(list(rsp_rate.values()))
    #         rsa = list(rsa.values())
    #         mean_rsa = {k: np.mean([t[k] for t in rsa]) for k in rsa[0]}
    #         mean_rsa["RSP_Rate"] = mean_resp_rate
    #         # dataframe reshaping
    #         if not index:
    #             index = "0"
    #             index_name = "Index"
    #         df_rsa = pd.DataFrame(mean_rsa, index=[index])
    #         df_rsa.index.name = index_name
    #         return df_rsa
    #     else:
    #         # dataframe reshaping
    #         df_rsa = pd.DataFrame(rsa).T
    #         df_rsa["RSP_Rate"] = rsp_rate.values()
    #         df_rsa.index.name = "Method"
    #         if index:
    #             return pd.concat([df_rsa], keys=[index], names=[index_name])
    #         return df_rsa


def _edr_peak_trough_mean(ecg: pd.Series, peaks: np.array, troughs: np.array) -> np.array:
    """Estimate respiration signal from ECG based on `peak-trough-mean` method.

    The `peak-trough-mean` method is based on computing the mean amplitude between R peaks (`peaks`) and
    minima before R peaks (`troughs`).

    Parameters
    ----------
    ecg : :class:`~pandas.Series`
        pandas series with ecg signal
    peaks : :class:`~numpy.array`
        array with peak indices
    troughs : :class:`~numpy.array`
        array with trough indices

    Returns
    -------
    :class:`~numpy.array`
        estimated raw respiration signal

    """
    peak_vals = np.array(ecg.iloc[peaks])
    trough_vals = np.array(ecg.iloc[troughs])
    return np.mean([peak_vals, trough_vals], axis=0)


def _edr_peak_trough_diff(ecg: pd.Series, peaks: np.array, troughs: np.array) -> np.array:
    """Estimate respiration signal from ECG based on `peak-trough-diff` method.

    The `peak-trough-diff` method is based on computing the amplitude difference between R peaks (`peaks`) and
    minima before R peaks (`troughs`).

    Parameters
    ----------
    ecg : :class:`~pandas.Series`
        pandas series with ecg signal
    peaks : :class:`~numpy.array`
        array with peak indices
    troughs : :class:`~numpy.array`
        array with trough indices

    Returns
    -------
    :class:`~numpy.array`
        estimated raw respiration signal

    """
    peak_vals = np.array(ecg.iloc[peaks])
    trough_vals = np.array(ecg.iloc[troughs])
    return peak_vals - trough_vals


def _edr_peak_peak_interval(
    ecg: pd.DataFrame, peaks: np.array, troughs: np.array  # pylint:disable=unused-argument
) -> np.array:
    """Estimate respiration signal from ECG based on `peak-peak-interval` method.

    The `peak-peak-interval` method is based on computing RR intervals.

    .. note::
        To ensure the same length for the resulting array after computing successive differences
        the first value will be replaced by the mean of all RR intervals in the array

    Parameters
    ----------
    ecg : :class:`~pandas.Series`
        pandas series with ecg signal (unused but needed for consistent method signature)
    peaks : :class:`~numpy.array`
        array with peak indices
    troughs : :class:`~numpy.array`
        array with trough indices (unused but needed for consistent method signature)

    Returns
    -------
    :class:`~numpy.array`
        estimated raw respiration signal

    """
    peak_interval = np.ediff1d(peaks, to_begin=0)
    peak_interval[0] = peak_interval.mean()
    return peak_interval


def _correct_outlier_correlation(rpeaks: pd.DataFrame, bool_mask: np.array, corr_thres: float, **kwargs) -> np.array:
    """Apply outlier correction method 'correlation'.

    This function compute the cross-correlation coefficient between every single beat and the average of all detected
    beats. It marks beats as outlier if the cross-correlation coefficient is below a certain threshold.

    Parameters
    ----------
    rpeaks : :class:`~pandas.DataFrame`
        dataframe with detected R peaks. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    bool_mask : :class:`numpy.array`
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    corr_thres : float
        threshold for cross-correlation coefficient. Beats below that threshold will be marked as outlier
    **kwargs : additional parameters required for this outlier function, such as:

        * ecg_signal :class:`~pandas.DataFrame`
          dataframe with processed ECG signal. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
        * sampling_rate : float
          sampling rate of recorded data in Hz

    Returns
    -------
    :class:`numpy.array`
        boolean array with beats marked as outlier. Logical 'or' combination of ``bool_mask`` and results from
        this algorithm

    """
    ecg_signal = kwargs.get("ecg_signal", None)
    sampling_rate = kwargs.get("sampling_rate", None)
    if any(v is None for v in [ecg_signal, sampling_rate]):
        raise ValueError(
            "Cannot apply outlier correction method 'correlation' because not all additionally required arguments "
            "were provided! Make sure you pass the following arguments: 'ecg_signal', 'sampling_rate'."
        )
    # signal outlier
    # segment individual heart beats
    heartbeats = nk.ecg_segment(ecg_signal["ECG_Clean"], rpeaks["R_Peak_Idx"], int(sampling_rate))
    heartbeats = nk.epochs_to_df(heartbeats)
    heartbeats_pivoted = heartbeats.pivot(index="Time", columns="Label", values="Signal")
    heartbeats = heartbeats.set_index("Index")
    heartbeats = heartbeats.loc[heartbeats.index.intersection(rpeaks["R_Peak_Idx"])].sort_values(by="Label")
    heartbeats = heartbeats[~heartbeats.index.duplicated()]
    heartbeats_pivoted.columns = heartbeats.index

    # compute the average over all heart beats and compute the correlation coefficient between all beats and
    # the average
    mean_beat = heartbeats_pivoted.mean(axis=1)
    heartbeats_pivoted["mean"] = mean_beat
    corr_coeff = heartbeats_pivoted.corr()["mean"].abs().sort_values(ascending=True)
    corr_coeff = corr_coeff.drop("mean")
    # compute RR intervals (in seconds) from R Peak Locations
    rpeaks["RR_Interval"] = np.ediff1d(rpeaks["R_Peak_Idx"], to_end=0) / sampling_rate

    # signal outlier: drop all beats that are below a correlation coefficient threshold
    return np.logical_or(bool_mask, rpeaks["R_Peak_Idx"].isin(corr_coeff[corr_coeff < corr_thres].index))


def _correct_outlier_quality(
    rpeaks: pd.DataFrame, bool_mask: np.array, quality_thres: float, **kwargs  # pylint:disable=unused-argument
) -> np.array:
    """Apply outlier correction method 'quality'.

    This function uses the ``ECG_Quality`` indicator from ``neurokit`` to assess signal quality. It marks beats as
    outlier if the quality indicator is below a certain threshold.


    Parameters
    ----------
    rpeaks : :class:`~pandas.DataFrame`
        dataframe with detected R peaks. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    bool_mask : :class:`numpy.array`
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    quality_thres : float
        threshold for signal quality indicator. Beats below that threshold will be marked as outlier


    Returns
    -------
    :class:`numpy.array`
        boolean array with beats marked as outlier. Logical 'or' combination of ``bool_mask`` and results from
        this algorithm

    """
    # signal outlier: drop all beats that are below a signal quality threshold
    return np.logical_or(bool_mask, rpeaks["R_Peak_Quality"] < quality_thres)


def _correct_outlier_statistical_rr(
    rpeaks: pd.DataFrame,
    bool_mask: np.array,
    stat_thres: float,
    **kwargs,  # pylint:disable=unused-argument
) -> np.array:
    """Apply outlier correction method 'statistical_rr'.

    This function marks beats as outlier if they are within the xx % highest or lowest RR intervals, i.e., if
    their z-score is above a threshold, e.g., ``1.96`` => 5% (2.5% highest, 2.5% lowest values);
    ``2.576`` => 1% (0.5% highest, 0.5% lowest values)


    Parameters
    ----------
    rpeaks : :class:`~pandas.DataFrame`
        dataframe with detected R peaks. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    bool_mask : :class:`numpy.array`
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    stat_thres : float
        threshold for z-score. Beats above that threshold will be marked as outlier


    Returns
    -------
    :class:`numpy.array`
        boolean array with beats marked as outlier. Logical 'or' combination of ``bool_mask`` and results from
        this algorithm

    """
    # statistical outlier: remove the x% highest and lowest RR intervals
    # (1.96 std = 5% outlier, 2.576 std = 1% outlier)
    rri = rpeaks["RR_Interval"]
    z_score = (rri - np.nanmean(rri)) / np.nanstd(rri, ddof=1)

    return np.logical_or(bool_mask, np.abs(z_score) > stat_thres)


def _correct_outlier_statistical_rr_diff(
    rpeaks: pd.DataFrame, bool_mask: np.array, stat_thres: float, **kwargs  # pylint:disable=unused-argument
) -> np.array:
    """Apply outlier correction method 'statistical_rr_diff'.

    This function marks beats as outlier if their successive differences of RR intervals are within the xx % highest or
    lowest values, i.e. if their z-score is above a threshold, e.g. ``1.96`` => 5% (2.5% highest, 2.5% lowest values);
    ``2.576`` => 1% (0.5% highest, 0.5% lowest values).


    Parameters
    ----------
    rpeaks : :class:`~pandas.DataFrame`
        dataframe with detected R peaks. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    bool_mask : :class:`numpy.array`
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    stat_thres : float
        threshold for z-score. Beats above that threshold will be marked as outlier


    Returns
    -------
    :class:`numpy.array`
        boolean array with beats marked as outlier. Logical 'or' combination of ``bool_mask`` and results from
        this algorithm

    """
    # statistical outlier: remove the x% highest and lowest successive differences of RR intervals
    # (1.96 std = 5% outlier, 2.576 std = 1% outlier)
    diff_rri = np.ediff1d(rpeaks["RR_Interval"], to_end=0)
    z_score = (diff_rri - np.nanmean(diff_rri)) / np.nanstd(diff_rri, ddof=1)

    return np.logical_or(bool_mask, np.abs(z_score) > stat_thres)


def _correct_outlier_artifact(
    rpeaks: pd.DataFrame,
    bool_mask: np.array,
    art_thres: float,  # pylint:disable=unused-argument
    **kwargs,  # pylint:disable=unused-argument
) -> np.array:
    """Apply outlier correction method 'artifact'.

    This function uses the artifact detection algorithm from `Berntson et al. (1990)`.
    Marks beats as outlier if they detected as such by this algorithm.


    Parameters
    ----------
    rpeaks : :class:`~pandas.DataFrame`
        dataframe with detected R peaks. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    bool_mask : :class:`numpy.array`
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    art_thres : float
        (not used but needed to ensure consistent method interface)


    Returns
    -------
    :class:`numpy.array`
        boolean array with beats marked as outlier. Logical 'or' combination of ``bool_mask`` and results from
        this algorithm


    References
    ----------
    Berntson, G. G., Quigley, K. S., Jang, J. F., & Boysen, S. T. (1990). An Approach to Artifact Identification:
    Application to Heart Period Data. *Psychophysiology*, 27(5), 586–598.
    https://doi.org/10.1111/j.1469-8986.1990.tb01982.x

    """
    # QD = Quartile Deviation = IQR / 2
    qd = iqr(rpeaks["RR_Interval"], nan_policy="omit") / 2.0
    # MAD = Minimal Artifact Difference
    mad = (rpeaks["RR_Interval"].median() - 2.9 * qd) / 3.0
    # MED = Maximum Expected Difference
    med = 3.32 * qd
    criterion = np.mean([mad, med])

    return np.logical_or(
        bool_mask,
        np.abs(rpeaks["RR_Interval"] - rpeaks["RR_Interval"].median()) > criterion,
    )


def _correct_outlier_physiological(
    rpeaks: pd.DataFrame, bool_mask: np.array, hr_thres: Tuple[float, float], **kwargs  # pylint:disable=unused-argument
) -> np.array:
    """Apply outlier correction method 'physiological'.

    This function marks beats as outlier if their heart rate is above or below a threshold that is very unlikely to be
    achieved physiologically.

    Parameters
    ----------
    rpeaks : :class:`~pandas.DataFrame`
        dataframe with detected R peaks. Output from :meth:`biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    bool_mask : :class:`numpy.array`
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    hr_thres : tuple
        lower and upper heart rate thresholds. Beats below and above these values will be marked as outlier.


    Returns
    -------
    :class:`numpy.array`
        boolean array with beats marked as outlier. Logical 'or' combination of ``bool_mask`` and results from
        this algorithm

    """
    # physiological outlier: minimum/maximum heart rate threshold
    bool_mask = np.logical_or(
        bool_mask,
        (rpeaks["RR_Interval"] > (60 / hr_thres[0])) | (rpeaks["RR_Interval"] < (60 / hr_thres[1])),
    )
    return bool_mask


def _get_outlier_params(
    outlier_correction: Optional[Union[str, None, Sequence[str]]] = "all",
    outlier_params: Optional[Dict[str, Union[float, Sequence[float]]]] = None,
) -> Tuple[Sequence[str], Dict[str, Union[float, Sequence[float]]], Dict[str, Callable]]:

    if outlier_correction == "all":
        outlier_correction = list(_outlier_correction_methods.keys())
    elif isinstance(outlier_correction, str):
        outlier_correction = [outlier_correction]
    elif outlier_correction in ["None", None]:
        outlier_correction = []

    try:
        outlier_funcs: Dict[str, Callable] = {key: _outlier_correction_methods[key] for key in outlier_correction}
    except KeyError as e:
        raise ValueError(
            "`outlier_correction` may only contain values from {}, None or `all`, not `{}`.".format(
                list(_outlier_correction_methods.keys()), outlier_correction
            )
        ) from e

    if outlier_params is None:
        outlier_params = {key: _outlier_correction_params_default[key] for key in outlier_funcs}

    # get outlier params (values not passed as arguments will be filled with default arguments)
    outlier_params = {
        key: outlier_params[key] if key in outlier_params else _outlier_correction_params_default[key]
        for key in outlier_funcs
    }
    return outlier_correction, outlier_params, outlier_funcs


_hrv_methods = {
    "hrv_time": nk.hrv_time,
    "hrv_nonlinear": nk.hrv_nonlinear,
    "hrv_frequency": nk.hrv_frequency,
}

_edr_methods = {
    "peak_trough_mean": _edr_peak_trough_mean,
    "peak_trough_diff": _edr_peak_trough_diff,
    "peak_peak_interval": _edr_peak_peak_interval,
}

_outlier_correction_methods: Dict[str, Callable] = {
    "correlation": _correct_outlier_correlation,
    "quality": _correct_outlier_quality,
    "artifact": _correct_outlier_artifact,
    "physiological": _correct_outlier_physiological,
    "statistical_rr": _correct_outlier_statistical_rr,
    "statistical_rr_diff": _correct_outlier_statistical_rr_diff,
}

_outlier_correction_params_default: Dict[str, Union[float, Sequence[float]]] = {
    "correlation": 0.3,
    "quality": 0.4,
    "artifact": 0.0,
    "physiological": (45, 200),
    "statistical_rr": 2.576,
    "statistical_rr_diff": 1.96,
}


def _assert_rpeaks_input(
    ecg_processor: "EcgProcessor",
    key: str,
    rpeaks: RPeakDataFrame,
) -> None:
    """Assert valid input for ECG processing functions that require only R peaks.

    This function checks if either ``ecg_processor`` **and** ``key`` are supplied as arguments *or*
    ``rpeaks``.

    Parameters
    ----------
    ecg_processor : :class:`~biopsykit.signals.ecg.EcgProcessor`, optional
        ``EcgProcessor`` object. If this argument is supplied, the ``key`` argument needs to be supplied as well
    key : str, optional
        Dictionary key of the phase to process. Needed when ``ecg_processor`` is passed as argument
    rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
        Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process`

    Raises
    ------
    ValueError
        if input is invalid

    """
    if all(x is None for x in [ecg_processor, key]) and rpeaks is None:
        raise ValueError("Either 'ecg_processor' and 'key', or 'rpeaks' must be passed as arguments!")
    if ecg_processor is not None and key is None:
        raise ValueError("Both of 'ecg_processor' and 'key' must be passed as arguments!")
    if ecg_processor is None and rpeaks is None:
        raise ValueError("'rpeaks' must be passed as arguments when 'ecg_processor' is None!")


def _assert_ecg_input(ecg_processor: "EcgProcessor", key: str, ecg_signal: EcgResultDataFrame, rpeaks: RPeakDataFrame):
    """Assert valid input for ECG processing functions that require both only ECG signal and R peaks.

    This function checks if either ``ecg_processor`` **and** ``key`` are supplied as arguments *or*
    ``ecg_signal`` **and** `rpeaks`.

    Parameters
    ----------
    ecg_processor : :class:`~biopsykit.signals.ecg.EcgProcessor`, optional
        ``EcgProcessor`` object. If this argument is supplied, the ``key`` argument needs to be supplied as well
    key : str, optional
        Dictionary key of the phase to process. Needed when ``ecg_processor`` is passed as argument
    ecg_signal : :class:`~biopsykit.utils.datatype_helper.EcgResultDataFrame`, optional
        Dataframe with processed ECG signal. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`
    rpeaks : :class:`~biopsykit.utils.datatype_helper.RPeakDataFrame`, optional
        Dataframe with detected R peaks. Output from :meth:`~biopsykit.signals.ecg.EcgProcessor.ecg_process()`

    Raises
    ------
    ValueError
        if input is invalid

    """
    if all(x is None for x in [ecg_processor, key]) and all(x is None for x in [ecg_signal, rpeaks]):
        raise ValueError("Either 'ecg_processor' and 'key', or 'rpeaks' and 'ecg_signal' must be passed as arguments!")
    if ecg_processor is not None and key is None:
        raise ValueError("Both of 'ecg_processor' and 'key' must be passed as arguments!")
    if ecg_signal is not None and rpeaks is None:
        raise ValueError("Both of 'ecg_signal' and 'rpeaks' must be passed as arguments!")


def _get_imputation_type(imputation_type: str, imputation_types: Sequence[str]) -> str:
    if imputation_type is None:
        imputation_type = "moving_average"
    elif imputation_type not in imputation_types:
        raise ValueError("'imputation_type' must be one of {}, not {}!".format(imputation_types, imputation_type))
    return imputation_type


def _check_dataframe_format(ecg_signal: EcgResultDataFrame, rpeaks: RPeakDataFrame):
    if ecg_signal is not None:
        is_ecg_result_dataframe(ecg_signal)
    is_r_peak_dataframe(rpeaks)
