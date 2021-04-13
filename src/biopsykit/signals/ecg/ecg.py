from typing import Optional, Dict, Tuple, Union, Sequence, Callable, List

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal as ss

from biopsykit.utils.datatype_helper import HeartRateSubjectDict
from tqdm.notebook import tqdm
import pytz

from biopsykit.utils.data_processing import split_data
from biopsykit.utils.array_handling import (
    find_extrema_in_radius,
    remove_outlier_and_interpolate,
    sanitize_input_1d,
)
from biopsykit.utils.time import tz, utc, check_tz_aware


class EcgProcessor:
    """
    Class for ECG processing pipeline. Simply pass a pandas dataframe, a dictionary of pandas dataframes,
    or a Dataset object from NilsPodLib and start processing ECG data.

    Each instance of `EcgProcessor` has four important attributes:

    self.data_dict : dict
        Dictionary with raw data, split into the specified sub-phases.
        If data is not split the dictionary only has one entry, accessible by the key 'Data'

    self.ecg_result : dict
        Dictionary with ECG processing results

        **Columns**:
            * ECG_Raw: Raw ECG signal
            * ECG_Clean: Cleaned (filtered) ECG signal
            * ECG_Quality: Quality indicator in the range of [0,1] for ECG signal quality
            * ECG_R_Peaks: 1.0 where R peak was detected in the ECG signal, 0.0 else
            * R_Peak_Outlier: 1.0 when a detected R peak was classified as outlier, 0.0 else
            * Heart_Rate: Computed Heart rate interpolated to signal length

    self.heart_rate : dict
        Dictionary with heart rate data derived from the ECG signal

        **Columns**:
            * Heart_Rate: Computed heart rate for each detected R peak

    self.rpeaks : dict
        Dictionary with R peak location indices derived from the ECG signal

        **Columns**:
            * R_Peak_Quality: Quality indicator in the range of [0,1] for signal quality
            * R_Peak_Idx: Index of detected R peak in the raw ECG signal
            * RR_Interval: Interval between the current and the successive R peak in seconds
            * R_Peak_Outlier: 1.0 when a detected R peak was classified as outlier, 0.0 else

    """

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        df: Optional[pd.DataFrame] = None,
        time_intervals: Optional[Union[pd.Series, Dict[str, Sequence[str]]]] = None,
        include_start: Optional[bool] = False,
        sampling_rate: Optional[float] = 256.0,
        timezone: Optional[Union[pytz.timezone, str]] = tz,
    ):
        """
        Initializes an `EcgProcessor` instance that can be used for ECG processing.

        You can either pass a data dictionary 'data_dict' containing ECG data, a Dataset object directly from
        NilsPodLib or a dataframe containing ECG data. For the latter both, you can additionally supply time
        information via `time_intervals` parameter to automatically split the data into subphases.

        Parameters
        ----------
        data_dict : dict, optional
            Dictionary with pandas dataframes containing ECG data
        df : pd.DataFrame, optional
            pandas dataframe with ECG data
        time_intervals : dict or pd.Series, optional
            time intervals indicating where the data should be split.
            Can either be a pandas Series with the `start` times of the single phases
            (the names of the phases are then derived from the index) or a dictionary with tuples indicating
            start and end times of the phases (the names of the phases are then derived from the dict keys)
            Default: ``None``
        include_start: bool, optional
            ``True`` to include the data from the beginning of the recording to the first time interval as the
            first interval, ``False`` otherwise. Default: ``False``
        sampling_rate : float, optional
            sampling rate of recorded data (not necessary if ``dataset`` is passed, then it is inferred from the dataset
            header)
        timezone : str or pytz.timezone, optional
            timezone of the acquired data to convert, either as string of as pytz object (default: 'Europe/Berlin')

        Raises
        ------
        ValueError
            If None of 'dataset', 'df', or 'data_dict' are supplied


        Examples
        --------
        >>> # Example using NilsPod Dataset
        >>> import biopsykit as bp
        >>> from biopsykit.io.nilspod import load_dataset_nilspod
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> import pandas as pd
        >>> from nilspodlib import Dataset
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
        >>> dataset = Dataset.from_bin_file(file_path)
        >>> df, sampling_rate = load_dataset_nilspod(dataset=dataset, datastreams=['ecg'], timezone=timezone)
        >>> ecg_processor = EcgProcessor(df=df, sampling_rate=sampling_rate, time_intervals=time_intervals)
        """

        if all([i is None for i in [df, data_dict]]):
            raise ValueError("Either 'df' or 'data_dict' must be specified as parameter!")

        self.sampling_rate: int = int(sampling_rate)

        if data_dict:
            self.data_dict = data_dict
        else:
            # check if localized
            if isinstance(df.index, pd.DatetimeIndex) and not check_tz_aware(df):
                # localize dataframe
                df = df.tz_localize(tz=utc).tz_convert(tz=timezone)

            if time_intervals is not None:
                # split data into subphases if time_intervals are passed
                data_dict = split_data(
                    time_intervals,
                    df=df,
                    timezone=timezone,
                    include_start=include_start,
                )
            else:
                data_dict = {"Data": df}

        self.data_dict: Dict[str, pd.DataFrame] = data_dict
        self.ecg_result: Dict[str, pd.DataFrame] = {}
        """
        Dictionary with ECG processing results

        **Columns**:
            * ECG_Raw: Raw ECG signal
            * ECG_Clean: Cleaned (filtered) ECG signal
            * ECG_Quality: Quality indicator in the range of [0,1] for ECG signal quality
            * ECG_R_Peaks: 1.0 where R peak was detected in the ECG signal, 0.0 else
            * R_Peak_Outlier: 1.0 when a detected R peak was classified as outlier, 0.0 else
            * Heart_Rate: Computed Heart rate interpolated to signal length
        """

        self.heart_rate: HeartRateSubjectDict = {}
        """
        self.heart_rate : dict
        :obj:`~biopsykit.utils.datatype_helper.HeartRateSubjectDict`, a dictionary with heart rate data
        derived from the ECG signal

        **Columns**:
            * Heart_Rate: Computed heart rate for each detected R peak
        """

        self.rpeaks: Dict[str, pd.DataFrame] = {}
        """
        Dictionary with R peak location indices derived from the ECG signal

        **Columns**:
            * R_Peak_Quality: Quality indicator in the range of [0,1] for signal quality
            * R_Peak_Idx: Index of detected R peak in the raw ECG signal
            * RR_Interval: Interval between the current and the successive R peak in seconds
            * R_Peak_Outlier: 1.0 when a detected R peak was classified as outlier, 0.0 else
        """

    @property
    def ecg(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the filtered ECG signal.

        Returns
        -------
        dict
            Dictionary with filtered ECG signal per sub-phase.
        """
        return {k: pd.DataFrame(v["ECG_Clean"]) for k, v in self.ecg_result.items()}

    @property
    def hr_result(self) -> Dict[str, pd.DataFrame]:
        """
        Returns the heart rate computation result from ECG R peak detection.

        Returns
        -------
        dict
            Dictionary with heart rate per sub-phase.
        """
        return self.heart_rate

    @property
    def phases(self) -> Sequence[str]:
        return list(self.ecg_result.keys())

    def ecg_process(
        self,
        outlier_correction: Optional[Union[str, None, Sequence[str]]] = "all",
        outlier_params: Optional[Union[str, Dict[str, Union[float, Sequence[float]]]]] = "default",
        title: Optional[str] = None,
        method: Optional[str] = "neurokit",
    ) -> None:
        """
        Processes the ECG signal and optionally performs outlier correction (see `correct_outlier`).


        Parameters
        ----------
        outlier_correction : list or 'all' or None, optional
            List containing the outlier correction methods to be applied. Pass ``None`` to not apply any outlier
            correction, ``all`` to apply all available outlier correction methods.
            See `EcgProcessor.outlier_corrections` to get a list of possible outlier correction. Default: ``all``
        outlier_params : dict
            Dictionary of parameters to be passed to the outlier correction methods.
            See `EcgProcessor.outlier_params_default` for the default parameters. Default: ``default``
        title : str
            optional title of the bar showing processing progress in Jupyter Notebooks
        method : {'neurokit', 'hamilton', 'pantompkins', 'elgendi', ... }, optional
            method for cleaning the ECG signal and R peak detection as defined by 'neurokit'. Default: ``neurokit``


        See Also
        --------
        EcgProcessor.correct_outlier, EcgProcessor.outlier_corrections, EcgProcessor.outlier_params_default


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

        for name, df in tqdm(self.data_dict.items(), desc=title):
            ecg_result, rpeaks = self._ecg_process(df, method=method)
            ecg_result, rpeaks = self.correct_outlier(
                ecg_signal=ecg_result,
                rpeaks=rpeaks,
                outlier_correction=outlier_correction,
                outlier_params=outlier_params,
                sampling_rate=self.sampling_rate,
            )
            heart_rate = pd.DataFrame({"Heart_Rate": 60 / rpeaks["RR_Interval"]})
            heart_rate_interpolated = nk.signal_interpolate(
                x_values=np.squeeze(rpeaks["R_Peak_Idx"].values),
                y_values=np.squeeze(heart_rate["Heart_Rate"].values),
                x_new=np.arange(0, len(ecg_result["ECG_Clean"])),
            )
            ecg_result["Heart_Rate"] = heart_rate_interpolated
            self.ecg_result[name] = ecg_result
            self.heart_rate[name] = heart_rate
            self.rpeaks[name] = rpeaks

    def _ecg_process(self, data: pd.DataFrame, method: Optional[str] = "neurokit") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Private method to perform the actual ECG processing.

        Parameters
        ----------
        data : pd.DataFrame
            ECG data as pandas dataframe. Needs to have one column named 'ecg'
        method : {'neurokit', 'hamilton', 'pantompkins', 'elgendi', ... }, optional
            method for cleaning the ECG signal and R peak detection as defined by 'neurokit'. Default: ``neurokit``

        Returns
        -------
        tuple
            A tuple of two dataframes: one containing the processed ECG signal, one containing the detected R peaks
        """
        # get numpy
        ecg_signal = data["ecg"].values
        # clean (i.e. filter) the ECG signal using the specified method
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate, method=method)

        # find peaks using the specified method
        # instant_peaks: array indicating where detected R peaks are in the raw ECG signal
        # rpeak_index array containing the indices of detected R peaks
        instant_peaks, rpeak_idx = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate, method=method)
        rpeak_idx = rpeak_idx["ECG_R_Peaks"]
        instant_peaks = np.squeeze(instant_peaks.values)

        # compute quality indicator
        quality = nk.ecg_quality(ecg_cleaned, rpeaks=rpeak_idx, sampling_rate=self.sampling_rate)

        # construct new dataframe
        ecg_signal_output = pd.DataFrame(
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
        rpeaks = ecg_signal_output.loc[ecg_signal_output["ECG_R_Peaks"] == 1.0, ["ECG_Quality"]]
        rpeaks.rename(columns={"ECG_Quality": "R_Peak_Quality"}, inplace=True)
        rpeaks.loc[:, "R_Peak_Idx"] = rpeak_idx
        # compute RR interval
        rpeaks["RR_Interval"] = np.ediff1d(rpeaks["R_Peak_Idx"], to_end=0) / self.sampling_rate
        # ensure equal length by filling the last value with the average RR interval
        rpeaks.loc[rpeaks.index[-1], "RR_Interval"] = rpeaks["RR_Interval"].mean()

        return ecg_signal_output, rpeaks

    @classmethod
    def outlier_corrections(cls) -> Sequence[str]:
        """
        Returns the keys of all possible outlier correction methods.

        Currently available outlier correction methods are:
            * `correlation`: Computes the cross-correlation coefficient between every single beat and the average of all
              detected beats. Marks beats as outlier if cross-correlation coefficient is below a certain threshold
            * `quality`: Uses the 'ECG_Quality' indicator from neurokit to assess signal quality. Marks beats as outlier if
              quality indicator of beat is below a certain threshold
            * `artifact`: Artifact detection based on work from `Berntson et al. (1990), Psychophysiology`
            * `physiological`: Physiological outlier removal. Marks beats as outlier if their heart rate is above or below a
              threshold that can not be achieved physiologically
            * `statistical_rr`: Statistical outlier removal based on RR intervals.
              Marks beats as outlier if the RR intervals are within the xx% highest or lowest values.
              Values are removed based on the z-score (e.g. 1.96 => 5%, 2.5% highest, 2.5% lowest values;
              2.576 => 1 %, 0.5 % highest, 0.5 % lowest values)
            * `statistical_rr_diff`: Statistical outlier removal based on successive differences of RR intervals.
            Marks beats as outlier if the difference of successive RR intervals are within the xx% highest or
              lowest heart rates. Values are removed based on the z-score
              (e.g. 1.96 => 5%, 2.5% highest, 2.5% lowest values; 2.576 => 1 %, 0.5 % highest, 0.5 % lowest values)

        See Also
        --------
        EcgProcessor.correct_outlier, EcgProcessor.outlier_params_default

        Returns
        -------
        list
            List containing the keys of all possible outlier correction methods
        """
        return list(_outlier_correction_methods.keys())

    @classmethod
    def outlier_params_default(cls) -> Dict[str, Union[float, Sequence[float]]]:
        """
        Returns all default parameters for outlier correction methods.


        **NOTE:** outlier correction method `artifact` has no threshold, but '0.0' is default parameter to provide
        a homogenous interface


        See Also
        --------
        EcgProcessor.correct_outlier, EcgProcessor.outlier_corrections

        Returns
        -------
        dict
            Dictionary containing the default parameters for the different outlier correction methods
        """
        return _outlier_correction_params_default

    @classmethod
    def correct_outlier(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[pd.DataFrame] = None,
        rpeaks: Optional[pd.DataFrame] = None,
        outlier_correction: Optional[Union[str, None, Sequence[str]]] = "all",
        outlier_params: Optional[Union[str, Dict[str, Union[float, Sequence[float]]]]] = "default",
        imputation_type: Optional[str] = "moving_average",
        sampling_rate: Optional[int] = 256,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs outlier correction of the detected R peaks.

        Different methods for outlier detection are available (see ``EcgProcessor.outlier_corrections()`` to get a list
        of possible outlier correction methods). All outlier methods work independently on the detected R peaks,
        the results will be combined by a logical 'or'. RR intervals classified as outlier will be removed and imputed
        either using linear interpolation (``imputation_type`` 'linear') or by replacing it with the average value
        of the 10 preceding and 10 succeding RR intervals (``imputation_type`` 'moving_average').

        To use this function, either simply pass an ``EcgProcessor`` object together with a ``key`` indicating
        which sub-phase should be processed or the two dataframes ``ecg_signal`` and ``rpeaks`` resulting from
        ``EcgProcessor.ecg_process()``.

        Parameters
        ----------
        ecg_processor : EcgProcessor, optional
            `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
        key : str, optional
            Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
        ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
        rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
        outlier_correction : list, optional
            List containing the outlier correction methods to be applied.
            Pass ``None`` to not apply any outlier correction, ``all`` to apply all available outlier correction
            methods. See `EcgProcessor.outlier_corrections` to get a list of possible outlier correction methods.
            Default: ``all``
        outlier_params: dict, optional
            Dictionary of parameters to be passed to the outlier correction methods or ``default``
            for default parameters (see `EcgProcessor.outlier_params_default` for more information).
            Default: ``default``
        imputation_type: str, optional
            Method to impute outlier: `linear` for linear interpolation between the RR intervals before and
            after R peak outlier, `moving_average` for average value of the 10 preceding and 10 succeding RR intervals.
            Default: ``moving_average``
        sampling_rate : float, optional
            Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz

        Returns
        -------
        tuple
            A tuple of two dataframes: one containing the processed ECG signal, one containing the detected R peaks

        See Also
        --------
        EcgProcessor.ecg_process, EcgProcessor.outlier_corrections, EcgProcessor.outlier_params_default

        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # use default outlier correction pipeline
        >>> ecg_signal, rpeaks = ecg_processor.correct_outlier(ecg_processor, key="Data")

        >>> # use custom outlier correction pipeline: only physiological and statistical RR interval outlier with
        >>> # custom thresholds
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
        """

        check_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        if outlier_correction == "all":
            outlier_correction = list(_outlier_correction_methods.keys())
        elif outlier_correction in ["None", None]:
            outlier_correction = list()

        imputation_types = ["linear", "moving_average"]
        if imputation_type not in imputation_types:
            raise ValueError("`imputation_type` must be one of {}, not {}!".format(imputation_types, imputation_type))

        try:
            outlier_funcs: Dict[str, Callable] = {key: _outlier_correction_methods[key] for key in outlier_correction}
        except KeyError:
            raise ValueError(
                "`outlier_correction` may only contain values from {}, None or `all`, not `{}`.".format(
                    list(_outlier_correction_methods.keys()), outlier_correction
                )
            )

        if outlier_params in ["default", None]:
            outlier_params = {key: _outlier_correction_params_default[key] for key in outlier_funcs}

        # get outlier params (values not passed as arguments will be filled with default arguments)
        outlier_params = {
            key: outlier_params[key] if key in outlier_params else _outlier_correction_params_default[key]
            for key in outlier_funcs
        }

        # copy dataframe to mark removed beats later
        rpeaks_copy = rpeaks.copy()
        # get the last index because it will get lost when computing the RR interval
        last_idx = rpeaks.iloc[-1]

        # initialize bool mask to mask outlier and add outlier column to rpeaks dataframe
        bool_mask = np.full(rpeaks.shape[0], False)
        rpeaks["R_Peak_Outlier"] = 0.0

        # TODO add source of different outlier methods for plotting?
        for key in outlier_funcs:
            bool_mask = outlier_funcs[key](ecg_signal, rpeaks, sampling_rate, bool_mask, outlier_params[key])

        # mark all removed beats as outlier in the ECG dataframe
        rpeaks[bool_mask] = None
        removed_beats = rpeaks_copy["R_Peak_Idx"][rpeaks["R_Peak_Idx"].isna()]
        # mark all outlier with 1.0 in the column R_Peak_Outlier column
        rpeaks.fillna({"R_Peak_Outlier": 1.0}, inplace=True)
        # also mark outlier in the ECG signal dataframe
        ecg_signal.loc[removed_beats.index, "R_Peak_Outlier"] = 1.0

        # interpolate the removed beats
        rpeaks.loc[rpeaks.index[-1]] = [
            rpeaks["R_Peak_Quality"].mean(),
            last_idx["R_Peak_Idx"],
            rpeaks["RR_Interval"].mean(),
            0.0,
        ]

        if imputation_type == "moving_average":
            rpeaks["RR_Interval"] = rpeaks["RR_Interval"].fillna(
                rpeaks["RR_Interval"].rolling(21, center=True, min_periods=0).mean()
            )

        rpeaks.interpolate(method="linear", limit_direction="both", inplace=True)
        # drop duplicate R peaks (can happen during outlier correction at edge cases)
        rpeaks.drop_duplicates(subset="R_Peak_Idx", inplace=True)

        return ecg_signal, rpeaks

    @classmethod
    def correct_rpeaks(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[pd.DataFrame] = None,
        rpeaks: Optional[pd.DataFrame] = None,
        sampling_rate: Optional[int] = 256,
    ) -> pd.DataFrame:
        """
        Performs an R peak correction algorithms to get less noisy HRV parameters. R peak correction comes from Neurokit
        and is based on an algorithm by `Lipponen et al. (2019), Journal of medical engineering & technology`.

        To use this function, either simply pass an `EcgProcessor` object together with a `key` indicating
        which sub-phase should be processed or the two dataframes `ecg_signal` and `rpeaks` resulting from
        `EcgProcessor.ecg_process()`.


        **NOTE** This algorithm might add additional R peaks or remove certain ones, so results of this function
        might not match with the R peaks of `EcgProcessor.rpeaks` or might not be used in combination with
        `EcgProcessor.ecg` since indices won't match.

        In this library it is **not** generally applied to the detected R peaks but only used right before passing
        R peaks to `EcgProcessor.hrv_process()`.


        Parameters
        ----------
        ecg_processor : EcgProcessor, optional
            `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
        key : str, optional
            Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
        ecg_signal : pd.DataFrame, optional
            dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
        rpeaks : pd.DataFrame, optional
            dataframe with R peaks. Output of `EcgProcessor.ecg_process()`
        sampling_rate : float, optional
            Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz

        Returns
        -------
        pd.DataFrame
            dataframe containing corrected R peak indices

        # TODO reference

        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # correct R peak locations
        >>> rpeaks_corrected = ecg_processor.correct_rpeaks(ecg_processor, key="Data")
        """

        check_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            rpeaks = ecg_processor.rpeaks[key]
            ecg_signal = ecg_processor.ecg_result[key]
            sampling_rate = ecg_processor.sampling_rate

        # fill missing RR intervals with interpolated R Peak Locations
        rpeaks_corrected = (rpeaks["RR_Interval"].cumsum() * sampling_rate).astype(int)
        rpeaks_corrected = np.append(
            rpeaks["R_Peak_Idx"].iloc[0],
            rpeaks_corrected.iloc[:-1] + rpeaks["R_Peak_Idx"].iloc[0],
        )
        artifacts, rpeaks_corrected = nk.signal_fixpeaks(rpeaks_corrected, sampling_rate, iterative=False)
        rpeaks_corrected = rpeaks_corrected.astype(int)
        return pd.DataFrame(rpeaks_corrected, columns=["R_Peak_Idx"])

    @classmethod
    def hrv_process(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[pd.DataFrame] = None,
        rpeaks: Optional[pd.DataFrame] = None,
        hrv_types: Optional[Sequence[str]] = ["hrv_time", "hrv_nonlinear"],
        correct_rpeaks: Optional[bool] = True,
        index: Optional[str] = None,
        index_name: Optional[str] = None,
        sampling_rate: Optional[int] = 256,
    ) -> pd.DataFrame:
        """
        Computes HRV features on the given data. By default it applies R peak correction
        (see `EcgProcessor.correct_rpeaks()`) before computing HRV features.

        To use this function, either simply pass an `EcgProcessor` object together with a `key` indicating
        which sub-phase should be processed or the two dataframes `ecg_signal` and `rpeaks` resulting from
        `EcgProcessor.ecg_process()`.

        Parameters
        ----------
        ecg_processor : EcgProcessor, optional
            `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
        key : str, optional
            Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
        ecg_signal : pd.DataFrame, optional
            dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
        rpeaks : pd.DataFrame, optional
            dataframe with R peaks. Output of `EcgProcessor.ecg_process()`
        hrv_types: list or str, optional
            List of HRV types to be computed. Must be a subset of ['hrv_time', 'hrv_nonlinear', 'hrv_frequency']
            or 'all' to compute all types of HRV.
            Default: ['hrv_time', 'hrv_nonlinear']. Refer to `neurokit.hrv` for further information on HRV
        correct_rpeaks : bool, optional
            Flag indicating whether to apply R peak correction (using `EcgProcessor.correct_rpeaks()`)
            before computing HRV features. Default: True
        index: str, optional
            Index value of the computed HRV features. Used to concatenate dataframes from multiple sub-phases later.
            Default: None
        index_name : str, optional
            Index name of the output dataframe. Only used if 'index' is also supplied
        sampling_rate : float
            Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz

        Returns
        -------
        dataframe
            dataframe with computed HRV features


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
        check_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        if correct_rpeaks:
            rpeaks = cls.correct_rpeaks(ecg_signal=ecg_signal, rpeaks=rpeaks, sampling_rate=sampling_rate)

        if hrv_types == "all":
            hrv_types = list(_hrv_methods.keys())

        # check whether all supplied hrv_types are valid
        for hrv_type in hrv_types:
            if hrv_type not in _hrv_methods:
                raise ValueError("`hrv_types` must be in {}, not {}".format(list(_hrv_methods.keys()), hrv_type))
        hrv_methods = {key: _hrv_methods[key] for key in hrv_types}

        # compute all HRV parameters
        list_hrv: List[pd.DataFrame] = [
            hrv_methods[key](rpeaks["R_Peak_Idx"], sampling_rate=sampling_rate) for key in hrv_methods
        ]
        # concat dataframe list
        hrv = pd.concat(list_hrv, axis=1)

        if index:
            # set index to dataframe if supplied
            hrv.index = [index]
            hrv.index.name = index_name
        return hrv

    def hrv_batch_process(self, hrv_types: Optional[Sequence[str]] = ["hrv_time", "hrv_nonlinear"]) -> pd.DataFrame:
        """
        Computes HRV (using `EcgProcessor.hrv_process()`) over all phases.

        Parameters
        ----------
        hrv_types: list or str, optional
            List of HRV types to be computed. Must be a subset of ['hrv_time', 'hrv_nonlinear', 'hrv_frequency']
            or 'all' to compute all types of HRV.
            Default: ['hrv_time', 'hrv_nonlinear']. Refer to `neurokit.hrv` for further information on HRV

        Returns
        -------
        dataframe
            Dataframe with HRV parameters over all phases

        """
        return pd.concat([self.hrv_process(self, key=key, index=key, hrv_types=hrv_types) for key in self.phases])

    @classmethod
    def ecg_extract_edr(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[pd.DataFrame] = None,
        rpeaks: Optional[pd.DataFrame] = None,
        edr_type: Optional[str] = "peak_trough_mean",
        sampling_rate: Optional[int] = 256,
    ) -> pd.DataFrame:
        """
        Estimate respiration signal from ECG signal (ECG-derived respiration, EDR).

        To use this function, either simply pass an `EcgProcessor` object together with a `key` indicating
        which sub-phase should be processed or the two dataframes `ecg_signal` and `rpeaks` resulting from
        `EcgProcessor.ecg_process()`.


        Parameters
        ----------
        ecg_processor : EcgProcessor, optional
            `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
        key : str, optional
            Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
        ecg_signal : str, optional
            dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
        rpeaks : str, optional
            dataframe with R peaks. Output of `EcgProcessor.ecg_process()`
        edr_type : {'peak_trough_mean', 'peak_trough_diff', 'peak_peak_interval'}, optional
            Method to use for estimating EDR. Must be one of 'peak_trough_mean', 'peak_trough_diff',
            or 'peak_peak_interval'. Default: 'peak_trough_mean'
        sampling_rate : float, optional
            Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz

        Returns
        -------
        dataframe
            dataframe containing the estimated respiration signal

        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # Extract respiration signal estimated from ECG using the 'peak_trough_diff' method
        >>> rsp_signal = ecg_processor.ecg_extract_edr(ecg_processor, key="Data", edr_type='peak_trough_diff')
        """

        check_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

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
        edr_signal = nk.signal_filter(edr_signal, sampling_rate=sampling_rate, lowcut=0.1, highcut=0.5, order=10)

        return pd.DataFrame(edr_signal, index=ecg_signal.index, columns=["ECG_Resp"])

    @classmethod
    def rsp_compute_rate(cls, rsp_signal: pd.DataFrame, sampling_rate: Optional[int] = 256) -> float:
        """
        Compute respiration rate for given interval from respiration signal. Based on `Karlen et al. (2013), TBME`.

        Parameters
        ----------
        rsp_signal : pd.DataFrame
            raw respiration signal (1D). Can be a 'true' respiration signal (e.g. from bioimpedance or Radar)
            or an 'estimated' respiration signal (e.g. from ECG-derived respiration)
        sampling_rate : float, optional
            Sampling rate of recorded data

        Returns
        -------
        float
            Respiration rate during the given interval in bpm (breaths per minute)

        # TODO reference

        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> # Extract respiration signal estimated from ECG using the 'peak_trough_diff' method
        >>> rsp_signal = ecg_processor.ecg_extract_edr(ecg_processor, key="Data", edr_type='peak_trough_diff')
        >>> # Compute respiration rate from respiration signal
        >>> rsp_rate = ecg_processor.rsp_compute_rate(rsp_signal)
        """

        # find peaks: minimal distance between peaks: 1 seconds
        rsp_signal = sanitize_input_1d(rsp_signal)
        edr_maxima = ss.find_peaks(rsp_signal, height=0, distance=sampling_rate)[0]
        edr_minima = ss.find_peaks(-1 * rsp_signal, height=0, distance=sampling_rate)[0]
        # threshold: 0.2 * Q3 (= 75th percentile)
        max_threshold = 0.2 * np.percentile(rsp_signal[edr_maxima], 75)
        # find all maxima that are above the threshold
        edr_maxima = edr_maxima[rsp_signal[edr_maxima] > max_threshold]

        # rearrange maxima into 2D array to that each row contains the start and end of one cycle
        rsp_cycles_start_end = np.vstack([edr_maxima[:-1], edr_maxima[1:]]).T

        # check for
        valid_resp_phases_mask = np.apply_along_axis(_check_contains_trough, 1, rsp_cycles_start_end, edr_minima)
        rsp_cycles_start_end = rsp_cycles_start_end[valid_resp_phases_mask]

        edr_signal_split: list = np.split(rsp_signal, rsp_cycles_start_end.flatten())[1::2]
        rsp_cycles_start_end = rsp_cycles_start_end.T
        rsp_peaks = rsp_cycles_start_end[0]
        rsp_troughs = np.array(list(map(lambda arr: np.argmin(arr), edr_signal_split))) + rsp_cycles_start_end[0]

        rsp_rate_peaks = _rsp_rate(rsp_peaks, sampling_rate, len(rsp_signal))
        rsp_rate_troughs = _rsp_rate(rsp_troughs, sampling_rate, len(rsp_signal))
        return np.concatenate([rsp_rate_peaks, rsp_rate_troughs]).mean()

    @classmethod
    def rsa_process(
        cls,
        ecg_signal: pd.DataFrame,
        rsp_signal: pd.DataFrame,
        sampling_rate: Optional[int] = 256,
    ) -> Dict[str, float]:
        """
        Computes respiratory sinus arrhythmia (RSA) based on ECG and respiration signal. RSA is computed both
        using the Peak-to-Trough (P2T) method and the Porges-Bohrer method. See neurokit ('nk.hrv_rsa')
        for further information.


        Parameters
        ----------
        ecg_signal : pd.DataFrame
            dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
        rsp_signal : pd.DataFrame
            raw respiration signal (1D). Can be a 'true' respiration signal (e.g. from bioimpedance or Radar)
            or an 'estimated' respiration signal (e.g. from ECG-derived respiration)
        sampling_rate : float, optional
            Sampling rate of recorded data. Default: 256 Hz

        Returns
        -------
        dict
            Dictionary containing computed RSA metrics.

        See Also
        --------
        nk.hrv_rsa

        Examples
        --------
        >>> from biopsykit.signals.ecg import EcgProcessor
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = EcgProcessor(...)

        >>> ecg_signal = ecg_processor.ecg_result['Data']
        >>> # Extract respiration signal estimated from ECG using the 'peak_trough_diff' method
        >>> rsp_signal = ecg_processor.ecg_extract_edr(ecg_processor, key="Data", edr_type='peak_trough_diff')
        >>> # Compute RSA from ECG and Respiration data
        >>> rsa_output = ecg_processor.rsa_process(ecg_signal, rsp_signal)
        """

        # ensure numpy
        rsp_signal = sanitize_input_1d(rsp_signal)
        # Process raw respiration input
        rsp_output = nk.rsp_process(rsp_signal, sampling_rate)[0]
        rsp_output.index = ecg_signal.index
        # Compute RSA
        return nk.hrv_rsa(ecg_signal, rsp_output, sampling_rate=sampling_rate)

    @classmethod
    def rsp_rsa_process(
        cls,
        ecg_processor: Optional["EcgProcessor"] = None,
        key: Optional[str] = None,
        ecg_signal: Optional[pd.DataFrame] = None,
        rpeaks: Optional[pd.DataFrame] = None,
        return_mean: Optional[bool] = True,
        index: Optional[str] = None,
        index_name: Optional[str] = None,
        sampling_rate: Optional[int] = 256,
    ) -> pd.DataFrame:
        """
        Computes respiration rate (RSP) and RSA metrics for the given ECG signal. Per default it estimates the
        respiration signal using all three available EDR methods and averages the results. Optionally, estimation
        results from the individual methods can be returned.s

        To use this function, either simply pass an `EcgProcessor` object together with a `key` indicating
        which sub-phase should be processed or the two dataframes `ecg_signal` and `rpeaks` resulting from
        `EcgProcessor.ecg_process()`.

        Parameters
        ----------
        ecg_processor : EcgProcessor, optional
            `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
        key : str, optional
            Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
        ecg_signal : pd.DataFrame, optional
            dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
        rpeaks : pd.DataFrame, optional
            dataframe with R peaks. Output of `EcgProcessor.ecg_process()`
        return_mean: bool, optional
            Flag indicating whether to return the averaged estimation results or individual results for each EDR
            estimation method.
            Default: ``True``
        index: str, optional
            Index value of the computed RSP and RSA features. Used to concatenate dataframes from multiple sub-phases
            later.
            Default: None
        index_name : str, optional
            Index name of the output dataframe. Only used if 'index' is also supplied
        sampling_rate : float
            Sampling rate of recorded data. Not needed if ``ecg_processor`` is supplied as parameter. Default: 256 Hz

        Returns
        -------
        pd.DataFrame
            dataframe of respiration rate and RSA estimation results

        Examples
        --------
        >>> import biopsykit as ep
        >>> # initialize EcgProcessor instance
        >>> ecg_processor = ep.EcgProcessor(...)

        >>> # Compute respiration rate and RSA. Extract respiration signal using all available
        >>> # methods and average the results ('return_mean' is True by default)
        >>> rsp_signal = ecg_processor.rsp_rsa_process(ecg_processor, key="Data")

        >>> # Compute respiration rate and RSA. Extract respiration signal using all available
        >>> # methods and return the single results
        >>> rsp_signal = ecg_processor.rsp_rsa_process(ecg_processor, key="Data", return_mean=False)
        """

        check_ecg_input(ecg_processor, key, ecg_signal, rpeaks)
        if ecg_processor:
            ecg_signal = ecg_processor.ecg_result[key]
            rpeaks = ecg_processor.rpeaks[key]
            sampling_rate = ecg_processor.sampling_rate

        # initialize dicts to store results
        rsp_rate = dict.fromkeys(_edr_methods.keys())
        rsa = dict.fromkeys(_edr_methods.keys())

        for method in _edr_methods.keys():
            # estimate respiration signal, compute respiration signal and RSA using all three methods
            rsp_signal = cls.ecg_extract_edr(
                ecg_signal=ecg_signal,
                rpeaks=rpeaks,
                sampling_rate=sampling_rate,
                edr_type=method,
            )
            rsp_rate[method] = cls.rsp_compute_rate(rsp_signal, sampling_rate)
            rsa[method] = cls.rsa_process(ecg_signal, rsp_signal, sampling_rate)

        if return_mean:
            # compute average respiration rate and RSA
            mean_resp_rate = np.mean(list(rsp_rate.values()))
            rsa = list(rsa.values())
            mean_rsa = {k: np.mean([t[k] for t in rsa]) for k in rsa[0]}
            mean_rsa["RSP_Rate"] = mean_resp_rate
            # dataframe reshaping
            if not index:
                index = "0"
                index_name = "Index"
            df_rsa = pd.DataFrame(mean_rsa, index=[index])
            df_rsa.index.name = index_name
            return df_rsa
        else:
            # dataframe reshaping
            df_rsa = pd.DataFrame(rsa).T
            df_rsa["RSP_Rate"] = rsp_rate.values()
            df_rsa.index.name = "Method"
            if index:
                return pd.concat([df_rsa], keys=[index], names=[index_name])
            return df_rsa


def normalize_heart_rate(
    dict_hr_subjects: Dict[str, Dict[str, pd.DataFrame]], normalize_to: str
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Normalizes heart rate per subject to the phase specified by `normalize_to`.

    The result is the relative change of heart rate compared to the mean heart rate in the `normalize_to` phase.

    Parameters
    ----------
    dict_hr_subjects : dict
        dictionary with heart rate data of all subjects as returned by `load_hr_excel_all_subjects`
    normalize_to : str
        phase (i.e., dict key) of data to normalize all other data to

    Returns
    -------
    dict
        dictionary with normalized heart rate data per subject
    """

    dict_hr_subjects_norm = {}
    for subject_id, dict_hr in dict_hr_subjects.items():
        bl_mean = dict_hr[normalize_to].mean()
        dict_hr_norm = {phase: (df_hr - bl_mean) / bl_mean * 100 for phase, df_hr in dict_hr.items()}
        dict_hr_subjects_norm[subject_id] = dict_hr_norm

    return dict_hr_subjects_norm


def _edr_peak_trough_mean(ecg: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> np.array:
    """
    Estimates a respiration signal from ECG based on computing the average between peaks (R peaks) and troughs
    (minima right before R peak).


    Parameters
    ----------
    ecg : pd.DataFrame
        dataframe with ecg signal
    peaks : array_like
        array with peak indices
    troughs : array_like
        array with trough indices

    Returns
    -------
    np.array
        estimated raw respiration signal
    """
    peak_vals = np.array(ecg.iloc[peaks])
    trough_vals = np.array(ecg.iloc[troughs])
    return np.mean([peak_vals, trough_vals], axis=0)


def _edr_peak_trough_diff(ecg: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> np.ndarray:
    """
    Estimates a respiration signal from ECG based on computing the difference between peaks (R peaks)
    and troughs (minima right before R peak).

    Parameters
    ----------
    ecg : pd.DataFrame
        dataframe with ecg signal
    peaks : array_like
        array with peak indices
    troughs : array_like
        array with trough indices

    Returns
    -------
    np.array
        estimated raw respiration signal
    """
    peak_vals = np.array(ecg.iloc[peaks])
    trough_vals = np.array(ecg.iloc[troughs])
    return peak_vals - trough_vals


def _edr_peak_peak_interval(ecg: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray) -> np.ndarray:
    """
    Estimates a respiration signal from ECG based on RR intervals.

    Parameters
    ----------
    ecg : pd.DataFrame
        dataframe with ecg signal (unused but needed for consistent method signature)
    peaks : array_like
        array with peak indices
    troughs : array_like
        array with trough indices (unused but needed for consistent method signature)

    Returns
    -------
    np.array
        estimated raw respiration signal
    """
    peak_interval = np.ediff1d(peaks, to_begin=0)
    peak_interval[0] = peak_interval.mean()
    return peak_interval


def _check_contains_trough(start_end: np.ndarray, minima: np.ndarray) -> bool:
    """
    Helper method to check whether exactly one minima is in the interval given by the array `start_end`: [start, end].

    Parameters
    ----------
    start_end : array_like
        array with start and end index
    minima : array_like
        array containing minima to be checked

    Returns
    -------
    bool
        True if exactly one minima is in the [start, end] interval, False otherwise
    """
    start, end = start_end
    return minima[(minima > start) & (minima < end)].shape[0] == 1


def _rsp_rate(extrema: np.ndarray, sampling_rate: int, desired_length: int) -> np.ndarray:
    """
    Computes the continuous respiration rate from extrema values.

    Parameters
    ----------
    extrema: array_like
        list of respiration extrema (peaks or troughs)
    sampling_rate : float
        Sampling rate of recorded data
    desired_length : int
        desired length of the output signal

    Returns
    -------
    array_like
        respiration rate array interpolated to desired length
    """
    rsp_rate_raw = (sampling_rate * 60) / np.ediff1d(extrema)
    # remove last sample
    x_old = extrema[:-1]
    x_new = np.linspace(x_old[0], x_old[-1], desired_length)
    return nk.signal_interpolate(x_old, rsp_rate_raw, x_new, method="linear")


def _correct_outlier_correlation(
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
    sampling_rate: int,
    bool_mask: np.array,
    corr_thres: float,
) -> np.array:
    """
    Outlier correction method 'correlation'.
    Computes the cross-correlation coefficient between every single beat and the average of all detected beats.
    Marks beats as outlier if cross-correlation coefficient is below a certain threshold.

    Parameters
    ----------
    ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
        rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
    sampling_rate : float
        Sampling rate of recorded data
    bool_mask : array_like
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    corr_thres : float
        Threshold for cross-correlation coefficient. Beats below that threshold will be marked as outlier

    Returns
    -------
    array_like
        boolean array with beats marked as outlier. Logical 'or' combination of `bool_mask` and results from
        this algorithm
    """
    # signal outlier
    # segment individual heart beats
    heartbeats = nk.ecg_segment(ecg_signal["ECG_Clean"], rpeaks["R_Peak_Idx"], sampling_rate)
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
    corr_coeff.drop("mean", inplace=True)
    # compute RR intervals (in seconds) from R Peak Locations
    rpeaks["RR_Interval"] = np.ediff1d(rpeaks["R_Peak_Idx"], to_end=0) / sampling_rate

    # signal outlier: drop all beats that are below a correlation coefficient threshold
    return np.logical_or(bool_mask, rpeaks["R_Peak_Idx"].isin(corr_coeff[corr_coeff < corr_thres].index))


def _correct_outlier_quality(
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
    sampling_rate: int,
    bool_mask: np.array,
    quality_thres: float,
) -> np.array:
    """
    Outlier correction method 'quality'.

    Uses the 'ECG_Quality' indicator from neurokit to assess signal quality.
    Marks beats as outlier if quality indicator of beat is below a certain threshold.

    Parameters
    ----------
    ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
        rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
    sampling_rate : float
        Sampling rate of recorded data
    bool_mask : array_like
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    quality_thres : float
        Threshold for signal quality indiicator. Beats below that threshold will be marked as outlier

    Returns
    -------
    array_like
        boolean array with beats marked as outlier. Logical 'or' combination of `bool_mask` and results from
        this algorithm
    """
    # signal outlier: drop all beats that are below a signal quality threshold
    return np.logical_or(bool_mask, rpeaks["R_Peak_Quality"] < quality_thres)


def _correct_outlier_statistical_rr(
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
    sampling_rate: int,
    bool_mask: np.array,
    stat_thres: float,
) -> np.array:
    """
    Outlier correction method 'statistical_rr'.

    Marks beats as outlier if they are within the xx % highest or lowest RR intervals, i.e. if their z-score is
    above a threshold (e.g. 1.96 => 5 %, 2.5 % highest, 2.5 % lowest values;
    2.576 => 1 %, 0.5 % highest, 0.5 % lowest values).

    Parameters
    ----------
    ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
    rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
    sampling_rate : float
        Sampling rate of recorded data
    bool_mask : array_like
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    stat_thres : float
        Threshold for z-score. Beats above that threshold will be marked as outlier

    Returns
    -------
    array_like
        boolean array with beats marked as outlier. Logical 'or' combination of `bool_mask` and results from
        this algorithm
    """
    # statistical outlier: remove the x% highest and lowest RR intervals
    # (1.96 std = 5% outlier, 2.576 std = 1% outlier)
    rri = rpeaks["RR_Interval"]
    z_score = (rri - np.nanmean(rri)) / np.nanstd(rri, ddof=1)

    return np.logical_or(bool_mask, np.abs(z_score) > stat_thres)


def _correct_outlier_statistical_rr_diff(
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
    sampling_rate: int,
    bool_mask: np.array,
    stat_thres: float,
) -> np.array:
    """
    Outlier correction method 'statistical_rr_diff'.

    Marks beats as outlier if their successive differences of RR intervals are within the xx % highest or
    lowest values, i.e. if their z-score is above a threshold (e.g. 1.96 => 5 %, 2.5 % highest, 2.5 % lowest values;
    2.576 => 1 %, 0.5 % highest, 0.5 % lowest values).

    Parameters
    ----------
    ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
    rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
    sampling_rate : float
        Sampling rate of recorded data
    bool_mask : array_like
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    stat_thres : float
        Threshold for z-score. Beats above that threshold will be marked as outlier

    Returns
    -------
    array_like
        boolean array with beats marked as outlier. Logical 'or' combination of `bool_mask` and results from
        this algorithm
    """
    # statistical outlier: remove the x% highest and lowest successive differences of RR intervals
    # (1.96 std = 5% outlier, 2.576 std = 1% outlier)
    diff_rri = np.ediff1d(rpeaks["RR_Interval"], to_end=0)
    z_score = (diff_rri - np.nanmean(diff_rri)) / np.nanstd(diff_rri, ddof=1)

    return np.logical_or(bool_mask, np.abs(z_score) > stat_thres)


def _correct_outlier_artifact(
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
    sampling_rate: int,
    bool_mask: np.array,
    art_thres: float,
) -> np.array:
    """
    Outlier correction method 'artifact'.

    Applies artifact detection algorithm from `Berntson et al. (1990), Psychophysiology`.
    Marks beats as outlier if they detected as such by this algorithm.

    Parameters
    ----------
    ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
    rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
    sampling_rate : float
        Sampling rate of recorded data
    bool_mask : array_like
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    art_thres : float
        Threshold for artifact removal. Not used currently

    Returns
    -------
    array_like
        boolean array with beats marked as outlier. Logical 'or' combination of `bool_mask` and results from
        this algorithm
    """
    from scipy.stats import iqr

    # note: art_thres only needed to have a uniform signature

    # compute artifact-detection criterion based on Berntson et al. (1990), Psychophysiology
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
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
    sampling_rate: int,
    bool_mask: np.array,
    hr_thres: Tuple[float, float],
) -> np.array:
    """
    Outlier correction method 'physiological'.

    Marks beats as outlier if their heart rate is above or below a threshold that can not be achieved
    physiologically.

    Parameters
    ----------
    ecg_signal : pd.DataFrame, optional
            dataframe with processed ECG signal. Output from `EcgProcessor.ecg_process()`
    rpeaks : pd.DataFrame, optional
            dataframe with detected R peaks. Output from `EcgProcessor.ecg_process()`
    sampling_rate : float
        Sampling rate of recorded data
    bool_mask : array_like
        boolean array with beats marked as outlier.
        Results of this outlier correction method will be combined with the array using a logical 'or'
    hr_thres : tuple
        tuple with lower and upper heart rate thresholds. Beats below and above these values will be marked as outlier.

    Returns
    -------
    array_like
        boolean array with beats marked as outlier. Logical 'or' combination of `bool_mask` and results from
        this algorithm
    """
    # physiological outlier: minimum/maximum heart rate threshold
    bool_mask = np.logical_or(
        bool_mask,
        (rpeaks["RR_Interval"] > (60 / hr_thres[0])) | (rpeaks["RR_Interval"] < (60 / hr_thres[1])),
    )
    return bool_mask


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


def check_ecg_input(
    ecg_processor: "EcgProcessor",
    key: str,
    ecg_signal: pd.DataFrame,
    rpeaks: pd.DataFrame,
) -> bool:
    """
    Checks valid input, i.e. if either `ecg_processor` **and** `key` are supplied as arguments *or* `ecg_signal` **and**
    `rpeaks`. Used as helper method for several functions.

    Parameters
    ----------
    ecg_processor : EcgProcessor
        `EcgProcessor` object. If this argument is passed, the `key` argument needs to be supplied as well
    key : str
        Dictionary key of the sub-phase to process. Needed when `ecg_processor` is passed as argument
    ecg_signal : str
        dataframe with ECG signal. Output of `EcgProcessor.ecg_process()`
    rpeaks : str
        dataframe with R peaks. Output of `EcgProcessor.ecg_process()`

    Returns
    -------
    ``True`` if correct input was supplied, raises ValueError otherwise

    Raises
    ------
    ValueError
        if invalid input supplied
    """

    if all([x is None for x in [ecg_processor, key, ecg_signal, rpeaks]]):
        raise ValueError("Either `ecg_processor` and `key` or `rpeaks` and `ecg_signal` must be passed as arguments!")
    if ecg_processor:
        if key is None:
            raise ValueError("`key` must be passed as argument when `ecg_processor` is passed!")

    return True
