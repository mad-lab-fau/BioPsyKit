from typing import ClassVar, get_args

import pandas as pd
from tpcp import Parameter, Pipeline
from tpcp._dataset import DatasetT
from typing_extensions import Self

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction, RPeakExtractionNeurokit
from biopsykit.signals.ecg.hrv_extraction import HrvExtraction
from biopsykit.signals.ecg.outlier_correction import (
    BaseRPeakOutlierDetection,
    RPeakOutlierCorrection,
    RPeakOutlierCorrectionHrvLipponen2019,
    RPeakOutlierDetectionBerntson1990,
    RPeakOutlierDetectionChain,
    RPeakOutlierDetectionCorrelation,
    RPeakOutlierDetectionPhysiological,
    RPeakOutlierDetectionQuality,
    RPeakOutlierDetectionRRDiffIntervalStatistics,
    RPeakOutlierDetectionRRIntervalStatistics,
)
from biopsykit.signals.ecg.outlier_correction._outlier_detection_chain import _validate_outlier_mask
from biopsykit.signals.ecg.preprocessing import BaseEcgPreprocessing, EcgPreprocessingNeurokit
from biopsykit.utils.dtypes import EcgRawDataFrame
from biopsykit.utils.exceptions import EcgProcessingError

__all__ = ["EcgProcessingPipeline"]


class EcgProcessingPipeline(Pipeline):
    """Process ECG using one outlier detector or an explicit named detector chain.

    ``outlier_detection_algo`` accepts a BaseRPeakOutlierDetection, including RPeakOutlierDetectionChain,
    or None to flag no outliers. Lists must be wrapped explicitly in a chain. Detector parameters remain
    recursively configurable through native tpcp ``get_params``/``set_params``. Correction is applied once
    after detection. The executed detector is retained in ``outlier_detection_algo_`` and its combined
    boolean mask in ``outlier_mask_``.

    ``ecg_clean_`` contains only preprocessing output; ``ecg_processed_`` contains the final ECG with
    extractor annotations and correction results. ``rpeaks_hrv_`` records the peaks supplied to HRV
    extraction, or the HRV correction output when only that step is enabled. Disabled HRV outputs are None.
    Configured algorithms remain unexecuted; corresponding attributes ending in ``_`` expose executed clones.
    The pipeline's ``handle_missing_events`` policy is forwarded to all supporting cloned steps.
    Results are replaced together on success and cleared before each run, including failed reruns.

    See :doc:`/guides/outlier_detection_chain` for detector configuration and
    :doc:`/guides/ecg_pipeline_results` for stage outputs and result lifecycle.
    """

    preprocessing_algo: Parameter[BaseEcgPreprocessing]
    r_peak_algo: Parameter[BaseEcgExtraction]
    outlier_detection_algo: Parameter[BaseRPeakOutlierDetection | None]
    outlier_correction_algo: Parameter[RPeakOutlierCorrection]
    hrv_r_peak_correction_algo: Parameter[RPeakOutlierCorrectionHrvLipponen2019 | None]
    hrv_extraction_algo: Parameter[HrvExtraction | None]

    handle_missing_events: Parameter[HANDLE_MISSING_EVENTS]

    _RESULT_ATTRS: ClassVar[tuple[str, ...]] = (
        "ecg_raw_",
        "ecg_clean_",
        "ecg_processed_",
        "rpeaks_raw_",
        "rpeaks_",
        "rpeaks_hrv_",
        "hrv_extracted_",
        "preprocessing_algo_",
        "r_peak_algo_",
        "outlier_detection_algo_",
        "outlier_mask_",
        "outlier_correction_algo_",
        "hrv_r_peak_correction_algo_",
        "hrv_extraction_algo_",
    )

    ecg_raw_: EcgRawDataFrame
    ecg_clean_: EcgRawDataFrame
    ecg_processed_: pd.DataFrame

    rpeaks_raw_: pd.DataFrame
    rpeaks_: pd.DataFrame
    rpeaks_hrv_: pd.DataFrame | None
    hrv_extracted_: pd.DataFrame | None
    outlier_detection_algo_: BaseRPeakOutlierDetection | None
    outlier_mask_: pd.DataFrame
    preprocessing_algo_: BaseEcgPreprocessing
    r_peak_algo_: BaseEcgExtraction
    outlier_correction_algo_: RPeakOutlierCorrection
    hrv_r_peak_correction_algo_: RPeakOutlierCorrectionHrvLipponen2019 | None
    hrv_extraction_algo_: HrvExtraction | None

    def __init__(
        self,
        *,
        preprocessing_algo: BaseEcgPreprocessing,
        r_peak_algo: BaseEcgExtraction,
        outlier_detection_algo: BaseRPeakOutlierDetection | None,
        outlier_correction_algo: RPeakOutlierCorrection,
        hrv_r_peak_correction_algo: RPeakOutlierCorrectionHrvLipponen2019 | None = None,
        hrv_extraction_algo: HrvExtraction | None = None,
        handle_missing_events: HANDLE_MISSING_EVENTS = "raise",
    ):
        self.preprocessing_algo = preprocessing_algo
        self.r_peak_algo = r_peak_algo
        self.outlier_detection_algo = outlier_detection_algo
        self.outlier_correction_algo = outlier_correction_algo
        self.hrv_r_peak_correction_algo = hrv_r_peak_correction_algo
        self.hrv_extraction_algo = hrv_extraction_algo

        self.handle_missing_events = handle_missing_events

    @classmethod
    def get_default_biopsykit_pipeline(
        cls,
        compute_hrv: bool = True,
        r_peak_imputation_type: str = "linear_interpolation",
        handle_missing_events: HANDLE_MISSING_EVENTS = "raise",
    ) -> Self:
        preprocessing_algo = EcgPreprocessingNeurokit(method="neurokit")
        r_peak_algo = RPeakExtractionNeurokit(handle_missing_events)
        outlier_detection_algo = RPeakOutlierDetectionChain(
            detectors=[
                ("berntson", RPeakOutlierDetectionBerntson1990()),
                ("physiological", RPeakOutlierDetectionPhysiological()),
                ("correlation", RPeakOutlierDetectionCorrelation()),
                ("quality", RPeakOutlierDetectionQuality()),
                ("rr_statistics", RPeakOutlierDetectionRRIntervalStatistics()),
                ("rr_diff_statistics", RPeakOutlierDetectionRRDiffIntervalStatistics()),
            ]
        )
        outlier_correction_algo = RPeakOutlierCorrection(imputation_type=r_peak_imputation_type)
        hrv_r_peak_correction_algo = None
        hrv_extraction_algo = None
        if compute_hrv:
            hrv_r_peak_correction_algo = RPeakOutlierCorrectionHrvLipponen2019()
            hrv_extraction_algo = HrvExtraction()

        return cls(
            preprocessing_algo=preprocessing_algo,
            r_peak_algo=r_peak_algo,
            outlier_detection_algo=outlier_detection_algo,
            outlier_correction_algo=outlier_correction_algo,
            hrv_r_peak_correction_algo=hrv_r_peak_correction_algo,
            hrv_extraction_algo=hrv_extraction_algo,
            handle_missing_events=handle_missing_events,
        )

    def run(self, datapoint: DatasetT) -> Self:
        """Run the pipeline on the given datapoint.

        The pipeline will first preprocess the ECG data, then extract the R-peaks, and finally apply outlier correction
        to the R-peaks. The results will be stored in the attributes of the class.

        Parameters
        ----------
        datapoint : DatasetT
            The dataset to process.

        Returns
        -------
        Self
            The instance of the class with the results set.
        """
        # A failed rerun must not expose outputs from the previous recording.
        for name in self._RESULT_ATTRS:
            self.__dict__.pop(name, None)
        if self.handle_missing_events not in get_args(HANDLE_MISSING_EVENTS):
            raise ValueError("handle_missing_events must be 'raise', 'warn', or 'ignore'.")
        if self.outlier_detection_algo is not None and not isinstance(
            self.outlier_detection_algo, BaseRPeakOutlierDetection
        ):
            raise TypeError(
                "outlier_detection_algo must be a detector, an explicit RPeakOutlierDetectionChain, or None."
            )
        sampling_rate_attrs = ["fs", "sampling_rate", "sampling_rate_hz", "sampling_rate_ecg"]
        sampling_rate = next((getattr(datapoint, att) for att in sampling_rate_attrs if hasattr(datapoint, att)), None)
        if sampling_rate is None:
            raise EcgProcessingError(
                f"No valid sampling rate attribute for the given datapoint found. Tried: {sampling_rate_attrs}."
            )

        # Preprocess the ECG data
        if not hasattr(datapoint, "ecg"):
            raise EcgProcessingError("The given datapoint does not have an 'ecg' attribute.")

        ecg = datapoint.ecg

        if len(ecg) == 0:
            raise EcgProcessingError("The given ECG data is empty.")

        ecg_raw = ecg.copy(deep=True)
        preprocessing_algo = self._clone_step(self.preprocessing_algo)
        r_peak_algo = self._clone_step(self.r_peak_algo)
        outlier_detection_algo = self._clone_step(self.outlier_detection_algo)
        outlier_correction_algo = self._clone_step(self.outlier_correction_algo)
        hrv_r_peak_correction_algo = self._clone_step(self.hrv_r_peak_correction_algo)
        hrv_extraction_algo = self._clone_step(self.hrv_extraction_algo)

        preprocessing_algo.clean(ecg=ecg_raw.copy(deep=True), sampling_rate_hz=sampling_rate)
        ecg_clean = preprocessing_algo.ecg_clean_.copy(deep=True)
        r_peak_algo.extract(ecg=ecg_clean.copy(deep=True), sampling_rate_hz=sampling_rate)
        rpeaks_raw = r_peak_algo.points_.copy(deep=True)
        # Some custom extractors only return points; NeuroKit also provides useful ECG annotations.
        ecg_with_events = getattr(r_peak_algo, "ecg_processed_", ecg_clean).copy(deep=True)

        outlier_mask = pd.DataFrame(False, index=rpeaks_raw.index, columns=["is_outlier"])
        if outlier_detection_algo is not None:
            outlier_detection_algo.detect_outlier(
                ecg=ecg_with_events.copy(deep=True), rpeaks=rpeaks_raw.copy(deep=True), sampling_rate_hz=sampling_rate
            )
            outlier_mask = _validate_outlier_mask(
                outlier_detection_algo.points_, rpeaks_raw.index, "outlier_detection_algo"
            ).to_frame("is_outlier")
        # Correct R-peak outliers
        outlier_correction_algo.correct_outlier(
            ecg=ecg_with_events.copy(deep=True),
            rpeaks=rpeaks_raw.copy(deep=True),
            outlier_detection_results=outlier_mask.copy(deep=True),
        )
        rpeaks = outlier_correction_algo.points_.copy(deep=True)
        ecg_processed = outlier_correction_algo.ecg_processed_.copy(deep=True)

        rpeaks_hrv, hrv_extracted = self._run_hrv(
            rpeaks, sampling_rate, hrv_r_peak_correction_algo, hrv_extraction_algo
        )

        # Publish one complete result set only after all enabled steps have succeeded.
        self.__dict__.update(
            ecg_raw_=ecg_raw,
            ecg_clean_=ecg_clean,
            ecg_processed_=ecg_processed,
            rpeaks_raw_=rpeaks_raw,
            rpeaks_=rpeaks,
            rpeaks_hrv_=rpeaks_hrv,
            hrv_extracted_=hrv_extracted,
            preprocessing_algo_=preprocessing_algo,
            r_peak_algo_=r_peak_algo,
            outlier_detection_algo_=outlier_detection_algo,
            outlier_mask_=outlier_mask,
            outlier_correction_algo_=outlier_correction_algo,
            hrv_r_peak_correction_algo_=hrv_r_peak_correction_algo,
            hrv_extraction_algo_=hrv_extraction_algo,
        )
        return self

    @staticmethod
    def _run_hrv(rpeaks, sampling_rate, correction, extraction):
        if correction is None and extraction is None:
            return None, None
        rpeaks_hrv = rpeaks.copy(deep=True)
        if correction is not None:
            correction.correct_outlier(rpeaks=rpeaks_hrv, sampling_rate_hz=sampling_rate)
            rpeaks_hrv = correction.points_.copy(deep=True)
        hrv_extracted = None
        if extraction is not None:
            extraction.extract(rpeaks=rpeaks_hrv.copy(deep=True), sampling_rate_hz=sampling_rate)
            hrv_extracted = extraction.hrv_extracted_.copy(deep=True)
        return rpeaks_hrv, hrv_extracted

    def _clone_step(self, algorithm):
        """Apply the pipeline policy only to cloned steps that expose a corresponding tpcp parameter."""
        if algorithm is None:
            return None
        executed = algorithm.clone()
        policies = {
            name: self.handle_missing_events
            for name in executed.get_params(deep=True)
            if name.split("__")[-1] == "handle_missing_events"
        }
        executed.set_params(**policies)
        return executed
