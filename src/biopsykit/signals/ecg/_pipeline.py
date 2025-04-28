from collections.abc import Sequence
from typing import Literal

from tpcp import Pipeline
from tpcp._dataset import DatasetT
from typing_extensions import Self

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.ecg.event_extraction import BaseEcgExtraction, RPeakExtractionNeurokit
from biopsykit.signals.ecg.outlier_correction import (
    BaseRPeakOutlierDetection,
    RPeakOutlierCorrection,
    RPeakOutlierDetectionBerntson1990,
    RPeakOutlierDetectionCorrelation,
    RPeakOutlierDetectionPhysiological,
    RPeakOutlierDetectionQuality,
    RPeakOutlierDetectionRRDiffIntervalStatistics,
    RPeakOutlierDetectionRRIntervalStatistics,
)
from biopsykit.signals.ecg.preprocessing import BaseEcgPreprocessing, EcgPreprocessingNeurokit
from biopsykit.utils.dtypes import EcgRawDataFrame

__all__ = ["EcgProcessingPipeline", "EcgProcessingPipelineBiopsykit"]


class EcgProcessingPipeline(Pipeline):

    preprocessing_algo: BaseEcgPreprocessing
    r_peak_algo: BaseEcgExtraction
    outlier_detection_algos: Sequence[BaseRPeakOutlierDetection]
    outlier_correction_algo: RPeakOutlierCorrection

    handle_missing_events: HANDLE_MISSING_EVENTS

    ecg_raw_: EcgRawDataFrame
    ecg_clean_: EcgRawDataFrame

    rpeaks_raw_: EcgRawDataFrame
    rpeaks_: EcgRawDataFrame

    def __init__(
        self,
        *,
        preprocessing_algo: BaseEcgPreprocessing,
        r_peak_algo: BaseEcgExtraction,
        outlier_detection_algos: Sequence[BaseRPeakOutlierDetection],
        outlier_correction_algo: RPeakOutlierCorrection,
        handle_missing_events: Literal[HANDLE_MISSING_EVENTS] = "warn",
    ):
        self.preprocessing_algo = preprocessing_algo
        self.r_peak_algo = r_peak_algo
        self.outlier_detection_algos = outlier_detection_algos
        self.outlier_correction_algo = outlier_correction_algo

        self.handle_missing_events = handle_missing_events

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
        preprocessing_algo = self.preprocessing_algo.clone()
        r_peak_algo = self.r_peak_algo.clone()
        outlier_detection_algos = [algo.clone() for algo in self.outlier_detection_algos]
        outlier_correction_algo = self.outlier_correction_algo.clone()

        sampling_rate_attrs = ["fs", "sampling_rate", "sampling_rate_hz", "sampling_rate_ecg"]
        sampling_rate = next((getattr(datapoint, att) for att in sampling_rate_attrs if hasattr(datapoint, att)), None)
        if sampling_rate is None:
            raise ValueError(
                f"No valid sampling rate attribute for the given datapoint found. Tried: {sampling_rate_attrs}."
            )

        # Preprocess the ECG data
        if not hasattr(datapoint, "ecg"):
            raise ValueError("The given datapoint does not have an 'ecg' attribute.")

        ecg = datapoint.ecg

        # Preprocess the ECG data
        preprocessing_algo.clean(ecg=ecg, sampling_rate_hz=sampling_rate)
        self.ecg_clean_ = preprocessing_algo.ecg_clean_

        # Extract the R-peaks
        r_peak_algo.extract(ecg=self.ecg_clean_, sampling_rate_hz=sampling_rate)
        self.rpeaks_raw_ = r_peak_algo.points_

        outlier_masks = []
        # Apply outlier detection and correction
        for algo in outlier_detection_algos:
            algo.detect_outlier(ecg=self.ecg_clean_, rpeaks=self.rpeaks_raw_, sampling_rate_hz=sampling_rate)
            # print(f"Outlier detection algorithm: {algo.__class__.__name__}")
            # print(f"Outlier detection results: {algo.points_}")
            outlier_masks.append(algo.points_)

        # Correct R-peak outliers
        outlier_correction_algo.correct_outlier(
            ecg=self.ecg_clean_,
            rpeaks=self.rpeaks_raw_,
            outlier_detection_results=outlier_masks,
        )
        self.rpeaks_ = outlier_correction_algo.points_
        self.ecg_clean_ = outlier_correction_algo.ecg_processed_


class EcgProcessingPipelineBiopsykit(EcgProcessingPipeline):

    r_peak_imputation_type: str

    def __init__(
        self,
        *,
        r_peak_imputation_type: str = "linear_interpolation",
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ):
        self.r_peak_imputation_type = r_peak_imputation_type
        preprocessing_algo = EcgPreprocessingNeurokit(method="neurokit")
        r_peak_algo = RPeakExtractionNeurokit(handle_missing_events)
        outlier_detection_algos = [
            RPeakOutlierDetectionBerntson1990(),
            RPeakOutlierDetectionPhysiological(),
            RPeakOutlierDetectionCorrelation(),
            RPeakOutlierDetectionQuality(),
            RPeakOutlierDetectionRRIntervalStatistics(),
            RPeakOutlierDetectionRRDiffIntervalStatistics(),
        ]
        outlier_correction_algo = RPeakOutlierCorrection(imputation_type=r_peak_imputation_type)

        super().__init__(
            preprocessing_algo=preprocessing_algo,
            r_peak_algo=r_peak_algo,
            outlier_detection_algos=outlier_detection_algos,
            outlier_correction_algo=outlier_correction_algo,
            handle_missing_events=handle_missing_events,
        )
