from typing import Literal

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

    See :doc:`/guides/outlier_detection_chain` for constructor examples, parameter paths, and migration.
    """

    preprocessing_algo: Parameter[BaseEcgPreprocessing]
    r_peak_algo: Parameter[BaseEcgExtraction]
    outlier_detection_algo: Parameter[BaseRPeakOutlierDetection | None]
    outlier_correction_algo: Parameter[RPeakOutlierCorrection]
    hrv_r_peak_correction_algo: Parameter[RPeakOutlierCorrectionHrvLipponen2019 | None]
    hrv_extraction_algo: Parameter[HrvExtraction | None]

    handle_missing_events: HANDLE_MISSING_EVENTS

    ecg_raw_: EcgRawDataFrame
    ecg_clean_: EcgRawDataFrame

    rpeaks_raw_: EcgRawDataFrame
    rpeaks_: EcgRawDataFrame
    hrv_extracted_: pd.DataFrame | None
    outlier_detection_algo_: BaseRPeakOutlierDetection | None
    outlier_mask_: pd.DataFrame

    def __init__(
        self,
        *,
        preprocessing_algo: BaseEcgPreprocessing,
        r_peak_algo: BaseEcgExtraction,
        outlier_detection_algo: BaseRPeakOutlierDetection | None,
        outlier_correction_algo: RPeakOutlierCorrection,
        hrv_r_peak_correction_algo: RPeakOutlierCorrectionHrvLipponen2019 | None = None,
        hrv_extraction_algo: HrvExtraction | None = None,
        handle_missing_events: Literal[HANDLE_MISSING_EVENTS] = "raise",
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
        preprocessing_algo = self.preprocessing_algo.clone()
        r_peak_algo = self.r_peak_algo.clone()
        if self.outlier_detection_algo is not None and not isinstance(
            self.outlier_detection_algo, BaseRPeakOutlierDetection
        ):
            raise TypeError(
                "outlier_detection_algo must be a detector, an explicit RPeakOutlierDetectionChain, or None."
            )
        outlier_detection_algo = (
            self.outlier_detection_algo.clone() if self.outlier_detection_algo is not None else None
        )
        outlier_correction_algo = self.outlier_correction_algo.clone()

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

        # Preprocess the ECG data
        preprocessing_algo.clean(ecg=ecg, sampling_rate_hz=sampling_rate)
        self.ecg_clean_ = preprocessing_algo.ecg_clean_

        # Extract the R-peaks
        r_peak_algo.extract(ecg=self.ecg_clean_, sampling_rate_hz=sampling_rate)
        self.rpeaks_raw_ = r_peak_algo.points_

        outlier_mask = pd.DataFrame(False, index=self.rpeaks_raw_.index, columns=["is_outlier"])
        if outlier_detection_algo is not None:
            outlier_detection_algo.detect_outlier(
                ecg=self.ecg_clean_, rpeaks=self.rpeaks_raw_, sampling_rate_hz=sampling_rate
            )
            outlier_mask = _validate_outlier_mask(
                outlier_detection_algo.points_, self.rpeaks_raw_.index, "outlier_detection_algo"
            ).to_frame("is_outlier")
        self.outlier_detection_algo_ = outlier_detection_algo
        self.outlier_mask_ = outlier_mask

        # Correct R-peak outliers
        outlier_correction_algo.correct_outlier(
            ecg=self.ecg_clean_,
            rpeaks=self.rpeaks_raw_,
            outlier_detection_results=outlier_mask,
        )
        self.rpeaks_ = outlier_correction_algo.points_
        self.ecg_clean_ = outlier_correction_algo.ecg_processed_

        r_peaks_hrv = self.rpeaks_.copy()
        if self.hrv_r_peak_correction_algo is not None:
            hrv_r_peak_correction_algo = self.hrv_r_peak_correction_algo.clone()
            hrv_r_peak_correction_algo.correct_outlier(
                rpeaks=r_peaks_hrv,
                sampling_rate_hz=sampling_rate,
            )
            r_peaks_hrv = hrv_r_peak_correction_algo.points_.copy()

        if self.hrv_extraction_algo is not None:
            hrv_extraction_algo = self.hrv_extraction_algo.clone()
            hrv_extraction_algo.extract(
                rpeaks=r_peaks_hrv,
                sampling_rate_hz=sampling_rate,
            )
            self.hrv_extracted_ = hrv_extraction_algo.hrv_extracted_.copy()

        return self
