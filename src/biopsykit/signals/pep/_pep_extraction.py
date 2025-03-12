import warnings
from typing import Literal

import pandas as pd
from typing_extensions import Self

__all__ = ["PepExtraction"]

from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS, BaseExtraction, CanHandleMissingEventsMixin
from biopsykit.utils.dtypes import (
    BPointDataFrame,
    HeartbeatSegmentationDataFrame,
    QPeakDataFrame,
    is_b_point_dataframe,
    is_heartbeat_segmentation_dataframe,
    is_pep_result_dataframe,
    is_q_peak_dataframe,
)
from biopsykit.utils.exceptions import EventExtractionError

NEGATIVE_PEP_HANDLING = Literal["nan", "zero", "keep"]


class PepExtraction(BaseExtraction, CanHandleMissingEventsMixin):
    _action_methods = "extract"

    handle_negative_pep: NEGATIVE_PEP_HANDLING
    pep_results_: pd.DataFrame

    def __init__(
        self,
        handle_negative_pep: NEGATIVE_PEP_HANDLING = "nan",
        handle_missing_events: HANDLE_MISSING_EVENTS = "warn",
    ) -> None:
        self.handle_negative_pep = handle_negative_pep
        super().__init__(handle_missing_events=handle_missing_events)

    def extract(
        self,
        *,
        heartbeats: HeartbeatSegmentationDataFrame,
        q_peak_samples: QPeakDataFrame,
        b_point_samples: BPointDataFrame,
        sampling_rate_hz: float,
    ) -> Self:
        """Compute PEP from Q-peak samples and B-point locations.

        Args:
            heartbeats:
                Heartbeat locations as DataFrame
            q_peak_samples:
                ECG signal as DataFrame
            b_point_samples:
                ICG signal as DataFrame
            sampling_rate_hz:
                Sampling rate of the signals in Hz

        Returns
        -------
            self
        """
        is_heartbeat_segmentation_dataframe(heartbeats)
        is_b_point_dataframe(b_point_samples)
        is_q_peak_dataframe(q_peak_samples)
        self._check_valid_missing_handling()
        self._check_valid_handle_pep()

        # do something
        pep_results = pd.DataFrame(
            index=heartbeats.index,
            columns=[
                "heartbeat_start_sample",
                "heartbeat_end_sample",
                "r_peak_sample",
                "rr_interval_sample",
                "rr_interval_ms",
                "heart_rate_bpm",
                "q_peak_sample",
                "b_point_sample",
                "pep_sample",
                "pep_ms",
                "nan_reason",
            ],
        )

        if heartbeats.empty:
            missing_str = "No heartbeats found, no PEP can be extracted!"
            self.pep_results_ = pep_results
            if self.handle_missing_events == "warn":
                warnings.warn(missing_str)
            elif self.handle_missing_events == "raise":
                raise EventExtractionError(missing_str)
            return self

        pep_results = pep_results.assign(
            heartbeat_start_time=heartbeats["start_time"],
            heartbeat_start_sample=pd.to_numeric(heartbeats["start_sample"]),
            heartbeat_end_sample=pd.to_numeric(heartbeats["end_sample"]),
            r_peak_sample=pd.to_numeric(heartbeats["r_peak_sample"]),
            rr_interval_sample=pd.to_numeric(heartbeats["rr_interval_sample"]),
            rr_interval_ms=pd.to_numeric(heartbeats["rr_interval_sample"] / sampling_rate_hz * 1000),
            heart_rate_bpm=pd.to_numeric(60 / (heartbeats["rr_interval_sample"] / sampling_rate_hz)),
            q_peak_sample=pd.to_numeric(q_peak_samples["q_peak_sample"]),
            b_point_sample=pd.to_numeric(b_point_samples["b_point_sample"]),
            pep_sample=pd.to_numeric(b_point_samples["b_point_sample"] - q_peak_samples["q_peak_sample"]),
        )

        pep_results = pep_results.assign(
            pep_ms=pep_results["pep_sample"] / sampling_rate_hz * 1000,
        )
        pep_results = self._add_invalid_pep_reason(pep_results, q_peak_samples, b_point_samples)
        pep_results = pep_results.astype(
            {
                "heartbeat_start_sample": "Int64",
                "heartbeat_end_sample": "Int64",
                "r_peak_sample": "Int64",
                "rr_interval_sample": "Int64",
                "rr_interval_ms": "Float64",
                "heart_rate_bpm": "Float64",
                "q_peak_sample": "Int64",
                "b_point_sample": "Int64",
                "pep_sample": "Int64",
                "pep_ms": "Float64",
                "nan_reason": "object",
            }
        )

        is_pep_result_dataframe(pep_results)

        self.pep_results_ = pep_results

        return self

    def _add_invalid_pep_reason(
        self,
        pep_results: pd.DataFrame,
        q_peaks: QPeakDataFrame,
        b_points: BPointDataFrame,
    ) -> pd.DataFrame:
        # extract nan_reason from q_peak_samples and add to pep_results
        pep_results = pep_results.assign(nan_reason=q_peaks["nan_reason"])
        # TODO add option to store multiple nan_reasons in one column?
        # extract nan_reason from b_point_samples
        nan_reason_b_point = b_points["nan_reason"].loc[~b_points["nan_reason"].isna()]
        # add nan_reason to pep_results
        if not nan_reason_b_point.empty:
            pep_results.loc[nan_reason_b_point.index, "nan_reason"] = nan_reason_b_point

        neg_pep_idx = pep_results["pep_ms"] < 0
        if self.handle_negative_pep == "zero":
            pep_results.loc[neg_pep_idx, ["pep_sample", "pep_ms"]] = 0
            pep_results.loc[neg_pep_idx, "nan_reason"] = "negative_pep"
        elif self.handle_negative_pep == "nan":
            pep_results.loc[neg_pep_idx, ["pep_sample", "pep_ms"]] = pd.NA
            pep_results.loc[neg_pep_idx, "nan_reason"] = "negative_pep"

        return pep_results

    def _check_valid_handle_pep(self):
        if self.handle_negative_pep not in ["nan", "zero", "keep"]:
            raise ValueError(f"Invalid value for 'handle_negative_pep': {self.handle_negative_pep}")
