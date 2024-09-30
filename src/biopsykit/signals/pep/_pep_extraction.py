import pandas as pd
from tpcp import Algorithm
from typing_extensions import Self

__all__ = ["PepExtraction"]


class PepExtraction(Algorithm):

    _action_methods = "extract"

    pep_results_: pd.DataFrame

    def extract(
        self,
        *,
        heartbeats: pd.DataFrame,
        q_wave_onset_samples: pd.DataFrame,
        b_point_samples: pd.DataFrame,
        sampling_rate_hz: int,
    ) -> Self:
        """Compute PEP from Q-wave onset and B-point locations.

        Args:
            heartbeats:
                Heartbeat locations as DataFrame
            q_wave_onset_samples:
                ECG signal as DataFrame
            b_point_samples:
                ICG signal as DataFrame
            sampling_rate_hz:
                Sampling rate of the signals in Hz

        Returns
        -------
            self
        """
        # do something
        pep_results = pd.DataFrame(index=heartbeats.index)

        pep_results = pep_results.assign(
            heartbeat_start_time=heartbeats["start_time"],
            heartbeat_start_sample=pd.to_numeric(heartbeats["start_sample"]),
            heartbeat_end_sample=pd.to_numeric(heartbeats["end_sample"]),
            r_peak_sample=pd.to_numeric(heartbeats["r_peak_sample"]),
            rr_interval_sample=pd.to_numeric(heartbeats["rr_interval_sample"]),
            rr_interval_ms=pd.to_numeric(heartbeats["rr_interval_sample"] / sampling_rate_hz * 1000),
            q_wave_onset_sample=pd.to_numeric(q_wave_onset_samples["q_wave_onset_sample"]),
            b_point_sample=pd.to_numeric(b_point_samples["b_point_sample"]),
            pep_sample=pd.to_numeric(b_point_samples["b_point_sample"] - q_wave_onset_samples["q_wave_onset_sample"]),
            nan_reason=q_wave_onset_samples["nan_reason"],
        )

        pep_results = pep_results.assign(
            pep_ms=pep_results["pep_sample"] / sampling_rate_hz * 1000,
        )
        pep_results = pep_results.convert_dtypes(infer_objects=True)

        self.pep_results_ = pep_results

        return self
