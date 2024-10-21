from typing import Optional

import pandas as pd
from biopsykit.signals._dtypes import assert_sample_columns_int
from biopsykit.signals.ecg.event_extraction._base_ecg_extraction import BaseEcgExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from tpcp import Parameter


class QPeakExtractionVanLien2013(BaseEcgExtraction):
    """Algorithm to extract Q-wave onset based on the detection of the R-peak, as suggested by Van Lien et al. (2013).

    The Q-wave onset is estimated by subtracting a fixed time interval from the R-peak location. The fixed time
    interval is defined by the parameter ``time_interval``.


    References
    ----------
    Van Lien, R., Schutte, N. M., Meijer, J. H., & De Geus, E. J. C. (2013). Estimated preejection period (PEP) based
    on the detection of the R-wave and dZ/dt-min peaks does not adequately reflect the actual PEP across a wide range
    of laboratory and ambulatory conditions. International Journal of Psychophysiology, 87(1), 60-69.
    https://doi.org/10.1016/j.ijpsycho.2012.11.001

    """

    # parameters
    time_interval_ms: Parameter[int]

    def __init__(self, time_interval_ms: Optional[int] = 40):
        """Initialize new QWaveOnsetExtractionVanLien algorithm instance.

        Parameters
        ----------
        time_interval_ms : int, optional
            Specify the constant time interval in milliseconds which will be subtracted from the R-peak for
            Q-wave onset estimation. Default: 40 ms
        """
        self.time_interval_ms = time_interval_ms

    # @make_action_safe
    def extract(
        self,
        *,
        ecg: Optional[pd.DataFrame],
        heartbeats: pd.DataFrame,
        sampling_rate_hz: int,
    ):
        """Extract Q-wave onset (start of ventricular depolarization) from given ECG cleaned signal.

        The results are saved in the ``points_`` attribute of the super class.

        Parameters
        ----------
        ecg: :class:`~pandas.DataFrame`
            ECG signal. Not used in this function since Q-wave onset is estimated from the R-peaks in the
            ``heartbeats`` DataFrame.
        heartbeats: :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
            location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        sampling_rate_hz: int
            Sampling rate of ECG signal in hz

        Returns
        -------
            self
        """
        # TODO handle missing?
        # convert the fixed time_interval from milliseconds into samples
        time_interval_in_samples = int((self.time_interval_ms / 1000) * sampling_rate_hz)

        # get the r_peaks from the heartbeats Dataframe
        r_peaks = heartbeats[["r_peak_sample"]]

        # subtract the fixed time_interval from the r_peak samples to estimate the q_wave_onset

        q_wave_onset = r_peaks - time_interval_in_samples

        q_wave_onset.columns = ["q_wave_onset_sample"]
        q_wave_onset = q_wave_onset.assign(nan_reason=pd.NA)
        q_wave_onset = q_wave_onset.convert_dtypes(infer_objects=True)

        _assert_is_dtype(q_wave_onset, pd.DataFrame)
        _assert_has_columns(q_wave_onset, [["q_wave_onset_sample", "nan_reason"]])

        assert_sample_columns_int(q_wave_onset)

        self.points_ = q_wave_onset
        return self
