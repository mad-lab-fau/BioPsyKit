import inspect

import numpy as np
import pandas as pd

from biopsykit.signals.ecg.aggregation import HeartRateResampling


class TestHeartRateResampling:
    def test_resampling_uses_configured_resample_rate(self):
        result = (
            HeartRateResampling(group_level=["subject", "condition"], resample_rate_hz=2.0, cut_to_shortest=False)
            .apply(self._heart_rate_data())
            .output_
        )

        time_sec = result.index.get_level_values("time_sec").to_numpy()
        np.testing.assert_array_equal(np.diff(time_sec), np.full(len(time_sec) - 1, 0.5))

    def test_default_resampling_uses_one_hz_rate(self):
        result = (
            HeartRateResampling(group_level=["subject", "condition"], cut_to_shortest=False)
            .apply(self._heart_rate_data())
            .output_
        )

        time_sec = result.index.get_level_values("time_sec").to_numpy()
        np.testing.assert_array_equal(np.diff(time_sec), np.ones(len(time_sec) - 1))

    def test_resample_rate_hz_is_a_parameter(self):
        assert "resample_rate_hz" in inspect.signature(HeartRateResampling).parameters

    @staticmethod
    def _heart_rate_data():
        index = pd.MultiIndex.from_product(
            [["Vp01"], ["Baseline"], range(4)], names=["subject", "condition", "r_peak_id"]
        )
        return pd.DataFrame(
            {
                "r_peak_time": pd.to_timedelta([0.0, 0.8, 2.2, 3.2], unit="s"),
                "heart_rate_bpm": [60.0, 70.0, 80.0, 90.0],
            },
            index=index,
        )
