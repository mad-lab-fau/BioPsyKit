import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import HANDLE_MISSING_EVENTS
from biopsykit.signals.icg.event_extraction._base_b_point_extraction import BaseBPointExtraction
from biopsykit.utils._datatype_validation_helper import _assert_has_columns, _assert_is_dtype
from biopsykit.utils.exceptions import EventExtractionError
from tpcp import Parameter

__all__ = ["BPointExtractionArbol2017"]


class BPointExtractionArbol2017(BaseBPointExtraction):
    """algorithm to extract B-point from cleaned ICG dZ/dt signal using the third derivative (see Arbol 2017)."""

    # input parameters
    window_b_detection_ms: Parameter[Union[str, int]]  # either 'R' or integer defining window length in ms
    save_icg_derivatives: Parameter[bool]
    correct_outliers: Parameter[bool]

    # results
    icg_derivatives_: Dict[int, pd.DataFrame]

    def __init__(
        self,
        window_b_detection_ms: Optional[Union[str, int]] = 150,
        save_icg_derivatives: Optional[bool] = False,
        correct_outliers: Optional[bool] = False,
    ):
        """Initialize new BPointExtraction_ThirdDeriv algorithm instance.

        Args:
            window_b_detection_ms : str, int
                defines the window in which the algorithm searches for the B-point,
                'R' -> search B-point in the region between R-peak and C-point,
                int -> search B-point in the region between xx ms before C-point and C-point
                (150 ms -> see Arbol 2017, procedure for visual detection; Lababidi 1970)
                (300 ms -> see Arbol 2017, 3rd derivative-based algorithm)
            save_icg_derivatives : bool
                when True 2nd and 3rd derivative of ICG signal are saved in icg_derivatives_, otherwise not
        """
        self.window_b_detection_ms = window_b_detection_ms
        self.save_icg_derivatives = save_icg_derivatives
        self.correct_outliers = correct_outliers

    # @make_action_safe
    def extract(  # noqa: C901, PLR0915
        self,
        *,
        icg: pd.Series,
        heartbeats: pd.DataFrame,
        c_points: pd.DataFrame,
        sampling_rate_hz: int,
        handle_missing: Optional[HANDLE_MISSING_EVENTS] = "warn",
    ):
        """Extract B-points from given cleaned ICG derivative signal.

        This algorithm extracts B-points based on the maximum of the third derivative of the ICG signal.

        The results are saved in the points_ attribute of the super class.

        Parameters
        ----------
        icg : :class:`~pandas.Series`
            cleaned ICG derivative signal
        heartbeats : :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
            location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
        c_points : :class:`~pandas.DataFrame`
            DataFrame containing one row per segmented C-point, each row contains location
            (in samples from beginning of signal) of that C-point or NaN if the location of that C-point
            is not correct
        sampling_rate_hz : int
            sampling rate of ICG derivative signal in hz
        handle_missing : one of {"warn", "raise", "ignore"}, optional
            How to handle missing data in the input dataframes. Default: "warn"

        Returns
        -------
            self

        Raises
        ------
        :exc:`~biopsykit.utils.exceptions.EventExtractionError`
            If the C-Point contains NaN values and handle_missing is set to "raise"

        """
        # result dfs
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point_sample", "nan_reason"])
        icg_derivatives = {}
        if self.save_icg_derivatives:
            icg_derivatives = {
                k: pd.DataFrame(columns=["icg_der", "icg_2nd_der", "icg_3rd_der"]) for k in heartbeats.index
            }

        # used subsequently to store ids of heartbeats where no B was detected because there was no C
        # (Bs should always be found, since they are set to the max of the 3rd derivative, and there is always a max)
        heartbeats_no_c_b = []
        # (but in case of wrongly detected Cs, the search window might be invalid, then no B can be found)
        heartbeats_no_b = []

        icg = icg.squeeze()

        # search B-point for each heartbeat of the given signal
        for idx, data in heartbeats.iterrows():
            # slice signal for current heartbeat
            heartbeat_start = data["start_sample"]
            heartbeat_end = data["end_sample"]
            heartbeat_icg_der = icg.iloc[heartbeat_start:heartbeat_end]

            # calculate derivatives for this heartbeat
            heartbeat_icg_2nd_der = np.gradient(heartbeat_icg_der)
            heartbeat_icg_3rd_der = np.gradient(heartbeat_icg_2nd_der)

            if self.save_icg_derivatives:
                icg_derivatives[idx]["icg_der"] = heartbeat_icg_der
                icg_derivatives[idx]["icg_2nd_der"] = heartbeat_icg_2nd_der
                icg_derivatives[idx]["icg_3rd_der"] = heartbeat_icg_3rd_der
                self.icg_derivatives_ = icg_derivatives

            # calculate R-peak and C-point position relative to start of current heartbeat
            heartbeat_r_peak = data["r_peak_sample"] - heartbeat_start
            heartbeat_c_point = c_points["c_point_sample"].iloc[idx] - heartbeat_start

            # C-point can be NaN, then, extraction of B is not possible, so B is set to NaN
            if pd.isna(heartbeat_c_point):
                heartbeats_no_c_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.NaN
                b_points.loc[idx, "nan_reason"] = "no_c_point"
                continue

            # set window end to C-point position and set window start according to specified method
            window_end = heartbeat_c_point
            if self.window_b_detection_ms == "R":
                window_start = heartbeat_r_peak
            elif isinstance(self.window_b_detection_ms, int):
                window_length_samples = int((self.window_b_detection_ms / 1000) * sampling_rate_hz)
                window_start = heartbeat_c_point - window_length_samples
            else:
                raise ValueError("That should never happen!")

            # might happen for wrongly detected Cs (search window becomes invalid)
            if window_start < 0 or window_end < 0:
                heartbeats_no_b.append(idx)
                b_points.loc[idx, "b_point_sample"] = np.NaN
                b_points.loc[idx, "nan_reason"] = "invalid_b_point_search_window"
                continue

            # find max in B window and calculate B-point relative to signal start
            heartbeat_b_window = heartbeat_icg_3rd_der[window_start:window_end]
            b_window_max = np.argmax(heartbeat_b_window)
            b_point = b_window_max + window_start + heartbeat_start

            """
            if not self.correct_outliers:
                if b_point < data['r_peak_sample']:
                    b_points['b_point'].iloc[idx] = np.NaN
                    #warnings.warn(f"The detected B-point is located before the R-Peak at heartbeat {idx}!"
                    #              f" The B-point was set to NaN.")
                else:
                    b_points['b_point'].iloc[idx] = b_point
            else:
                b_points['b_point'].iloc[idx] = b_point
            """
            b_points["b_point_sample"].iloc[idx] = b_point

        # inform user about missing B-points
        if len(heartbeats_no_c_b) > 0 or len(heartbeats_no_b) > 0:
            nan_rows = b_points[b_points["b_point_sample"].isna()]
            n = len(nan_rows)
            nan_rows = nan_rows.drop(index=heartbeats_no_c_b)
            nan_rows = nan_rows.drop(index=heartbeats_no_b)

            missing_str = (
                f"No B-point detected in {n} heartbeats:\n"
                f"- For heartbeats {heartbeats_no_c_b} no B point could be extracted, "
                f"because there was no C point\n"
                f"- For heartbeats {heartbeats_no_b} the search window was invalid probably due to "
                f"wrongly detected C points\n"
                f"- for heartbeats {nan_rows.index.to_numpy()} apparently also no B point was found "
                f"for some other reasons"
            )

            if handle_missing == "warn":
                warnings.warn(missing_str)
            elif handle_missing == "raise":
                raise EventExtractionError(missing_str)

        _assert_is_dtype(b_points, pd.DataFrame)
        _assert_has_columns(b_points, [["b_point_sample", "nan_reason"]])

        self.points_ = b_points.convert_dtypes(infer_objects=True)
        return self