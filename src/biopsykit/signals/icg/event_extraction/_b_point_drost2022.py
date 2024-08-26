import warnings
from typing import Optional

import numpy as np
import pandas as pd
from biopsykit.signals._base_extraction import BaseExtraction
from tpcp import Parameter, make_action_safe


class BPointExtractionDrost2022(BaseExtraction):
    """algorithm to extract B-point based on the maximum distance of the dZ/dt curve and a straight line
    fitted between the C-Point and the Point on the dZ/dt curve 150 ms before the C-Point.
    """

    # input parameters
    correct_outliers: Parameter[bool]

    def __init__(self, correct_outliers: Optional[bool] = False):
        """Initialize new BPointExtractionDrost algorithm instance.

        Parameters
        ----------
        correct_outliers : bool
            Indicates whether to perform outlier correction (True) or not (False)
        """
        self.correct_outliers = correct_outliers

    @make_action_safe
    def extract(
        self, signal_clean: pd.DataFrame, heartbeats: pd.DataFrame, c_points: pd.DataFrame, sampling_rate_hz: int
    ):
        """Function which extracts B-points from given ICG cleaned signal.

        Args:
            signal_clean:
                cleaned ICG signal
            heartbeats:
                pd.DataFrame containing one row per segmented heartbeat, each row contains start, end, and R-peak
                location (in samples from beginning of signal) of that heartbeat, index functions as id of heartbeat
            c_points:
                pd.DataFrame containing one row per segmented C-point, each row contains location
                (in samples from beginning of signal) of that C-point or NaN if the location of that C-point
                is not correct
            sampling_rate_hz:
                sampling rate of ECG signal in hz

        Returns
        -------
            saves resulting B-point locations (samples) in points_ attribute of super class,
            index is C-point (/heartbeat) id
        """
        # Create the b_point Dataframe. Use the heartbeats id as index
        b_points = pd.DataFrame(index=heartbeats.index, columns=["b_point"])

        # get the c_point locations from the c_points dataframe and search for entries containing NaN
        check_c_points = np.isnan(c_points.values.astype(float))

        # iterate over each heartbeat
        for idx, _data in heartbeats.iterrows():
            # check if c_points contain NaN. If this is the case, set the b_point to NaN and continue
            # with the next iteration
            if check_c_points[idx]:
                b_points["b_point"].iloc[idx] = np.NaN
                warnings.warn(f"The C-Point contains NaN at heartbeat {idx}! The index of the B-Point was set to NaN.")
                continue
            else:
                # Get the C-Point location at the current heartbeat id
                c_point = c_points["c_point"].iloc[idx]

            # Calculate the start position of the straight line (150 ms before the C-Point)
            line_start = c_point - int((150 / 1000) * sampling_rate_hz)

            # Calculate the values of the straight line
            line_values = self.get_line_values(line_start, signal_clean[line_start], c_point, signal_clean[c_point])

            # Get the interval of the cleaned ICG-signal in the range of the straight line
            signal_clean_interval = signal_clean[line_start:c_point]

            # Calculate the distance between the straight line and the cleaned ICG-signal
            distance = line_values["result"].values - signal_clean_interval.values

            # Calculate the location of the maximum distance and transform the index relative to the complete signal
            # to obtain the B-Point location
            b_point = distance.argmax() + line_start

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

            b_points["b_point"].iloc[idx] = b_point

        points = b_points
        self.points_ = points
        return self

    @staticmethod
    def get_line_values(start_x: int, start_y: float, c_x: int, c_y: float):
        """Function which computes the values of a straight line fitted between the C-Point and the Point 150 ms before
        the C-Point.

        Args:
            start_x:
                int index of the Point 150 ms before the C-Point
            start_y:
                float value of the Point 150 ms before the C-Point
            c_x:
                int index of the C-Point
            c_y:
                float value of the C-Point

        Returns
        -------
            pd.DataFrame containing the values of the straight line for each index between the C-Point and the Point
            150 ms before the C-Point
        """
        # Compute the slope of the straight line
        slope = (c_y - start_y) / (c_x - start_x)

        # Get the sample positions where we want to calculate the values of the straight line
        index = np.arange(0, (c_x - start_x), 1)
        line_values = pd.DataFrame(index, columns=["index"])

        # Compute the values of the straight line for each index
        line_values["result"] = (line_values["index"] * slope) + start_y

        return line_values
