"""Utility functions for sleep/wake detection algorithms."""

from typing import Optional
from typing_extensions import Literal

import numpy as np

EPOCH_LENGTH = Literal[30, 60]


def rescore(predictions: np.ndarray, epoch_length: Optional[EPOCH_LENGTH] = 30) -> np.ndarray:
    """Apply Webster's rescoring rules to sleep/wake predictions.

    Parameters
    ----------
    predictions : array_like
        sleep/wake predictions
    epoch_length : int
        length of actigraphy epoch in seconds

    Returns
    -------
    array_like
        rescored sleep/wake predictions

    """
    rescored = predictions.copy()
    # rules a through c
    rescored = _apply_recording_rules_a_c(rescored, epoch_length)
    # rules d and e
    rescored = _apply_recording_rules_d_e(rescored, epoch_length)

    # wake phases of 1 minute, surrounded by sleep, get rescored
    for t in range(1, len(rescored) - 1):  # pylint:disable=consider-using-enumerate
        if rescored[t] == 1 and rescored[t - 1] == 0 and rescored[t + 1] == 0:
            rescored[t] = 0

    return rescored


def _apply_recording_rules_a_c(rescored: np.ndarray, epoch_length: EPOCH_LENGTH):  # pylint:disable=too-many-branches
    wake_bin = 0
    for t in range(len(rescored)):  # pylint:disable=consider-using-enumerate
        if rescored[t] == 1:
            wake_bin += 1
        else:
            if epoch_length == 30:
                if wake_bin >= 30:
                    # rule c: at least 15 minutes of wake, next 4 minutes of sleep get rescored
                    rescored[t : t + 8] = 0
                elif 20 <= wake_bin < 30:
                    # rule b: at least 10 minutes of wake, next 3 minutes of sleep get rescored
                    rescored[t : t + 6] = 0
                elif 8 <= wake_bin < 20:
                    # rule a: at least 4 minutes of wake, next 1 minute of sleep gets rescored
                    rescored[t : t + 2] = 0
                wake_bin = 0
            else:
                if wake_bin >= 15:
                    # rule c: at least 15 minutes of wake, next 4 minutes of sleep get rescored
                    rescored[t : t + 4] = 0
                elif 10 <= wake_bin < 15:
                    # rule b: at least 10 minutes of wake, next 3 minutes of sleep get rescored
                    rescored[t : t + 3] = 0
                elif 4 <= wake_bin < 10:
                    # rule a: at least 4 minutes of wake, next 1 minute of sleep gets rescored
                    rescored[t : t + 1] = 0
                wake_bin = 0

    return rescored


def _apply_recording_rules_d_e(rescored: np.ndarray, epoch_length: EPOCH_LENGTH):  # pylint:disable=too-many-branches
    # rule d/e: 6/10 minutes or less of sleep surrounded by at least 10/20 minutes of wake on each side get rescored
    if epoch_length == 30:
        sleep_rules = [12, 20]
        wake_rules = [20, 40]
    else:
        sleep_rules = [6, 10]
        wake_rules = [10, 20]

    for sleep_thres, wake_thres in zip(sleep_rules, wake_rules):
        sleep_bin = 0
        start_ind = 0
        for t in range(wake_thres, len(rescored) - wake_thres):
            if rescored[t] == 1:
                sleep_bin += 1
                if sleep_bin == 1:
                    start_ind = t
            else:
                sum1 = np.sum(rescored[start_ind - wake_thres : start_ind])
                sum2 = np.sum(rescored[t : t + wake_thres])
                if sleep_thres >= sleep_bin > 0 == sum1 and sum2 == 0:
                    rescored[start_ind:t] = 0
                sleep_bin = 0

    return rescored
