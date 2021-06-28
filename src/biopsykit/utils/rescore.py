import numpy as np


def rescore(predictions: np.array) -> np.array:
    """
    Application of Webster's rescoring rules as described in the Cole-Kripke paper.

    :param predictions: array of predictions
    :return: rescored predictions
    """
    rescored = predictions.copy()
    # rules a through c
    wake_bin = 0
    for t in range(len(rescored)):
        if rescored[t] == 0:
            wake_bin += 1
        else:
            if 15 <= wake_bin:
                # rule c: at least 15 minutes of wake, next 4 minutes of sleep get rescored
                rescored[t : t + 4] = 0
            elif 10 <= wake_bin < 15:
                # rule b: at least 10 minutes of wake, next 3 minutes of sleep get rescored
                rescored[t : t + 3] = 0
            elif 4 <= wake_bin < 10:
                # rule a: at least 4 minutes of wake, next 1 minute of sleep gets rescored
                rescored[t : t + 1] = 0
            wake_bin = 0

    # rule d/e: 6/10 minutes or less of sleep surrounded by at least 10/20 minutes of wake on each side get rescored
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
                if 0 < sleep_bin <= sleep_thres and sum1 == 0 and sum2 == 0:
                    rescored[start_ind:t] = 0
                sleep_bin = 0

    # # wake phases of 1 minute, surrounded by sleep, get rescored
    # for t in range(1, len(rescored) - 1):
    #     if rescored[t] == 1 and rescored[t - 1] == 0 and rescored[t + 1] == 0:
    #         rescored[t] = 1

    return rescored


