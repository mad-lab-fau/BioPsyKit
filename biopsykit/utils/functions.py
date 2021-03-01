import numpy as np


def se(d):
    return np.std(d, ddof=1) / np.sqrt(len(d))
