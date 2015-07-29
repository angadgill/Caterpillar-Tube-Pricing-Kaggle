__author__ = 'angad'
import numpy as np


def error(prediction, actual):
    n = len(prediction)
    prediction = np.array(prediction)
    actual = np.array(actual)
    return np.sqrt(np.sum(np.square((np.log1p(prediction) - np.log1p(actual))))/n)
