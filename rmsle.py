__author__ = 'angad'
import numpy as np


def error(prediction, actual):
    n = len(prediction)
    prediction = np.array(prediction)
    actual = np.array(actual)
    return np.sum(np.square((np.log(prediction+1) - np.log(actual+1))))/n
