__author__ = 'angad'
import numpy as np


def scale(x):
    x = np.array(x)
    return (x - x.mean())/x.std()