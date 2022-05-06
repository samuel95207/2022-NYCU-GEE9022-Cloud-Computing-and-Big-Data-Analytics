import numpy as np

def accuracy(y, y_pred):
    return np.mean(y == y_pred)

