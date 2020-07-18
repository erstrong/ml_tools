import numpy as np


def accuracy(y_true, y_pred):
    """
    :param y_true: True values
    :param y_pred: Predicted values
    :return: Accuracy
    """
    return (np.array(y_pred) == y_true).sum() / y_true.shape[0]
