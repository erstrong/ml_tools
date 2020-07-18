import numpy as np


#############
# Distributions
#############
def sigmoid(val):
    """
    :param val: Logit values
    :return: Probabilities
    """
    return 1 / (1 + np.exp(-val))


def softmax(logits):
    """
    :param logits: Logit values
    :return: Probabilities for each class
    """
    exp = np.exp(logits)
    logit_sum = np.sum(exp, axis=1)
    probabilities = exp / logit_sum.reshape(-1, 1)
    return probabilities


#############
# Loss Functions
#############
def binary_cross_entropy(x, y, y_hat):
    """
    :param x: Training features
    :param y: Training target
    :param y_hat: Predicted probabilities
    :return: Loss value
    """
    one_cost = np.dot(y.T, np.log(y_hat))
    zero_cost = np.dot((1 - y).T, np.log(1 - y_hat))
    return (-1 / x.shape[0]) * (one_cost + zero_cost)[0][0]


def categorical_cross_entropy(y_hat, y):
    """
    :param y: Training target
    :param y_hat: Predicted probabilities
    :return: Cost value
    """
    ones_cost = -np.log(y_hat[np.arange(y_hat.shape[0]), y])
    return np.sum(ones_cost) / y_hat.shape[0]
