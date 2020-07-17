import numpy as np
from ml_tools.base import BaseModel


class BinaryLogisticRegression(BaseModel):

    def __init__(self, learning_rate=1e-9, n_iters=1000, threshold=0.5, verbose=False):
        """
        :param learning_rate: Rate at which the model learns from errors
        :param n_iters: Number of times weights will be updated
        :param threshold: Threshold for prediction
        :param verbose: Flag to print cost during training after each 100 iterations
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.threshold = threshold
        self.verbose = verbose

        self.beta = None
        self.cost = None

    def fit(self, x, y):
        """
        :param x: Features for training
        :param y: Target for training
        """
        self.beta = np.zeros((x.shape[1], 1))
        cost = 0

        for i in range(0, self.n_iters):
            # Calculate the sigmoid of the dot product of x and theta
            y_hat = self.predict_proba(x)

            # Calculate the loss function
            cost = self.cost_function(x, y, y_hat)

            if self.verbose:
                if i % 100 == 0:
                    print(cost)

            # Update the weights
            self.update_weights(x, y, y_hat)

        self.cost = float(cost)

    def predict(self, x):
        """
        Predicts probabilities for each row and evaluates against threshold
        :param x: Features for prediction
        :return: Predicted value for each row
        """
        return 1 * (self.predict_proba(x) >= self.threshold)

    def predict_proba(self, x):
        """
        :param x: Features for prediction
        :return: Probability for each row
        """
        return self.sigmoid(np.dot(x, self.beta))

    def sigmoid(self, val):
        """
        :param val: Logit values
        :return: Probabilities
        """
        return 1 / (1 + np.exp(-val))

    def cost_function(self, x, y, y_hat):
        """
        :param x: Training features
        :param y: Training target
        :param y_hat: Predicted probabilities
        :return: Cost value
        """
        one_cost = np.dot(y.T, np.log(y_hat))
        zero_cost = np.dot((1 - y).T, np.log(1 - y_hat))
        return (-1 / x.shape[0]) * (one_cost + zero_cost)[0][0]

    def update_weights(self, x, y, y_hat):
        """
        :param x: Training features
        :param y: Training target
        :param y_hat: Predicted probabilities
        """
        error_Xy = np.dot(x.T, (y_hat - y))
        self.beta = self.beta - (self.learning_rate / x.shape[0] * error_Xy)
