import numpy as np
from ml_tools.base import BaseModel
from ml_tools.stats import *


class BinaryLogisticRegression(BaseModel):

    def __init__(self, learning_rate=1e-9, n_iters=1000, init=None, threshold=0.5, verbose=False, seed=12345):
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
        self.init = init
        self.seed = seed

        self.weights = None
        self.loss_list = []

    def fit(self, x, y):
        """
        :param x: Features for training
        :param y: Target for training
        """
        if self.init=='random':
            np.random.seed(self.seed)
            self.weights = np.random.rand(x.shape[1], 1)
        else:
            self.weights = np.zeros((x.shape[1], 1))

        for i in range(0, self.n_iters):
            self.partial_fit(x, y)

            if self.verbose:
                if i % 100 == 0:
                    print(self.loss_list[-1])

    def partial_fit(self, x, y):
        """
        Performs a single iteration update of weights
        :param x: Features for training
        :param y: Target for training
        """
        # Calculate the sigmoid of the dot product of x and theta
        y_hat = self.predict_proba(x)

        # Calculate the loss function
        self.loss_list.append(binary_cross_entropy(x, y, y_hat))

        # Update the weights
        self.update_weights(x, y, y_hat)

    def predict(self, x):
        """
        Predicts probabilities for each row and evaluates against threshold
        :param x: Features for prediction
        :return: Predicted class for each row
        """
        return 1 * (self.predict_proba(x) >= self.threshold)

    def predict_proba(self, x):
        """
        :param x: Features for prediction
        :return: Probability for each row
        """
        return sigmoid(np.dot(x, self.weights))

    def update_weights(self, x, y, y_hat):
        """
        :param x: Training features
        :param y: Training target
        :param y_hat: Predicted probabilities
        """
        error_Xy = np.dot(x.T, (y_hat - y))
        self.weights -= (self.learning_rate / x.shape[0] * error_Xy)


class MultinomialLogisticRegression(BaseModel):
    # reference: https://towardsdatascience.com/ml-from-scratch-multinomial-logistic-regression-6dda9cbacf9d

    def __init__(self, learning_rate=1e-9, n_iters=1000, init=None, verbose=False, seed=12345):
        """
        :param learning_rate: Rate at which the model learns from errors
        :param n_iters: Number of times weights will be updated
        :param verbose: Flag to print cost during training after each 100 iterations
        :param seed: Random seed
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.weights = None
        self.biases = None
        self.n_classes = 0
        self.loss_list = []
        self.init = init
        self.seed = seed

    def fit(self, x, y):
        """
        :param x: Features for training
        :param y: Target for training
        """

        self.n_classes = y.max() + 1
        if self.init == 'random':
            np.random.seed(self.seed)
            self.weights = np.random.rand(self.n_classes, x.shape[1])
            self.biases = np.random.rand(self.n_classes, 1)
        else:
            self.weights = np.zeros((self.n_classes, x.shape[1]))
            self.biases = np.zeros((self.n_classes, 1))

        y = y.astype(int)

        for i in range(self.n_iters):
            self.partial_fit(x, y)

            if self.verbose:
                if i % 100 == 0:
                    print(self.loss_list[-1])

    def partial_fit(self, x, y):
        """
        Performs a single iteration update of weights
        :param x: Features for training
        :param y: Target for training
        """
        # Calculate the softmax distributions for x
        y_hat = self.predict_proba(x)

        self.loss_list.append(categorical_cross_entropy(y_hat, y))

        # Update the weights
        self.update_weights(x, y, y_hat)

    def predict(self, x):
        """
        Predicts probabilities for each row and selects the max
        :param x: Features for prediction
        :return: Predicted class for each row
        """
        y_hat = self.predict_proba(x)
        return np.argmax(y_hat, axis=1)

    def predict_proba(self, x):
        """
        :param x: Features for prediction
        :return: Probability for each row for each class
        """
        biases = np.array([self.biases.reshape(1, -1)[0] for _ in range(x.shape[0])])
        logits = np.dot(self.weights, x.T).T + biases
        probabilities = softmax(logits)
        return probabilities

    def update_weights(self, x, y, y_hat):
        """
        :param x: Training features
        :param y: Training target
        :param y_hat: Predicted probabilities
        """
        y_hat[np.arange(x.shape[0]), y] -= 1

        self.weights -= (self.learning_rate * np.dot(y_hat.T, x))
        self.biases -= (self.learning_rate * np.sum(y_hat, axis=0).reshape(-1, 1))

