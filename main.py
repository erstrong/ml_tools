from ml_tools.regression import BinaryLogisticRegression
from ml_tools.metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    x, y = make_classification(n_samples=5000, n_features=10, n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, random_state=9)

    x_train, x_test, y_train, y_test = train_test_split(x, y.reshape(-1, 1), test_size=0.1, random_state=9)

    print('Actual: ' + str(y_test.sum()/y_test.shape[0]))

    # Binary Logistic Regression
    lr = BinaryLogisticRegression(learning_rate=1e-3, verbose=True)
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    print('BLR Accuracy: ' + str(accuracy(y_test, y_hat)))

