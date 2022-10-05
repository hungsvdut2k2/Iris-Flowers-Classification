import numpy as np


class logistic_regression:
    def __init__(self, theta, features, label) -> None:
        self.theta = theta
        self.features = features
        self.label = label

    def compute_value(self):
        return np.dot(self.theta, self.features.T)

    def sigmoid_function(self):
        return 1 / (1 + np.exp(self.compute_value()))

    def loss_function(self):
        predicted_value = self.sigmoid_function()
        return -predicted_value * np.log(self.label) - (1 - predicted_value) * np.log(
            1 - self.label
        )

    def compute_gradient(self):
        return (
            np.dot(self.features.T, (self.label - self.sigmoid_function()))
            / self.features.shape[0]
        )

    def gradient_descent(self):
        for i in range(1000):
            self.theta -= self.compute_gradient()
        return self.theta
