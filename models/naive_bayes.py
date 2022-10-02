from cmath import sqrt
import numpy as np
import math


class naive_bayes:
    def __init__(self, train_set, input_data) -> None:
        self.train_set = train_set
        self.input_data = input_data

    def gaussian_distribution(self, label):
        labels = self.train_set[:, -1]
        features = self.train_set[np.where(labels == label), :-1]
        result = []
        index = 0
        for column in features.T:
            mean_value = np.mean(column)
            variance_value = np.var(column)
            data = self.input_data[index]
            result.append(
                math.exp((mean_value - data) ** 2 / 2 * variance_value**2)
                / (variance_value * sqrt(2 * math.pi))
            )
            index += 1
        return np.array(result)

    def proability_of_labels(self):
        result = []
        for label in np.unique(self.train_set[:, -1]):
            result.append(
                len(np.where(self.train_set[:, -1] == label)[0])
                / self.train_set.shape[0]
            )
        return np.array(result)

    def proability_of_input_in_condition_label(self):
        result = []
        for label in np.unique(self.train_set[:, -1]):
            result.append(np.prod(self.gaussian_distribution(label=label)))
        return np.array(result)

    def bayes_rule(self):
        return (
            self.proability_of_input_in_condition_label() * self.proability_of_labels()
        )

    def voting(self):
        max_value_index = 0
        bayes_values = self.bayes_rule()
        for i in range(1, bayes_values.shape[0]):
            if bayes_values[i] > bayes_values[max_value_index]:
                max_value_index = i
        return max_value_index

    def predict(self):
        return np.unique(self.train_set[:, -1])[self.voting()]
