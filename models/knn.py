from dis import dis
import numpy as np


class knn:
    def __init__(self, features, labels, data_point) -> None:
        self.features = features
        self.labels = labels
        self.data_point = data_point

    def euclidean_distance(self):
        distance_matrix = []
        for values in self.features:
            distance_matrix.append(np.linalg.norm(self.data_point - values))
        return np.array(distance_matrix).reshape(self.features.shape[0], 1)

    def manhattan_distance(self):
        distance_matrix = []
        for values in self.features:
            distance_matrix.append(np.sum(np.abs(self.data_point - values)))
        return np.array(distance_matrix).reshape(self.features.shape[0], 1)

    def chebyshev_distance(self):
        distance_matrix = []
        for values in self.features:
            distance_matrix.append(np.amax(np.abs(self.data_point - values)))
        return np.array(distance_matrix).reshape(self.features.shape[0], 1)
