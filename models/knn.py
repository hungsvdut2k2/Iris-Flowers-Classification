import numpy as np


class knn:
    def __init__(self, features, data_point, k) -> None:
        self.features = features
        self.data_point = data_point
        self.k = k

    def euclidean_distance(self):
        distance_matrix = []
        for data in self.features:
            values = data[:4]
            distance_matrix.append((np.linalg.norm(self.data_point - values), data[-1]))
        return np.array(distance_matrix).reshape(self.features.shape[0], 2)

    def manhattan_distance(self):
        distance_matrix = []
        for data in self.features:
            values = data[:4]
            distance_matrix.append((np.sum(np.abs(self.data_point - values)), data[-1]))
        return np.array(distance_matrix).reshape(self.features.shape[0], 2)

    def chebyshev_distance(self):
        distance_matrix = []
        for data in self.features:
            values = data[:4]
            distance_matrix.append(
                (np.amax(np.abs(self.data_point - values)), data[-1])
            )
        return np.array(distance_matrix).reshape(self.features.shape[0], 2)

    def ranking(self, method):
        distance_matrix = []
        if method == "euclidean":
            distance_matrix = self.euclidean_distance()
        elif method == "manhattan":
            distance_matrix = self.manhattan_distance()
        elif method == "chebyshev":
            distance_matrix = self.chebyshev_distance()
        else:
            raise "doesn't exist that method"
        ranking_distance = distance_matrix[distance_matrix[:, 0].argsort()]
        return ranking_distance

    def voting(self, method):
        k_nearest = self.ranking(method=method)[: self.k]
        return k_nearest
