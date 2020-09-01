# K近邻


import numpy as np


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        def distance(one_sample):
            # 欧式距离
            distances = []
            for i in range(self.X_train.shape[0]):
                distances.append(np.sqrt(np.sum((self.X_train[i, :] - one_sample) ** 2)))
            return distances

        def get_k_neighbor_labels(distances):
            distances = np.array(distances)
            idx = distances.argsort()
            nearest_idx = idx[:self.k]
            return self.y_train[nearest_idx]

        def vote(one_sample):
            distances = distance(one_sample)
            nearest_labels = get_k_neighbor_labels(distances)
            labels_count = {}
            for i in nearest_labels:
                labels_count[i] = labels_count.get(i, 0) + 1
            labels_count = sorted(labels_count, key=lambda label: labels_count[label])
            return labels_count[-1]

        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(vote(X[i, :]))
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)


