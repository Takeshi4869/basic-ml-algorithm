# 朴素贝叶斯分类器
#
# 加入拉普拉斯修正

import numpy as np


class NaiveBayesClassifier:
    def __init__(self, smoothing=False):
        self.label_prob = {}
        self.condition_prob = {}
        self.smoothing = smoothing

    def _cal_prob(self, array, labels):
        n = len(array)
        probs = {}
        for label in labels:
            if self.smoothing:
                # 拉普拉斯修正
                probs[label] = (np.sum(array==label) + 1 ) / (n + len(labels))
            else:
                probs[label] = np.sum(array==label) / n
        return probs

    def fit(self, X, y):
        m, d = X.shape
        self.label_prob = self._cal_prob(y, np.unique(y))
        for label in np.unique(y):
            self.condition_prob[label] = {}
            for i in range(d):
                self.condition_prob[label][i] = self._cal_prob(X[y==label, i], np.unique(X[:, i]))

    def predict(self, X):
        y_pred = []
        m, d = X.shape
        for i in range(m):
            probs = {}
            for label, pr in self.label_prob.items():
                prob = pr
                for j in range(d):
                    prob *= self.condition_prob[label][j].get(X[i, j], 0)
                probs[label] = prob
            y_pred.append(max(probs, key=probs.get))
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y == y_pred) / len(y)

