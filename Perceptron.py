# 感知机


import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, max_iter=200):
        self.eta =  learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = 1

    def _f(self, X, y):
        return np.dot(self.w, X.T) + self.b

    def fit(self, X, y):
        """接受标签为1 -1的数据"""
        m, d = X.shape
        self.w = np.ones(d)
        for _ in range(self.max_iter):
            has_wrong = False
            for i in range(m):
                if y[i]*(np.dot(self.w.T, X[i])+self.b) <= 0:
                    has_wrong = True
                    self.w += self.eta * y[i] * X[i]
                    self.b += self.eta * y[i]
            if not has_wrong:
                break

    def predict(self, X):
        y_pred = np.sign(np.dot(X, self.w) + self.b)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)