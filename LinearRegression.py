# 线性回归
#
# 使用闭式解方法实现线性回归

import numpy as np


class LinearRegression(object):
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        # 闭式解方法
        m, d = X.shape
        ones = np.ones(m)
        X = np.column_stack((X, ones))
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def predict(self, X):
        m, d = X.shape
        ones = np.ones(m)
        X = np.column_stack((X, ones))
        pred = np.dot(X, self.theta)
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(np.sum((y_pred - y)**2)) / len(X)