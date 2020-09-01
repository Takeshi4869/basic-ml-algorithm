import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k
        self.W = None

    def _demean(self, X):
        return X - np.mean(X, axis=0)

    def fit(self, X):
        # 中性化
        _X = self._demean(X)
        # 计算协方差矩阵
        cov = np.cov(_X.T)
        # 计算特征值、特征向量
        value, vector = np.linalg.eig(cov)
        # 取最大的k个特征值对应的特征向量
        index = np.argsort(value)[::-1][:self.k]
        self.W = vector[:, index]
        print(self.W.shape)

    def transform(self, X):
        return np.dot(self._demean(X), self.W)

