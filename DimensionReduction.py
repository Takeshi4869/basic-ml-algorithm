# 降维算法

# 实现了PCA, LDA. 其中LDA仅支持二分类问题, 且只能从高维降至一维

import numpy as np


class PCA:
    def __init__(self, k=2):
        self.k = k
        self.W = None

    def _demean(self, X):
        return X - np.mean(X, axis=0)

    def fit(self, X):
        # 中心化
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


class LDA:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        # 划分出第一类样本与第二类样本
        label = np.unique(y)
        X0 = X[y == label[0]]
        X1 = X[y == label[1]]

        # 获取第一类样本与第二类样本中心点
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)

        # 计算第一类样本与第二类样本协方差矩阵
        simga0 = np.dot((X0 - mu0).T, (X0 - mu0))
        simga1 = np.dot((X1 - mu1).T, (X1 - mu1))

        # 计算类内散度矩阵
        S_w = simga0 + simga1

        # 计算W
        self.W = np.dot(np.linalg.inv(S_w), (mu0 - mu1))

    def transform(self, X):
        return np.dot(X, self.W)
