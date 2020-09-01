# 集成学习算法
#
# 实现了Adaboost, Bagging, RandomForest, 基分类器均采用sklearn中的DecisionTreeClassifier


import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.T = n_estimators
        self.alphas = []
        self.clf = []

    def _prob(self, X, y, h, D):
        y_pred = h.predict(X)
        return np.sum((y != y_pred) * D)

    def fit(self, X, y):
        """接受标签为1 -1的数据"""
        # 初始化样本权值分布
        m = X.shape[0]
        D = np.ones(m) / m
        for t in range(self.T):
            # 设置决策树的最大深度，防止过拟合
            h = DecisionTreeClassifier(max_depth=3)
            # 基于分布D训练处分类器h
            h.fit(X, y, sample_weight=D)

            # 估计h的误差
            epsilon = self._prob(X, y, h, D)
            if epsilon > 0.5:
                break

            # 更新样本分布, 并正则化
            alpha = 1 / 2 * np.log((1 - epsilon) / epsilon)
            D = D * np.exp(-alpha * h.predict(X) * y)
            D /= np.sum(D)

            self.clf.append(h)
            self.alphas.append(alpha)

    def predict(self, X):
        score = np.zeros(X.shape[0])
        for i in range(len(self.alphas)):
            score += self.alphas[i] * self.clf[i].predict(X)
        return np.sign(score)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy


def bootstrapping(m):
    D_bs = np.zeros(m)
    for i in range(m):
        idx = random.randint(0, m - 1)
        D_bs[idx] += 1 / m
    return D_bs


class Bagging:
    def __init__(self, n_estimators=200):
        self.T = n_estimators
        self.clf = []

    def fit(self, X, y):
        m, d = X.shape
        for t in range(self.T):
            # 自助采样产生样本分布
            D_bs = bootstrapping(m)
            h = DecisionTreeClassifier()
            h.fit(X, y, sample_weight=D_bs)
            self.clf.append(h)

    def predict(self, X):
        scores = []
        d = X.shape[1]
        for y in [0, 1]:
            pred = np.array([h.predict(X.reshape(-1, d)) == y for h in self.clf])
            score = np.sum(pred, axis=0)
            scores.append(score)
        y_pred = np.argmax(scores, axis=0)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy


class RandomForest:
    def __init__(self, n_estimators=200, k=None):
        self.T = n_estimators
        self.k = k
        self.clf = []

    def fit(self, X, y):
        m, d = X.shape
        if self.k is None:  # 默认使用 log_2_d
            max_features = 'log2'
        else:
            max_features = self.k

        for t in range(self.T):
            # 自助采样产生样本分布
            D_bs = bootstrapping(m)
            # 随机属性选择
            h = DecisionTreeClassifier(max_features=max_features)
            h.fit(X, y, sample_weight=D_bs)
            self.clf.append(h)

    def predict(self, X):
        scores = []
        d = X.shape[1]
        for y in [0, 1]:
            pred = np.array([h.predict(X.reshape(-1, d)) == y for h in self.clf])
            score = np.sum(pred, axis=0)
            scores.append(score)
        y_pred = np.argmax(scores, axis=0)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
