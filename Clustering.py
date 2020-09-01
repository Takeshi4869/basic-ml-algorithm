# 聚类算法
#
# 实现了KMeans，

import numpy as np
import random


class KMeans:
    def __init__(self, k=2, max_iter=500, epsilon=0.001, p=2):
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.p = p
        np.random.seed(1)

    def _distance(self, X, one_sample):
        return (np.sum((abs(X - one_sample)) ** self.p, axis=1)) ** (1 / self.p)

    def fit(self, X, y):
        return

    def predict(self, X):
        m, d = X.shape
        # 随机选取k个样本作为初始的聚类中心
        centroids = np.array(random.choices(X, k=self.k))
        label = np.zeros(m)
        for _ in range(self.max_iter):
            # 将该样本归类到与其最近的中心
            for i in range(m):
                dists = self._distance(centroids, X[i, :])
                label[i] = np.argmin(dists)

            # 计算新的聚类中心
            centroids_old = centroids
            for i in range(self.k):
                centroids[i] = np.mean(X[label == i], axis=0)

            # 算法收敛，退出迭代
            if (abs(centroids_old - centroids)).all() < self.epsilon:
                break

        return label

    def score(self, X, y):
        """返回 JC系数, FM指数, Rand指数"""
        m = len(y)
        y_pred = self.predict(X)
        ss = sd = ds = dd = 0
        for i in range(m):
            for j in range(i+1, m):
                if y_pred[i] == y_pred[j] and y[i] == y[j]:
                    ss += 1
                elif y_pred[i] == y_pred[j] and y[i] != y[j]:
                    sd += 1
                elif y_pred[i] != y_pred[j] and y[i] == y[j]:
                    ds += 1
                else:
                    dd += 1

        jc = ss / (ss+sd+ds)
        fm = ss / np.sqrt((ss+sd)*(ss+ds))
        ri = 2*(ss+dd) / (m*(m-1))
        return jc, fm, ri
