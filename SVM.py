# 支持向量机
#
# 用QP(软间隔和硬间隔)方法和SMO算法实现支持向量机分类

import numpy as np
import random
from cvxopt import solvers, matrix


def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)


def gaussian_kernel(x1, x2, sigma=1.5):
    return np.exp(-np.linalg.norm(x1-x2)**2/(2*(sigma**2)))


def sigmoid_kernel(x1, x2, beta=10, theta=0):
    return np.tanh(np.dot(x1, x2.T)*beta+theta)


class SMO:
    def __init__(self, C=1, kernel='linear', tol=1e-4, max_iter=300):
        # training data
        self.X = None
        self.y = None
        self.K = None
        # training result
        self.alpha = None
        self.b = 0
        # argument
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        if kernel == "linear":
            self.kernel = linear_kernel

    def _kernel_matrix(self, X1, X2):
        kernel = self.kernel
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                K[i, j] = kernel(X1[i, :], X2[j, :])
        return K

    def _select_j(self, i, m):
        j = i
        while i == j:
            j = random.randint(0, m - 1)
        return j

    def _g(self, i):
        g = np.dot((self.alpha*self.y).T, self.K[:, i]) + self.b
        return g

    def _E(self, i):
        g = self._g(i)
        return g - self.y[i]

    def _clip(self, a, L, H):
        if a >= H:
            return H
        elif a <= L:
            return L
        else:
            return a

    def _kkt(self, i):
        y_g = self._g(i) * self.y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def _init_data(self, X, y):
        self.X = X
        self.y = y
        self.K = self._kernel_matrix(X, X)

    def fit(self, X, y):
        # 初始化数据和参数
        self._init_data(X, y)
        m, n = X.shape
        self.alpha = np.zeros((m, 1))
        b = 0
        C = self.C
        K = self.K

        for _ in range(self.max_iter):
            for i in range(m):
                j = self._select_j(i, m)
                alpha_i = self.alpha[i]
                alpha_j = self.alpha[j]
                E_i = self._E(i)
                E_j = self._E(j)

                if self._kkt(i):
                    continue

                # 计算 L H
                if y[i] != y[j]:
                    L = max(0, alpha_j - alpha_i)
                    H = min(C, C + alpha_j - alpha_i)
                else:
                    L = max(0, alpha_i + alpha_j - C)
                    H = min(C, alpha_i + alpha_j)
                if L == H:
                    continue

                # 计算 eta
                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta == 0:
                    continue

                # 更新 alpha_j alpha_i
                self.alpha[j] -= y[j] * (E_i - E_j) / eta
                self.alpha[j] = self._clip(self.alpha[j], L, H)
                self.alpha[i] = alpha_i + y[i]*y[j]*(alpha_j - self.alpha[j])

                # 更新 b
                b1 = b - E_i - y[i]*(self.alpha[i]-alpha_i)*K[i, i] - y[j]*(self.alpha[j]-alpha_j)*K[i, j]
                b2 = b - E_j - y[i]*(self.alpha[i]-alpha_i)*K[i, j] - y[j]*(self.alpha[j]-alpha_j)*K[j, j]
                if 0 < self.alpha[i] < C:
                    b = b1
                elif 0 < self.alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                self.b = b

    def predict(self, X):
        kernel = self.kernel
        scores = []
        alpha_y = self.alpha * self.y
        for i in range(X.shape[0]):
            score = 0
            for j in range(self.X.shape[0]):
                score += alpha_y[j] * self.kernel(self.X[j, :], X[i, :])
            score += self.b
            scores.append(score[0])
        scores = np.array(scores)
        y_pred = np.sign(scores)
        return y_pred


class SVM:
    def __init__(self, C=1, tol=1e-4, max_iter=200, kernel='linear', method='QP'):
        self.max_iter = max_iter
        self._kernel = kernel
        self.C = C
        self.tol = tol
        self.method = method
        self.kernel = kernel
        self.b = 0
        self.w = 0

    def _get_hyperplane(self, X, y, alpha):
        w = np.dot(X.T, alpha * y)
        b = np.mean(y - np.dot(w.T, X.T).T)
        return w, b

    def _qp_method_hard(self, X, y):
        m = X.shape[0]
        y = y.reshape((-1, 1))
        y.dtype = np.float

        K = y * X
        P = np.dot(K, K.T) + 1e-5 * np.eye(m)  # 使P为正定矩阵，否则不收敛，且terminated (singular KKT matrix)
        P = matrix(P)
        q = -matrix(np.ones((m, 1)))
        G = -matrix(np.eye(m))
        h = matrix(np.zeros(m))
        A = matrix(y.reshape((1, -1)))
        b = matrix(np.zeros(1))
        solvers.options['maxiters'] = self.max_iter
        solvers.options['show_progress'] = False
        alpha = np.array(solvers.qp(P, q, G, h, A, b)['x'])
        self.w, self.b = self._get_hyperplane(X, y, alpha)

    def _qp_method_soft(self, X, y):
        C = self.C
        m = X.shape[0]
        y = y.reshape((-1, 1))
        y.dtype = np.float

        K = y * X
        P = np.dot(K, K.T) + 1e-5 * np.eye(m)  # 使P为正定矩阵，否则不收敛，且terminated (singular KKT matrix)
        P = matrix(P)
        q = -matrix(np.ones((m, 1)))
        G = matrix(np.vstack((-np.eye(m), np.eye(m))))
        h = matrix(np.vstack((np.zeros((m, 1)), np.ones((m, 1)) * C)))
        A = matrix(y.reshape((1, -1)))
        b = matrix(np.zeros(1))
        solvers.options['maxiters'] = self.max_iter
        solvers.options['show_progress'] = False
        alpha = np.array(solvers.qp(P, q, G, h, A, b)['x'])
        self.w, self.b = self._get_hyperplane(X, y, alpha)

    def _smo(self, X, y):
        y = y.reshape((-1, 1))
        self.smo = SMO(self.C, tol=self.tol, kernel=self.kernel, max_iter=self.max_iter)
        self.smo.fit(X, y)

    def fit(self, X, y):
        if self.method == 'qp_hard':
            self._qp_method_hard(X, y)
        elif self.method == 'qp_soft':
            self._qp_method_soft(X, y)
        elif self.method == 'smo':
            self._smo(X, y)

    def predict(self, X):
        if self.method == 'smo':
            y_pred = self.smo.predict(X)
        else:
            score = np.dot(X, self.w) + self.b
            y_pred = np.where(score > 0, 1, -1).reshape(-1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)