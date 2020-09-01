# 对数几率回归
#
# 使用闭式解法(closed), 梯度下降法(gdc), 牛顿法(newton)实现对数几率回归

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, max_iter=100, epsilon=1e-3, method='newton', learning_rate=0.01):
        self.theta = None
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.method = method
        self.lr = learning_rate
        assert self.method in ['closed', 'newton', 'gdc']

    def _prob(self, x, y):
        z = np.dot(self.theta.T, x)
        pr = 1 / (1 + np.exp(-z))
        return pr

    def _loss(self, X, y):
        loss = 0
        for i in range(len(X)):
            x = X[i,:].reshape([-1,1])
            res = -y[i]*np.dot(self.theta.T, x) + np.log(1+np.exp(np.dot(self.theta.T, x)))
            loss += res.ravel()[0]
        return loss

    def _gradient(self, X, y):
        res = 0
        for i in range(X.shape[0]):
            x = X[i,:].reshape([-1,1])
            res += - x * (y[i] - self._prob(self.theta, x))
        return res

    def _Hessian(self, X):
        m, d = X.shape
        hessian = np.zeros((d, d))
        for i in range(m):
            x = X[i,:].reshape([-1,1])
            hessian += np.dot(x, x.T) * self._prob(self.theta, x) * (1 - self._prob(self.theta, x))
        return hessian

    def _Newton(self, X, y):
        m, d = X.shape
        ones = np.ones(m)
        X = np.column_stack((X, ones))
        self.theta = np.zeros((d+1, 1))
        loss_old = self._loss(X, y)

        for _ in range(self.max_iter):
            hessian = self._Hessian(X)
            gradient = self._gradient(X, y)
            if hessian.all() == 0:
                return

            self.theta -= np.dot(np.linalg.inv(hessian), gradient)

            # 判断收敛条件
            loss_new = self._loss(X, y)
            if abs(loss_new - loss_old) < self.epsilon:
                return
            loss_old = loss_new

    def _closed_sol(self, X, y):
        m, d = X.shape
        ones = np.ones(m)
        X = np.column_stack((X, ones))
        self.theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y.reshape((-1, 1)))

    def _gradient_descent(self, X, y):
        m, d = X.shape
        ones = np.ones(m)
        X = np.column_stack((X, ones))
        self.theta = np.zeros((d + 1, 1))
        loss_old = self._loss(X, y)

        for _ in range(self.max_iter):
            self.theta -= self.lr * self._gradient(X, y)
            # print(self.theta)

            # 判断收敛条件
            loss_new = self._loss(X, y)
            print(loss_new)
            if abs(loss_new - loss_old) < self.epsilon:
                print("return ")
                return
            loss_old = loss_new
        return

    def fit(self, X, y):
        if self.method == 'newton':
            self._Newton(X, y)
        elif self.method == 'closed':
            self._closed_sol(X, y)
        elif self.method == 'gdc':
            self._gradient_descent(X, y)

    def predict(self, X):
        m, d = X.shape
        ones = np.ones(m)
        X = np.column_stack((X, ones))
        z = np.dot(X, self.theta)
        score = sigmoid(z)
        y_pred = np.where(score > 0.5, 1, 0).reshape(-1)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        acc = np.sum(y_pred == y) / len(y)
        return acc