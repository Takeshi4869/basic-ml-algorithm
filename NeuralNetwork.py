# 神经网络
#
# 用反向传播算法实现结构为2-2-1的神经网络(NeuralNetwork221), 支持特征数为2的数据
#    o - o
#     \ / \
#      x   o
#     / \ /
#    o - o

import numpy as np


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    # 避免极大溢出
    else:
        return np.exp(x) / (1 + np.exp(x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse_loss(y_true, y_pred):
    return np.sum((y_true-y_pred)**2) / len(y_true)


class NeuralNetwork221:
    def __init__(self, silence=True):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        # 显示epoch的loss
        self.silence = silence

    def output(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def fit(self, feature, label):
        learn_rate = 0.1
        epochs = 200
        for epoch in range(epochs):
            for x, y in zip(feature, label):
                x1, x2 = x
                # 向前传播过程
                sum_h1 = self.w1*x1 + self.w2*x2 + self.b1  # （隐层第一个节点收到的输入之和）
                h1 = sigmoid(sum_h1)                        # （隐层第一个节点的输出）
                sum_h2 = self.w3*x1 + self.w4*x2 + self.b2  # （隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2)                        # （隐层第二个节点的输出）
                sum_ol = self.w5*h1 + self.w6*h2 + self.b3  # （输出层节点收到的输入之和）
                ol = sigmoid(sum_ol)                        # （输出层节点的对率输出）
                y_pred = ol

                # 计算梯度
                d_L_d_ypred = y_pred - y            # （损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = deriv_sigmoid(sum_ol) * h1       # （输出层对率输出对w5的梯度）
                d_ypred_d_w6 = deriv_sigmoid(sum_ol) * h2       # （输出层对率输出对w6的梯度）
                d_ypred_d_b3 = deriv_sigmoid(sum_ol)            # （输出层对率输出对b3的梯度）
                d_ypred_d_h1 = deriv_sigmoid(sum_ol) * self.w5  # （输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = deriv_sigmoid(sum_ol) * self.w6  # （输出层输出对率对隐层第二个节点的输出的梯度）
                # 隐层梯度
                d_h1_d_w1 = deriv_sigmoid(sum_h1) * x1  # （隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = deriv_sigmoid(sum_h1) * x2  # （隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = deriv_sigmoid(sum_h1)       # （隐层第一个节点的输出对b1的梯度）
                d_h2_d_w3 = deriv_sigmoid(sum_h2) * x1  # （隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = deriv_sigmoid(sum_h2) * x2  # （隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = deriv_sigmoid(sum_h2)       # （隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5  # （更新w5）
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6  # （更新w6）
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3  # （更新b3）
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1  # （更新w1）
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2  # （更新w2）
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1  # （更新b1）
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3  # （更新w3）
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4  # （更新w4）
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2  # （更新b2）

            if not self.silence:
                # 计算epoch的loss
                if epoch % 100 == 0:
                    y_preds = np.apply_along_axis(self.output, 1, feature)
                    loss = mse_loss(label, y_preds)
                    print("Epoch", epoch, "loss", loss)

    def score(self, X, y):
        y_pred = self.predict(X)
        return mse_loss(y, y_pred)

    def predict(self, X, threshold=0.5):
        scores = []
        for x in X:
            scores.append(self.output(x))
        scores = np.array(scores)
        y_pred = np.where(scores > threshold, 1, 0)
        return y_pred
