import numpy as np


class DecisionTree:
    def __init__(self, algorithm='ID3'):
        self.tree = {}
        self.algorithm = algorithm

    def cal_entropy(self, label):
        probs = np.unique(label, return_counts=True)[1] / len(label)
        entropy = 0
        for prob in probs:
            entropy -= prob * np.log2(prob)
        return entropy

    def cal_gain(self, X, y, index):
        gain = self.cal_entropy(y)
        n = len(y)
        feature = X[:, index]
        for val in np.unique(feature):
            gain -= len(y[feature == val]) / n * self.cal_entropy(y[feature == val])
        return gain

    def cal_gain_ratio(self, X, y, index):
        iv = self.cal_entropy(X[:, index])
        gain = self.cal_gain(X, y, index)
        return gain / iv

    def cal_gini_index(self, X, y, index):
        def cal_gini(label):
            probs = np.unique(label, return_counts=True)[1] / len(label)
            gini = 1 - np.sum(probs**2)
            return gini

        feature = X[:, index]
        gini_index = 0
        n = len(y)
        for val in np.unique(feature):
            gini_index += len(y[feature == val]) / n * cal_gini(y[feature == val])
        return gini_index

    def ID3(self, feature, label):
        # 样本全属于同一类别
        if len(np.unique(label)) == 1:
            return label[0]

        # 样本中只有一个特征或所有样本的特征都一样
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:
            counts = np.unique(label, return_counts=True)
            return counts[0][np.argmax(counts[1])]

        m, d = feature.shape
        gains = []
        for idx in range(d):
            gain = self.cal_gain(feature, label, idx)
            gains.append(gain)
        best_feature = np.argmax(gains)
        tree = {}
        tree[best_feature] = {}
        value_set = np.unique(feature[:, best_feature])
        for val in value_set:
            sub_feature = feature[feature[:, best_feature] == val]
            sub_label = label[feature[:, best_feature] == val]
            tree[best_feature][val] = self.ID3(sub_feature, sub_label)

        return tree

    def fit(self, X, y):
        if self.algorithm == 'ID3':
            self.tree = self.ID3(X, y)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            node = self.tree
            while isinstance(node, dict):
                feature = list(node.keys())[0]
                assert len(node.keys()) == 1
                node = node[feature][X[i, feature]]
            y_pred.append(node)
        return np.array(y_pred)

