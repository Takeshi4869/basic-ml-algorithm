import unittest
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def cross_validation(model, X, y, fold=5):
    """交叉验证法"""
    m = X.shape[0]
    idx = list(range(m))
    np.random.shuffle(idx)
    _X = X[idx, :]
    _y = y[idx]
    size = m // fold
    scores = []
    for i in range(fold):
        _X_test = _X[i * size:(i + 1) * size, ]
        _y_test = _y[i * size:(i + 1) * size, ]
        _X_train = np.vstack((_X[:i * size, ], _X[(i + 1) * size:, ]))
        _y_train = np.hstack((_y[:i * size], _y[(i + 1) * size:]))
        model.fit(_X_train, _y_train)
        scores.append(model.score(_X_test, _y_test))
    return scores


class LinearRegressionTestCase(unittest.TestCase):
    def testLinearRegression(self):
        from LinearRegression import LinearRegression
        boston = load_boston()
        X_train = boston.data
        y_train = boston.target
        lr = LinearRegression()
        mse = np.mean(cross_validation(lr, X_train, y_train))
        print("Linear Regression with closed solution method: MSE =", mse)


class LogisticRegressionTestCase(unittest.TestCase):
    def testNewton(self):
        from LogisticRegression import LogisticRegression
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        lr = LogisticRegression(method='newton')
        acc = np.mean(cross_validation(lr, X_train, y_train))
        print("Logisitic regression with Newton method: accuracy =", acc)

    def testClosedSol(self):
        from LogisticRegression import LogisticRegression
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        lr = LogisticRegression(method='closed')
        acc = np.mean(cross_validation(lr, X_train, y_train))
        print("Logisitic regression with closed solution method: accuracy =", acc)

    # 本数据中不收敛
    # def testGradientDescent(self):
    #     from LogisticRegression import LogisticRegression
    #     breast_cancer = load_breast_cancer()
    #     X_train = breast_cancer.data
    #     y_train = breast_cancer.target
    #     lr = LogisticRegression(method='gdc')
    #     acc = np.mean(cross_validation(lr, X_train, y_train))
    #     print("Logisitic regression with closed solution method: accuracy =",


class DecisionTreeTestCase(unittest.TestCase):
    def testSmoke(self):
        from DecisionTree import DecisionTree
        dt = DecisionTree()
        feature = np.array([[0, 1], [1, 0], [1, 2], [0, 0], [1, 1]])
        label = np.array([0, 1, 0, 0, 1])
        dt.fit(feature, label)
        y_pred = dt.predict(feature)
        assert (y_pred == label).all()


class SVMTestCase(unittest.TestCase):
    def testQP(self):
        from SVM import SVM
        dir_path = os.path.dirname(os.path.realpath(__file__))
        X_train = np.genfromtxt(dir_path + "/datasets/X_train.csv", delimiter=',')
        y_train = np.genfromtxt(dir_path + "/datasets/y_train.csv", delimiter=',')
        y_train[y_train == 0] = -1
        svm = SVM(method='qp_hard')
        acc = np.mean(cross_validation(svm, X_train, y_train))
        print("SVM with QP method (hard margin): accuracy =", acc)
        svm = SVM(method='qp_soft')
        acc = np.mean(cross_validation(svm, X_train, y_train))
        print("SVM with QP method (soft margin): accuracy =", acc)

    def testSMO(self):
        from SVM import SVM
        dir_path = os.path.dirname(os.path.realpath(__file__))
        X_train = np.genfromtxt(dir_path + "/datasets/X_train.csv", delimiter=',')
        y_train = np.genfromtxt(dir_path + "/datasets/y_train.csv", delimiter=',')
        y_train[y_train == 0] = -1
        svm = SVM(method='smo')
        acc = np.mean(cross_validation(svm, X_train, y_train))
        print("SVM with SMO: accuracy =", acc)


class BayesTestCase(unittest.TestCase):
    def testIris(self):
        from NaiveBayesClassifier import NaiveBayesClassifier
        iris = load_iris()
        X_train = iris.data
        y_train = iris.target
        nb = NaiveBayesClassifier(smoothing=True)
        accuracy = np.mean(cross_validation(nb, X_train, y_train))
        print("NaiveBayesClassifier with Laplacian correction: accuracy:", accuracy)


class NeuralNetworkTestCase(unittest.TestCase):
    def testNN221(self):
        from NeuralNetwork import NeuralNetwork221
        from sklearn.decomposition import PCA
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        pca = PCA(n_components=2)
        y_train = breast_cancer.target
        X_train = pca.fit_transform(X_train, y_train)
        model = NeuralNetwork221()
        accuracy = np.mean(cross_validation(model, X_train, y_train))
        print("NeuralNetwork 2-2-1: MSE =", accuracy)


class PerceptionTestCase(unittest.TestCase):
    def testBreastCancer(self):
        from Perceptron import Perceptron
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        y_train[y_train == 0] = -1
        perceptron = Perceptron()
        accuracy = np.mean(cross_validation(perceptron, X_train, y_train))
        print("Perceptron: accuracy =", accuracy)


class KNNTestCase(unittest.TestCase):
    def testBreastCancer(self):
        from KNN import KNNClassifier
        breast_cancer = load_breast_cancer()
        knn = KNNClassifier(k=5)
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        accuracy = np.mean(cross_validation(knn, X_train, y_train))
        print("KNN: accuracy =", accuracy)


class ClusteringTestCase(unittest.TestCase):
    def testKMeans(self):
        from Clustering import KMeans
        iris = load_iris()
        X_test = iris.data
        y_test = iris.target
        kmeans = KMeans(k=3)
        jc, fm, ri = kmeans.score(X_test, y_test)
        print("KMeans: JC={}, FM={}, RI={}".format(jc, fm, ri))


class EnsemblingTestCase(unittest.TestCase):
    def testAdaboost(self):
        from Ensembling import AdaBoost
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        y_train[y_train == 0] = -1
        ada = AdaBoost()
        acc = np.mean(cross_validation(ada, X_train, y_train))
        print("AdaBoost: accuracy = ", acc)

    def testBagging(self):
        from Ensembling import Bagging
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        bag = Bagging()
        acc = np.mean(cross_validation(bag, X_train, y_train))
        print("Bagging: accuracy =", acc)

    def testRandomForest(self):
        from Ensembling import RandomForest
        breast_cancer = load_breast_cancer()
        X_train = breast_cancer.data
        y_train = breast_cancer.target
        rf = RandomForest()
        acc = np.mean(cross_validation(rf, X_train, y_train))
        print("RandomForest: accuracy =", acc)


class DimensionReductionTestCase(unittest.TestCase):
    def testPCA(self):
        from DimensionReduction import PCA
        breast_cancer = load_iris()
        X = breast_cancer.data
        y = breast_cancer.target
        pca = PCA(k=2)
        pca.fit(X)
        X_transformed = pca.transform(X)
        sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], hue=y)
        plt.title("PCA")
        plt.show()

    def testLDA(self):
        from DimensionReduction import LDA
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        lda = LDA()
        lda.fit(X, y)
        X_transformed = lda.transform(X)
        sns.scatterplot(x=X_transformed, y=0, hue=y)
        plt.title("LDA")
        plt.show()


if __name__ == '__main__':
    unittest.main()