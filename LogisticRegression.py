import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def calc_z(w, x, b):
    return np.dot(w, x.T) + b


class LogisticRegression:
    def __init__(self, m):
        self.w = np.zeros((1, m))
        self.b = 0
        self.lr = 0.001
        self.epochs = 1000

    def gradient_descent(self, x_train, y_train):
        for epoch in range(self.epochs):
            for i in range(len(x_train)):
                Z = calc_z(self.w, x_train[i], self.b)[0]
                A = sigmoid(Z)

                self.w -= self.lr * (A - y_train[i]) * x_train[i]
                self.b -= self.lr * (A - y_train[i])

    def predict(self, X):
        return np.array([sigmoid(calc_z(self.w, x, self.b)[0]) for x in X])




