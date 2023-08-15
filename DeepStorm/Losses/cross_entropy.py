import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.trainable = False

    def forward(self, y_hat, y):
        self.y_hat = y_hat
        return -np.sum(y*np.log(y_hat + np.finfo(float).eps))

    def backward(self, y):
        return np.where(y == 1, -y/(self.y_hat+np.finfo(float).eps), 0)
