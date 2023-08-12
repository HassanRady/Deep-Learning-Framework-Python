import numpy as np

from DeepStorm.Layers.Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, y):
        sigmoid_grad = self.output * (1 - self.output)
        return y * sigmoid_grad

