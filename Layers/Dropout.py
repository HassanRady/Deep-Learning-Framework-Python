import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.p = probability
        self.trainable = False
        self.testing_phase = False

    def forward(self, input_tensor):
        if self.testing_phase == True:
            return input_tensor

        self.mask = np.random.rand(
            input_tensor.shape[0], input_tensor.shape[1]) < self.p
        res = input_tensor * self.mask
        res /= self.p
        return res

    def backward(self, error_tensor):
        res = error_tensor * self.mask
        res /= self.p
        return res
