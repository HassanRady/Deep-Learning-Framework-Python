# TODO: handle testing phase

import numpy as np
from DeepStorm.Layers.Base import BaseLayer


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
            input_tensor.shape[-2], input_tensor.shape[-1]) < self.p
        res = np.multiply(input_tensor, self.mask)
        res /= self.p
        return res

    def backward(self, error_tensor):
        res = error_tensor * self.mask
        res /= self.p
        return res
