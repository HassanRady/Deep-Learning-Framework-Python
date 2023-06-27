from DeepStorm.Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.pos = x > 0
        return np.maximum(x, 0)

    def backward(self, y):
        return self.pos * y
