import numpy as np


class Constant:
    def __init__(self, value=0.01) -> None:
        self.value = value

    def initialize(self, weights_shape=None, fan_in=None, fan_out=None):
        return np.full((fan_in, fan_out), self.value)