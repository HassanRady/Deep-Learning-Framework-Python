import numpy as np


class Constant:
    def __init__(self, value=0.01) -> None:
        self.value = value
        self.fan_in = None
        self.fan_out = None
        self.weights_shape = None

    def initialize(self, weights_shape=None, fan_in=None, fan_out=None):
        self.weights_shape = weights_shape
        self.fan_in = fan_in
        self.fan_out = fan_out
        return np.full((fan_in, fan_out), self.value)