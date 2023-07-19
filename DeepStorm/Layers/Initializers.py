import numpy as np


class Constant:
    def __init__(self, value=0.01) -> None:
        self.value = value

    def initialize(self, weights_shape=None, fan_in=None, fan_out=None):
        return np.full((fan_in, fan_out), self.value)


class UniformRandom:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.rand(*weights_shape)
        return weights


class Xavier:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt((2) / (fan_in + fan_out))
        weights = np.random.normal(0, sigma, size=weights_shape)
        return weights


class He:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt((2/(fan_in))), size=(weights_shape))
