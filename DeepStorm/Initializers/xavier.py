import numpy as np


class Xavier:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt((2) / (fan_in + fan_out))
        return np.random.normal(0, sigma, size=weights_shape)