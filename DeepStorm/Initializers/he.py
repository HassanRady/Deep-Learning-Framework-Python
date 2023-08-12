import numpy as np


class He:
    def __init__(self) -> None:
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt((2/(fan_in))), size=(weights_shape))