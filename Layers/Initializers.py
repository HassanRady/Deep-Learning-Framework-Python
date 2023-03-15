import numpy as np

class Constant:
    def __init__(self, value) -> None:
        self.value = value
    
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.zeros(weights_shape) + self.value
        return weights


















class UniformRandom:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.rand(weights_shape)
        return weights

class Xavier:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return 0

class He:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return 0
