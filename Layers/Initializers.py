import numpy as np

class Constant:
    def __init__(self, value=0.1) -> None:
        self.value = value
    
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.zeros(weights_shape) + self.value
        return weights



class UniformRandom:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, ):
        weights = np.random.rand(*weights_shape)
        return weights



class Xavier:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt( (2) / (fan_in + fan_out) )
        weights = np.random.normal(0, sigma, size=weights_shape)
        return weights



class He:
    def __init__(self) -> None:
        pass
    
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt( (2) / (fan_in) )
        weights = np.random.normal(0, sigma, size=weights_shape)
        return weights
