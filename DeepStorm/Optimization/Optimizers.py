from DeepStorm.logger import get_file_logger
import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


_logger = get_file_logger(__name__, 'debug')


class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = 0

    def calculate_update(self, weight_tensor, weight_gradient):
        self.v = self.momentum * self.v - self.learning_rate * weight_gradient
        weight_tensor = weight_tensor + self.v
        return weight_tensor


class Adam:
    def __init__(self, learning_rate=1e-03, mu=0.9, rho=0.9, epsilon=1e-07) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.epsilon = epsilon
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, weight_gradient):
        self.k = self.k + 1

        self.v = self.mu * self.v + (1 - self.mu) * weight_gradient
        self.r = self.rho * self.r + \
            (1 - self.rho) * np.square(weight_gradient)

        self.v_hat = (self.v)/(1 - self.mu**self.k)
        self.r_hat = (self.r)/(1 - self.rho**self.k)

        weight_tensor = weight_tensor - self.learning_rate * \
            (self.v_hat)/(np.sqrt(self.r_hat) + self.epsilon)
        return weight_tensor
