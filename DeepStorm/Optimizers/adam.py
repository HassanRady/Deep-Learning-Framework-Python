import numpy as np

from DeepStorm.logger import get_file_logger


_logger = get_file_logger(__name__, 'debug')


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