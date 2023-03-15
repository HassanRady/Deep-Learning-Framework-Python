class Sgd:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor =  weight_tensor - self.learning_rate * gradient_tensor
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
    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

    def calculate_update(self, weight_tensor, weight_gradient):
        return 0