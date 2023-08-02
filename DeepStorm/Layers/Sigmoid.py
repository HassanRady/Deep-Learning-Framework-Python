from DeepStorm.Layers.Base import BaseLayer
import numpy as np


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        """
        Sigmoid activation function.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, y):
        """
        Backward pass for the sigmoid activation function.

        Parameters:
            grad_output: Gradient of the loss with respect to the output of the sigmoid.
            cached_output: Output of the sigmoid during the forward pass.

        Returns:
            Gradient of the loss with respect to the input of the sigmoid.
        """
        sigmoid_grad = self.output * (1 - self.output)
        return y * sigmoid_grad

