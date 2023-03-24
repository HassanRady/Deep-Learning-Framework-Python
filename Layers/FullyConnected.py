from Layers.Base import BaseLayer
import numpy as np

from logger import get_file_logger
_logger = get_file_logger(__name__)
class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.N = 0
        self.trainable = True
        self._optimizer = None

        self.input_size = input_size
        self.output_size = output_size

        self.gradient_weights = None

        self.weights = np.random.rand(input_size+1, output_size)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1] = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        self.weights[1:] = bias_initializer.initialize( (self.input_size, self.output_size), 1, self.output_size)

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, v):
        self._optimizer = v


    def forward(self, X): # X is nXm W is mXm`
        self.N = X.shape[0]

        self.input = np.concatenate((X, np.ones((self.N, 1))), axis=1)

        self.output = self.input.dot(self.weights) 
 
        return self.output
    
    def backward(self, y):
        self.gradient_weights = np.dot(self.input.T, y)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return y.dot(self.weights.T)[:, :-1]
        