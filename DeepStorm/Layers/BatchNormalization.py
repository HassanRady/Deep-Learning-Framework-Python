from DeepStorm.logger import get_file_logger
from DeepStorm.Layers.Base import BaseLayer
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

_logger = get_file_logger(__name__, 'logs')


class BatchNorm2d(BaseLayer):
    def __init__(self, num_features, eps=1e-11, momentum=0.8):
        super().__init__()
        self.trainable = True

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = np.ones(num_features)
        self.bias = np.zeros(num_features)

        self.mean = np.zeros(self.num_features, dtype=np.float64)
        self.variance = np.ones(self.num_features)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, v):
        self._optimizer = v

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    def normalize_train(self, tensor):
        batch_mean = np.mean(tensor, axis=(0, 2, 3), keepdims=False)
        batch_variance = np.var(tensor, axis=(0, 2, 3), keepdims=False)
        batch_std = np.sqrt(batch_variance + self.eps)
        n = tensor.size / tensor.shape[1]

        self.mean = self.momentum * batch_mean + \
            (1 - self.momentum) * self.mean
        self.variance = self.momentum * batch_variance * n / (n - 1) + \
            (1 - self.momentum) * self.variance

        self.input_tensor_normalized = (
            tensor - batch_mean[None, :, None, None])/batch_std[None, :, None, None]
        input_batch_normalized = self.weight[None, :, None, None] * \
            self.input_tensor_normalized + self.bias[None, :, None, None]
        return input_batch_normalized

    def normalize_test(self, tensor):
        input_tensor_normalized = (
            tensor - self.mean[None, :, None, None])/np.sqrt(self.variance[None, :, None, None] + self.eps)
        input_normalized = self.weight[None, :, None, None] * \
            input_tensor_normalized + self.bias[None, :, None, None]
        return input_normalized

    def forward(self, input_tensor):  # input shape BATCHxCHANNELSxHEIGHTxWIDTH
        self.batch_size = input_tensor.shape[0]

        if self.training:
            self.input_normalized = self.normalize_train(input_tensor)
        else:
            self.input_normalized = self.normalize_test(input_tensor)

        return self.input_normalized

    def backward(self, error_tensor):

        gradient_weight = np.sum(
            error_tensor * self.input_tensor_normalized, axis=(0, 2, 3))
        gradient_bias = error_tensor.sum(axis=(0, 2, 3))
        gradient_input_normalized = error_tensor * \
            self.weight[None, :, None, None]

        der_input_tensor = 1./(self.batch_size * np.sqrt(self.variance[None, :, None, None] + self.eps)) * (self.batch_size * gradient_input_normalized -
                                                                                                            gradient_input_normalized.sum(axis=0) -
                                                                                                            self.input_tensor_normalized * np.sum(gradient_input_normalized * self.input_tensor_normalized, axis=0))
        if self.optimizer:
            self.weight = self.optimizer.calculate_update(
                self.weight, gradient_weight)
            self.bias = self.optimizer.calculate_update(
                self.bias, gradient_bias)

        return der_input_tensor


if __name__ == "__main__":
    import NeuralNetworkTests

    NeuralNetworkTests.TestBatchNorm().setUp()
    NeuralNetworkTests.TestBatchNorm()
