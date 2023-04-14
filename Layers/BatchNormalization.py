from Layers.Base import BaseLayer
import numpy as np
from logger import get_file_logger
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

_logger = get_file_logger(__name__)


class BatchNorm2d(BaseLayer):
    def __init__(self, num_features, eps=1e-11, momentum=0.8):
        super().__init__()
        self.trainable = True

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.mean = np.zeros(self.num_features)
        self.variance = np.ones(self.num_features)

        self.test_phase = False
        self.input_tensor = None

    def reshape_tensor_for_input(self, tensor):
        return np.concatenate(tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]), axis=1).T

    def reshape_tensor_for_output(self, tensor):
        return np.transpose(
            tensor.reshape(self.input_tensor.shape[0], self.input_tensor.shape[2] * self.input_tensor.shape[3],
                           self.input_tensor.shape[1]), (0, 2, 1)).reshape(self.input_tensor.shape[0],
                                                                           self.input_tensor.shape[1],
                                                                           self.input_tensor.shape[2],
                                                                           self.input_tensor.shape[3])

    def normalize_train(self, tensor):
        self.batch_mean = np.mean(tensor, axis=0)
        self.batch_variance = np.var(tensor, axis=0)

        input_tensor_normalized = (
            tensor - self.batch_mean)/np.sqrt(self.batch_variance + self.eps)

        input_normalized = self.gamma * input_tensor_normalized + self.beta

        if np.all(self.mean == 0):
            self.mean = self.batch_mean
            self.variance = self.batch_variance
        else:
            self.mean = self.momentum * self.mean + \
                (1 - self.momentum) * self.batch_mean
            self.variance = self.momentum * self.variance + \
                (1 - self.momentum) * self.batch_variance
        return input_normalized

    def normalize_test(self, tensor):
        input_tensor_normalized = (
            tensor - self.mean)/np.sqrt(self.variance + self.eps)
        input_normalized = self.gamma * input_tensor_normalized + self.beta
        return input_normalized

    def forward(self, input_tensor):  # input shape BATCHxCHANNELSxHEIGHTxWIDTH
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]

        input_tensor_reshaped = self.reshape_tensor_for_input(input_tensor)

        if not self.test_phase:
            self.input_normalized = self.normalize_train(input_tensor_reshaped)
        else:
            self.input_normalized = self.normalize_test(input_tensor_reshaped)

        self.forward_output_reshaped = self.reshape_tensor_for_output(
            self.input_normalized)
        return self.forward_output_reshaped
    
    def backward(self, error_tensor):
        error_tensor = self.reshape_tensor_for_input(error_tensor)

        der_gamma = np.sum(error_tensor * self.input_normalized, axis=0)
        der_beta = error_tensor.sum(axis=0)
        der_input_normalized = error_tensor * self.gamma

        der_input_tensor = 1/self.batch_size / np.sqrt(self.batch_variance) * (self.batch_size * der_input_normalized - 
                      der_input_normalized.sum(axis=0) - 
                      self.input_normalized * (der_input_normalized * self.input_normalized).sum(axis=0))    

        der_input_normalized = self.reshape_tensor_for_output(der_input_normalized )
        return der_input_normalized


if __name__ == "__main__":
    import NeuralNetworkTests

    NeuralNetworkTests.TestBatchNorm().setUp()
    NeuralNetworkTests.TestBatchNorm()
