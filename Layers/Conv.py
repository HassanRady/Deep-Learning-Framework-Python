from Layers.Base import BaseLayer
import numpy as np
from scipy import signal

from logger import get_file_logger
_logger = get_file_logger(__name__)


class Conv(BaseLayer):
    def __init__(self, stride_shape: int or tuple, convolution_shape: list, num_kernels: int):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape

        self.batch_size = 1

        self.input_channels = convolution_shape[0]
        self.output_channels = num_kernels
        self.kernel_size = convolution_shape[1:]

        self.weight_shape = (
            num_kernels, self.input_channels, *self.kernel_size)
        self.weights = np.random.randn(*self.weight_shape)
        self.bias = np.random.randn(self.output_channels, 1)

        self.is_1d_input = len(convolution_shape) < 4  

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weight_shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.output_channels)
        self.bias = bias_initializer.initialize(
            self.output_channels, self.input_channels, self.output_channels)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, v):
        self._optimizer = v

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def get_shape_after_conv(self, x, f, p=1, s=1) -> int:
        return 1 + (x - f + 2*p)/s
    
    def pad_img(img):
        pass

    
    def convolve(self, slice, kernel, bias):
        np.sum(slice * kernel) + bias 

    def forward(self, input_tensor: np.array):  # input shape BXCXHXW
        self.N = len(input_tensor)
        in_channels = input_tensor[1]




        return self.output

    def backward(self, error_tensor):
        return error_tensor

