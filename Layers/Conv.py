import numpy as np
from scipy import signal

from Layers.Base import BaseLayer
from Layers.Initializers import UniformRandom
class Conv(BaseLayer):
    def __init__(self, stride_shape: int or tuple, convolution_shape: list, num_kernels: int):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        
        self.batch_size = 1

        self.input_channels = convolution_shape[0]
        self.output_channels = num_kernels
        self.kernel_size = convolution_shape[1]

        self.weight_shape = (num_kernels, *convolution_shape)
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, v):
        self._optimizer = v

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weight_shape, np.prod(self.convolution_shape),
                                                       np.prod(self.convolution_shape) * self.output_channels)
        self.bias = bias_initializer.initialize(self.output_channels, self.input_channels, self.output_channels)

    def get_shape_after_conv(self, x, f, p=1, s=1) -> int:
        return 1 + (x - f + 2*p)/s

    def forward(self, input_tensor: np.array): # input shape BXCXHXW
        self.batch_size = input_tensor[0]
        input_height = input_tensor[2]
        input_width = input_tensor[3]

        self.output_shape = (self.batch_size, self.output_channels,
                              self.get_shape_after_conv(input_height, self.kernel_size),
                              self.get_shape_after_conv(input_width, self.kernel_size))
        

        self.output = np.copy(self.bias)
        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(input_tensor[:, j, :, :], self.weights[i, j], "valid")


        return input_tensor
    
    def backward(self, error_tensor):
        return error_tensor


