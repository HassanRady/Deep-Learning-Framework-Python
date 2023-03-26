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

        self.input_channels = self.convolution_shape[0]
        self.output_channels = num_kernels
        self.kernel_size = self.convolution_shape[1:]

        self.weight_shape = (
            self.output_channels, self.input_channels, *self.kernel_size)
        self.weights = np.random.randn(*self.weight_shape)
        self.bias = np.random.randn(self.output_channels, 1)


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

    def get_shape_after_conv(self, x, f, p=0, s=1) -> int:
        return 1 + (x - f + 2*p)/s
    
    def pad_img(img, dim1, dim2):
        return np.pad(img, ((dim1, dim1), (dim2, dim2)), mode="constant")

    def pad_1d(x, start, end):
        return np.pad(x, (start, end), mode="constant")

    def get_num_of_pad_needed_to_same_img(self, dim_len, kernel_size, stride):
        return (dim_len*(stride - 1) + kernel_size - stride)//2
    
    def convolve(self, slice, kernel, bias):
        return np.sum(slice * kernel) + bias 

    def get_dims_for_output_img(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        input_size_dim1 = input_tensor.shape[2]
        input_size_dim2 = input_tensor.shape[3]
        kernel_size_dim1 = self.kernel_size[0]
        kernel_size_dim2 = self.kernel_size[1]
        stride_size_dim1 = self.stride_shape[0]
        stride_size_dim2 = self.stride_shape[1]

        return input_size_dim1, input_size_dim2, kernel_size_dim1, kernel_size_dim2, stride_size_dim1, stride_size_dim2

    def get_output_shape_for_img(self, input_size_dim1, input_size_dim2, kernel_size_dim1, kernel_size_dim2, stride_size_dim1, stride_size_dim2, pad_size_dim1, pad_size_dim2):
        
        output_dim1 = self.get_shape_after_conv(input_size_dim1, kernel_size_dim1, pad_size_dim1, stride_size_dim1)
        output_dim2 = self.get_shape_after_conv(input_size_dim2, kernel_size_dim2, pad_size_dim2, stride_size_dim2)

        return (self.N, self.output_channels, output_dim1, output_dim2)

    def forward(self, input_tensor: np.array):  # input shape BATCHxCHANNELSxHIGHTxWIDTH
        input_size_dim1, input_size_dim2, kernel_size_dim1, kernel_size_dim2, stride_size_dim1, stride_size_dim2 = self.get_dims_for_output_img(input_tensor)
        pad_size_dim1 = self.get_num_of_pad_needed_to_same_img(input_size_dim1, kernel_size_dim1, stride_size_dim1)
        pad_size_dim2 = self.get_num_of_pad_needed_to_same_img(input_size_dim2, kernel_size_dim2, stride_size_dim2)
        output_shape = self.get_output_shape_for_img(input_size_dim1, input_size_dim2, kernel_size_dim1, kernel_size_dim2, stride_size_dim1, stride_size_dim2, pad_size_dim1, pad_size_dim2)
        
        output = np.zeros(output_shape)

        for n in range(self.N):
            one_sample = input_tensor[n]
            input_in_channels = one_sample.shape[0]
            one_sample_padded = self.pad_img(one_sample, pad_size_dim1, pad_size_dim2)

            for out_channel in range(self.output_channels):
                kernel = self.weights[out_channel]

                for in_channel in range(input_in_channels):
                    kernel_for_channel = kernel[in_channel]





        return self.output

    def backward(self, error_tensor):
        return error_tensor

