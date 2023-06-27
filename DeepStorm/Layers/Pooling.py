from DeepStorm.logger import get_file_logger
import numpy as np
from DeepStorm.Layers.Base import BaseLayer
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


_logger = get_file_logger(__name__, 'logs')


class MaxPool2d(BaseLayer):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.trainable = False
        self.initializable = False

        if isinstance(kernel_size, int):
            kernel_size = self.to_tuple(kernel_size)
        self.kernel_size = kernel_size
        self.kernel_size_dim1 = self.kernel_size[0]
        self.kernel_size_dim2 = self.kernel_size[1]

        if isinstance(stride, int):
            stride = self.to_tuple(stride)
        self.stride_shape = stride
        self.stride_size_dim1 = self.stride_shape[0]
        self.stride_size_dim2 = self.stride_shape[1]

    def to_tuple(self, int_value):
        return (int_value, int_value)

    def get_shape_after_pooling(self, dim_size, kernel_size, stride) -> int:
        return 1 + (dim_size - kernel_size)//stride

    def get_output_shape_for_img(self, input_size_dim1, input_size_dim2):
        output_dim1 = self.get_shape_after_pooling(
            input_size_dim1, self.kernel_size_dim1, self.stride_size_dim1)
        output_dim2 = self.get_shape_after_pooling(
            input_size_dim2, self.kernel_size_dim2, self.stride_size_dim2)

        return (self.batch_size, self.output_channels, output_dim1, output_dim2)

    def generate_slice(self, image, output_dim1, output_dim2):
        for i in range(output_dim1):
            for j in range(output_dim2):
                start_dim1 = i * self.stride_size_dim1
                end_dim1 = start_dim1 + self.kernel_size_dim1
                start_dim2 = j * self.stride_size_dim2
                end_dim2 = start_dim2 + self.kernel_size_dim2
                slice = image[start_dim1:end_dim1, start_dim2:end_dim2]
                yield slice, i, j, start_dim1, end_dim1, start_dim2, end_dim2

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        self.batch_size, self.output_channels, input_size_dim1, input_size_dim2 = input_tensor.shape

        self.forward_output_shape = self.get_output_shape_for_img(
            input_size_dim1, input_size_dim2)
        (_, _, output_size_dim1, output_size_dim2) = self.forward_output_shape

        output = np.zeros(self.forward_output_shape)

        for n in range(self.batch_size):
            for channel in range(self.output_channels):
                for slice, i, j, start_dim1, end_dim1, start_dim2, end_dim2 in self.generate_slice(self.input_tensor[n, channel], output_size_dim1, output_size_dim2):
                    output[n, channel, i, j] = np.max(slice)

        return output

    def backward(self, error_tensor):
        batch_size, output_channels, input_size_dim1, input_size_dim2 = error_tensor.shape
        input_gradient = np.zeros_like(self.input_tensor)

        for n in range(batch_size):
            for channel in range(output_channels):
                for slice, i, j, start_dim1, end_dim1, start_dim2, end_dim2 in self.generate_slice(self.input_tensor[n, channel], input_size_dim1, input_size_dim2):

                    mask = slice == np.max(slice)
                    input_gradient[n, channel, start_dim1:end_dim1,
                                   start_dim2:end_dim2] += mask * error_tensor[n, channel, i, j]
        return input_gradient
