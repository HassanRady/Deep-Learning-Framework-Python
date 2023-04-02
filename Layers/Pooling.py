from logger import get_file_logger
import numpy as np
from Layers.Base import BaseLayer
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


_logger = get_file_logger(__name__)

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.pooling_shape = pooling_shape
        self.pooling_size_dim1 = self.pooling_shape[0]
        self.pooling_size_dim2 = self.pooling_shape[1]

        self.stride_shape = stride_shape
        self.stride_size_dim1 = self.stride_shape[0]
        self.stride_size_dim2 = self.stride_shape[1]

    def get_shape_after_pooling(self, dim_size, kernel_size, stride) -> int:
        return 1 + (dim_size - kernel_size)//stride

    def get_output_shape_for_img(self, input_size_dim1, input_size_dim2):
        output_dim1 = self.get_shape_after_pooling(
            input_size_dim1, self.pooling_size_dim1, self.stride_size_dim1)
        output_dim2 = self.get_shape_after_pooling(
            input_size_dim2, self.pooling_size_dim2, self.stride_size_dim2)

        return (self.batch_size, self.output_channels, output_dim1, output_dim2)

    def generate_slice(self, image, output_dim1, output_dim2):
        for i in range(output_dim1):
            for j in range(output_dim2):
                start_dim1 = i * self.stride_size_dim1
                end_dim1 = i * self.stride_size_dim1 + self.pooling_size_dim1
                start_dim2 = j * self.stride_size_dim2
                end_dim2 = j * self.stride_size_dim2 + self.pooling_size_dim2
                slice = image[:, start_dim1:end_dim1, start_dim2:end_dim2]
                yield slice, i, j

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size, self.output_channels, input_size_dim1, input_size_dim2 = input_tensor.shape
        self.forward_output_shape = self.get_output_shape_for_img(input_size_dim1, input_size_dim2)
        (_, _, output_size_dim1, output_size_dim2) = self.forward_output_shape
        output = np.zeros(self.forward_output_shape)

        for n in range(self.batch_size):
            for channel in range(self.output_channels):
                for slice, i, j in self.generate_slice(input_tensor[n], output_size_dim1, output_size_dim2):
                    output[n, channel, i, j] = np.max(slice)

        return output
            


    def backward(self, error_tensor):
        return error_tensor
