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


class Conv(BaseLayer):
    def __init__(self, stride_shape: int or tuple, convolution_shape: list, num_kernels: int):
        super().__init__()
        self.trainable = True
        self._optimizer = None

        self.stride_shape = stride_shape
        self.stride_size_dim1 = self.stride_shape[0]
        self.stride_size_dim2 = self.stride_shape[1]

        self.convolution_shape = convolution_shape

        self.input_channels = self.convolution_shape[0]
        self.output_channels = num_kernels
        self.kernel_size = self.convolution_shape[1:]
        self.kernel_size_dim1 = self.kernel_size[0]
        self.kernel_size_dim2 = self.kernel_size[1]
        self.is_reduction_kernel = self.kernel_size == (1, 1)

        self.pad_size_dim1 = self.get_pad_size(self.kernel_size_dim1)
        self.pad_size_dim2 = self.get_pad_size(self.kernel_size_dim2)

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

    def get_shape_after_conv(self, dim_size, kernel_size, pad, stride) -> int:
        (start_pad, end_pad) = pad
        return 1 + (dim_size - kernel_size + start_pad + end_pad)//stride

    def get_pad_size(self, kernel_size):
        # if its an odd kernel
        if kernel_size % 2 == 1:
            start_pad = (kernel_size - 1)//2
            return (start_pad, start_pad)
        # else it is even
        start_pad = kernel_size//2 - 1
        end_pad = kernel_size//2
        return (start_pad, end_pad)

    def pad_img_same(self, imgs, dim1, dim2):
        (start_pad_dim1, end_pad_dim1) = dim1
        (start_pad_dim2, end_pad_dim2) = dim2
        return np.pad(imgs, ((0, 0), (0, 0), (start_pad_dim1, end_pad_dim1), (start_pad_dim2, end_pad_dim2)), mode="constant")

    def remove_pad(self, imgs):
        (start_pad_dim1, end_pad_dim1) = self.pad_size_dim1
        (start_pad_dim2, end_pad_dim2) = self.pad_size_dim2
        if self.is_reduction_kernel:
            return imgs
        return imgs[:, start_pad_dim1:-end_pad_dim1, start_pad_dim2:-end_pad_dim2]

    def convolve(self, slice, kernel, bias):
        return np.sum(slice * kernel) + bias

    def get_output_shape_for_img(self, input_size_dim1, input_size_dim2):

        output_dim1 = self.get_shape_after_conv(
            input_size_dim1, self.kernel_size_dim1, self.pad_size_dim1, self.stride_size_dim1)
        output_dim2 = self.get_shape_after_conv(
            input_size_dim2, self.kernel_size_dim2, self.pad_size_dim2, self.stride_size_dim2)

        return (self.batch_size, self.output_channels, output_dim1, output_dim2)

    def generate_slice(self, image, output_dim1, output_dim2):
        for i in range(output_dim1):
            for j in range(output_dim2):
                start_dim1 = i * self.stride_size_dim1
                end_dim1 = i * self.stride_size_dim1 + self.kernel_size_dim1
                start_dim2 = j * self.stride_size_dim2
                end_dim2 = j * self.stride_size_dim2 + self.kernel_size_dim2
                slice = image[:, start_dim1:end_dim1, start_dim2:end_dim2]
                yield slice, i, j

    def forward(self, input_tensor: np.array):  # input shape BATCHxCHANNELSxHIGHTxWIDTH
        self.input_tensor = input_tensor
        (self.batch_size, _, input_size_dim1, input_size_dim2) = input_tensor.shape
        self.input_tensor_padded = self.pad_img_same(
            input_tensor, self.pad_size_dim1, self.pad_size_dim2)

        self.forward_output_shape = self.get_output_shape_for_img(
            input_size_dim1, input_size_dim2)
        (_, _, output_dim1, output_dim2) = self.forward_output_shape
        self.forward_output = np.zeros(self.forward_output_shape)

        for n in range(self.batch_size):
            one_sample_padded = self.input_tensor_padded[n]
            for out_channel in range(self.output_channels):
                kernel = self.weights[out_channel]
                bias = self.bias[out_channel]

                for slice, i, j in self.generate_slice(one_sample_padded, output_dim1, output_dim2):
                    self.forward_output[n, out_channel, i,
                                        j] = self.convolve(slice, kernel, bias)

        return self.forward_output

    def backward(self, error_tensor):
        (_, _, output_dim1, output_dim2) = self.forward_output_shape

        output = np.zeros_like(self.input_tensor)
        gradient_input = np.zeros_like(self.input_tensor_padded)
        self._gradient_bias = np.zeros_like(self.bias)
        self._gradient_weights = np.zeros_like(self.weights)

        for n in range(self.batch_size):
            one_sample_padded = self.input_tensor_padded[n]

            for out_channel in range(self.output_channels):
                for slice, i, j in self.generate_slice(one_sample_padded, output_dim1, output_dim2):
                    start_dim1 = i * self.stride_size_dim1
                    end_dim1 = i * self.stride_size_dim1 + self.kernel_size_dim1
                    start_dim2 = j * self.stride_size_dim2
                    end_dim2 = j * self.stride_size_dim2 + self.kernel_size_dim2

                    self._gradient_weights[out_channel] += slice * \
                        error_tensor[n, out_channel, i, j]
                    self._gradient_bias[out_channel] += error_tensor[n,
                                                                     out_channel, i, j]
                    gradient_input[n, :, start_dim1:end_dim1, start_dim2:end_dim2] += error_tensor[n,
                                                                                                   out_channel, i, j] * self.weights[out_channel]

            output[n] = self.remove_pad(gradient_input[n])

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(
                self.weights, self.gradient_weights)

        return output


if __name__ == "__main__":

    from NeuralNetworkTests import TestConv

    tests = TestConv()
    tests.setUp()
    tests.test_forward_size_stride_uneven_image()
