import unittest
import numpy as np
from scipy.ndimage import gaussian_filter

from DeepStorm.Layers.conv import Conv2d


class TestConv2d(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class DummyWeightInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    class DummyBiasInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            return np.array([0])

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.in_channels = 3
        self.kernel_shape = (5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 105
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = Conv2d(in_channels=self.in_channels, stride=(
            1, 1), kernel_size=self.kernel_shape, out_channels=self.num_kernels, padding="same")
        self.assertTrue(layer.trainable)

    def test_forward_size(self):
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                      kernel_size=self.kernel_shape, stride=(1, 1), padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size,
                         self.num_kernels, *self.input_shape[1:]))

    def test_forward_size_stride(self):
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                      kernel_size=self.kernel_shape, stride=(3, 2), padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.num_kernels, 4, 7))

    def test_forward_size_stride2(self):
        conv = Conv2d(in_channels=self.in_channels, stride=(
            3, 3), kernel_size=self.kernel_shape, out_channels=self.num_kernels, padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.num_kernels, 4, 5))

    def test_forward_size_same(self):
        conv = Conv2d(stride=(1, 1), kernel_size=self.kernel_shape,
                      out_channels=self.num_kernels, in_channels=self.in_channels, padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.num_kernels, 10, 14))

    def test_forward_size_stride_uneven_image(self):
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels +
                      1, kernel_size=self.kernel_shape, stride=(3, 2), padding="same")
        input_tensor = np.array(range(
            int(np.prod(self.uneven_input_shape) * (self.batch_size + 1))), dtype=float)
        input_tensor = input_tensor.reshape(
            self.batch_size + 1, *self.uneven_input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size+1, self.num_kernels+1, 4, 8))

    def test_forward(self):
        np.random.seed(1337)
        conv = Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(3, 3), stride=(1, 1), padding="same")
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(
            input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_multi_channel(self):
        np.random.seed(1337)
        maps_in = 2
        bias = 1
        conv = Conv2d(in_channels=maps_in, out_channels=1,
                      kernel_size=3, stride=1, padding="same")
        kernel = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.weights = np.repeat(kernel[None, ...], maps_in, axis=1)
        conv.bias = np.array([bias])
        input_tensor = np.random.random((1, maps_in, 10, 14))
        expected_output = bias
        for map_i in range(maps_in):
            expected_output = expected_output + \
                gaussian_filter(
                    input_tensor[0, map_i, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
        self.assertAlmostEqual(difference, 0., places=1)

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = Conv2d(in_channels=3, out_channels=1,
                      kernel_size=3, stride=1, padding="same")
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [
            [1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3, 10, 14))
        input_tensor[:, 0] = tensor.copy()
        input_tensor[:, 1] = tensor.copy()
        input_tensor[:, 2] = tensor.copy()
        expected_output = 3 * \
            gaussian_filter(
                input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2)

    def test_backward_size(self):
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                      kernel_size=self.kernel_shape, stride=1, padding="same")
        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape,
                         (self.batch_size, *self.input_shape))

    def test_backward_size_stride(self):
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                      kernel_size=self.kernel_shape, stride=(3, 2), padding="same")

        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape,
                         (self.batch_size, *self.input_shape))

    def test_1x1_convolution(self):
        conv = Conv2d(in_channels=3, out_channels=self.num_kernels,
                      kernel_size=(1, 1), stride=1, padding="same")

        input_tensor = np.array(
            range(self.input_size * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size,
                         self.num_kernels, *self.input_shape[1:]))
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape,
                         (self.batch_size, *self.input_shape))

    def test_layout_preservation(self):
        conv = Conv2d(in_channels=3, out_channels=1,
                      kernel_size=3, stride=1, padding='same', weights_initializer=self.DummyWeightInitializer(), bias_initializer=self.DummyBiasInitializer())
        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(
            np.sum(np.abs(np.squeeze(output_tensor) - input_tensor[:, 1, :, :])), 0.)