import unittest
from DeepStorm.Layers import *
from DeepStorm.Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import tabulate
import argparse

ID = 2  # identifier for dispatcher


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

    def test_trainable(self):
        layer = FullyConnected.Linear(
            self.input_size, self.output_size)
        self.assertTrue(layer.trainable)

    def test_forward_size(self):
        layer = FullyConnected.Linear(
            self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = FullyConnected.Linear(
            self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected.Linear(
            self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)),
                            np.sum(np.power(new_output_tensor, 2)))

    def test_update_bias(self):
        input_tensor = np.zeros([self.batch_size, self.input_size])
        layer = FullyConnected.Linear(
            self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)),
                            np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(
            (self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.Linear(
            self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random(
            (self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.Linear(
            self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(
            layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.Linear(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

    def test_initialization(self):
        input_size = 4
        categories = 10
        layer = FullyConnected.Linear(input_size, categories, TestLinear.TestInitializer(), Initializers.Constant(0.5))
        init = TestLinear.TestInitializer()
        # self.assertEqual(layer.weights_initializer, input_size)
        # self.assertEqual(init.fan_out, categories)
        if layer.weights.shape[0] > layer.weights.shape[1]:
            self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5)
        else:
            self.assertLessEqual(np.sum(layer.weights) - 35, 1e-5)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size, :] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = ReLU.ReLU()
        self.assertFalse(layer.trainable)

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(
            np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(
            (self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = SoftMax.SoftMax()
        self.assertFalse(layer.trainable)

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax.SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)))

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10)

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12)

    def test_regression_backward_high_loss_w_CrossEntropy(self):
        input_tensor = self.label_tensor - 1
        input_tensor *= -10.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertAlmostEqual(element, 1/3, places=3)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1, places=3)

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.)

    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax.SoftMax())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.categories, self.batch_size))
        # print(input_tensor)
        layer = SoftMax.SoftMax()
        prediction = layer.forward(input_tensor.T)
        # print(prediction)
        expected_values = np.array([[0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724,
                                     0.21732724, 0.21732724],
                                    [0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387,
                                     0.23779387, 0.23779387],
                                    [0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794,
                                     0.26018794, 0.26018794],
                                    [0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095,
                                     0.28469095, 0.28469095]])
        # print(expected_values)
        # print(prediction)
        np.testing.assert_almost_equal(expected_values, prediction.T)


class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(Loss.CrossEntropyLoss())
        difference = Helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0)

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places=4)


class TestOptimizers2(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]))

    def test_sgd_with_momentum(self):
        optimizer = Optimizers.SgdWithMomentum(1., 0.9)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.9]))

    def test_adam(self):
        optimizer = Optimizers.Adam(1., 0.01, 0.02)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))

        result = optimizer.calculate_update(result, .5)
        np.testing.assert_almost_equal(result, np.array([-0.9814473195614205]))


class TestInitializers(unittest.TestCase):
    class DummyLayer:
        def __init__(self, input_size, output_size):
            self.weights = []
            self.shape = (output_size, input_size)

        def initialize(self, initializer):
            self.weights = initializer.initialize(
                self.shape, self.shape[1], self.shape[0])

    def setUp(self):
        self.batch_size = 9
        self.input_size = 400
        self.output_size = 400
        self.num_kernels = 20
        self.num_channels = 20
        self.kernelsize_x = 41
        self.kernelsize_y = 41

    def _performInitialization(self, initializer):
        np.random.seed(1337)
        layer = TestInitializers.DummyLayer(self.input_size, self.output_size)
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return layer.shape, weights_after_init

    def test_uniform_shape(self):
        weights_shape, weights_after_init = self._performInitialization(
            Initializers.UniformRandom())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_uniform_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(
            Initializers.UniformRandom())

        p_value = stats.kstest(weights_after_init.flat,
                               'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_xavier_shape(self):
        weights_shape, weights_after_init = self._performInitialization(
            Initializers.Xavier())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_xavier_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(
            Initializers.Xavier())

        scale = np.sqrt(2) / np.sqrt(self.input_size + self.output_size)
        p_value = stats.kstest(weights_after_init.flat,
                               'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_he_shape(self):
        weights_shape, weights_after_init = self._performInitialization(
            Initializers.He())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_he_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(
            Initializers.He())

        scale = np.sqrt(2) / np.sqrt(self.input_size)
        p_value = stats.kstest(weights_after_init.flat,
                               'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)


class TestFlatten(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.input_shape = (3, 4, 11)
        self.input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        self.input_tensor = self.input_tensor.reshape(
            self.batch_size, *self.input_shape)

    def test_trainable(self):
        layer = Flatten.Flatten()
        self.assertFalse(layer.trainable)

    def test_flatten_forward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        input_vector = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_vector = input_vector.reshape(
            self.batch_size, np.prod(self.input_shape))
        self.assertLessEqual(np.sum(np.abs(output_tensor-input_vector)), 1e-9)

    def test_flatten_backward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        backward_tensor = flatten.backward(output_tensor)
        self.assertLessEqual(
            np.sum(np.abs(self.input_tensor - backward_tensor)), 1e-9)


class TestConv2d(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

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
        layer = Conv.Conv2d(in_channels=self.in_channels, stride=(
            1, 1), kernel_size=self.kernel_shape, out_channels=self.num_kernels, padding="same")
        self.assertTrue(layer.trainable)

    def test_forward_size(self):
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                           kernel_size=self.kernel_shape, stride=(1, 1), padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size,
                         self.num_kernels, *self.input_shape[1:]))

    def test_forward_size_stride(self):
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                           kernel_size=self.kernel_shape, stride=(3, 2), padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.num_kernels, 4, 7))

    def test_forward_size_stride2(self):
        conv = Conv.Conv2d(in_channels=self.in_channels, stride=(3, 3), kernel_size=self.kernel_shape, out_channels=self.num_kernels, padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.num_kernels, 4, 5))

    def test_forward_size_same(self):
        conv = Conv.Conv2d(stride=(1, 1), kernel_size=self.kernel_shape, out_channels=self.num_kernels, in_channels=self.in_channels, padding="same")
        input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,
                         (self.batch_size, self.num_kernels, 10, 14))

    def test_forward_size_stride_uneven_image(self):
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels +
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
        conv = Conv.Conv2d(in_channels=1, out_channels=1,
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
        conv = Conv.Conv2d(in_channels=maps_in, out_channels=1,
                           kernel_size=3, stride=1, padding="same")
        filter = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.weights = np.repeat(filter[None, ...], maps_in, axis=1)
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
        conv = Conv.Conv2d(in_channels=3, out_channels=1,
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
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                           kernel_size=self.kernel_shape, stride=1, padding="same")
        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape,
                         (self.batch_size, *self.input_shape))

    def test_backward_size_stride(self):
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                           kernel_size=self.kernel_shape, stride=(3, 2), padding="same")

        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape,
                         (self.batch_size, *self.input_shape))

    def test_1x1_convolution(self):
        conv = Conv.Conv2d(in_channels=3, out_channels=self.num_kernels,
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
        conv = Conv.Conv2d(in_channels=3, out_channels=1,
                           kernel_size=3, stride=1, padding='same', weights_initializer=self.TestInitializer(), bias_initializer=Initializers.Constant(0.0))
        # conv.initialize(self.TestInitializer(), Initializers.Constant(0.0))
        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(
            np.sum(np.abs(np.squeeze(output_tensor) - input_tensor[:, 1, :, :])), 0.)

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv2d(in_channels=3, out_channels=self.hidden_channels,
                      kernel_size=3, stride=1, padding='same'))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 5e-2)

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv2d(in_channels=3, out_channels=self.hidden_channels,
                      kernel_size=3, stride=1, padding='same'))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(
            layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 36])
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv2d(in_channels=3, out_channels=self.hidden_channels,
                      kernel_size=3, stride=2, padding='same'))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(
            layers, input_tensor, label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv2d(in_channels=3, out_channels=self.hidden_channels,
                      kernel_size=3, stride=1, padding='same'))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(
            layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_weights_init(self):
        # simply checks whether you have not initialized everything with zeros
        conv = Conv.Conv2d(in_channels=100, out_channels=150,
                           kernel_size=10, stride=1, padding='same')
        self.assertGreater(np.mean(np.abs(conv.weights)), 1e-3)

    def test_bias_init(self):
        conv = Conv.Conv2d(in_channels=1, out_channels=150 *
                           100 * 10 * 10, kernel_size=1, stride=1, padding='same')
        self.assertGreater(np.mean(np.abs(conv.bias)), 1e-3)

    def test_gradient_stride(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 35])
        input_tensor = np.abs(np.random.random((2, 6, 5, 14)))
        layers = list()
        layers.append(Conv.Conv2d(in_channels=6, out_channels=1,
                      kernel_size=3, stride=(1, 2), padding='same'))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_update(self):
        input_tensor = np.random.uniform(-1, 1,
                                         (self.batch_size, *self.input_shape))
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                           kernel_size=self.kernel_shape, stride=(3, 2), padding='same')
        conv.optimizer = Optimizers.Sgd(1)
        conv.initialize(Initializers.He(), Initializers.Constant(0.1))
        # conv.weights = np.random.rand(4, 3, 5, 8)
        # conv.bias = 0.1 * np.ones(4)
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)),
                            np.sum(np.power(new_output_tensor, 2)))

    def _test_initialization(self):
        conv = Conv.Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                           kernel_size=self.kernel_shape, stride=(1, 1), padding='same', weights_initializer=TestConv2d.TestInitializer())
        init = TestConv2d.TestInitializer()
        # conv.initialize(init, Initializers.Constant(0.1))
        self.assertEqual(
            init.fan_in, self.kernel_shape[0] * self.kernel_shape[1] * self.in_channels)
        
        self.assertEqual(init.fan_out, np.prod(
            self.kernel_shape[:]) * self.num_kernels)


class TestMaxPool2d(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(1337)
        self.input_tensor = np.random.uniform(-1,
                                              1, (self.batch_size, *self.input_shape))

        self.categories = 12
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten.Flatten())
        self.layers.append(L2Loss())
        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def test_trainable(self):
        layer = Pooling.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.assertFalse(layer.trainable)

    def test_shape(self):
        layer = Pooling.MaxPool2d(stride=(2, 2), kernel_size=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(
            np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_overlapping_shape(self):
        layer = Pooling.MaxPool2d(stride=(2, 1), kernel_size=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(
            np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_subsampling_shape(self):
        layer = Pooling.MaxPool2d(stride=(3, 2), kernel_size=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(
            np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_gradient_stride(self):
        self.layers[0] = Pooling.MaxPool2d(stride=(2, 2), kernel_size=(2, 2))
        difference = Helpers.gradient_check(
            self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_overlapping_stride(self):
        label_tensor = np.random.random((self.batch_size, 24))
        self.layers[0] = Pooling.MaxPool2d(stride=(2, 1), kernel_size=(2, 2))
        difference = Helpers.gradient_check(
            self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_subsampling_stride(self):
        label_tensor = np.random.random((self.batch_size, 6))
        self.layers[0] = Pooling.MaxPool2d(stride=(3, 2), kernel_size=(2, 2))
        difference = Helpers.gradient_check(
            self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_layout_preservation(self):
        pool = Pooling.MaxPool2d(stride=(1, 1), kernel_size=(1, 1))
        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.)

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = Pooling.MaxPool2d(stride=(2, 2), kernel_size=(2, 2))
        batch_size = 2
        input_tensor = np.array(
            range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[4]]], [[[13]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0)

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = Pooling.MaxPool2d(stride=2, kernel_size=2)
        batch_size = 2
        input_tensor = np.array(
            range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array(
            [[[[5.,  7.], [13., 15.]]], [[[21., 23.], [29., 31.]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0)


class TestBatchNorm2d(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 200
        self.channels = 2
        self.input_shape = (self.channels, 3, 3)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(0)
        self.input_tensor = np.abs(np.random.random(
            (self.input_size, self.batch_size))).T
        self.input_tensor_conv = np.random.uniform(
            -1, 1, (self.batch_size, *self.input_shape))

        self.categories = 5
        self.label_tensor = np.zeros([self.categories, self.batch_size]).T
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten.Flatten())
        self.layers.append(FullyConnected.Linear(
            self.input_size, self.categories))
        self.layers.append(L2Loss())

        self.plot_shape = (
            self.input_shape[1], self.input_shape[0] * np.prod(self.input_shape[2:]))

    @staticmethod
    def _channel_moments(tensor, channels):

        tensor = np.transpose(tensor, (0, *range(2, tensor.ndim), 1))
        tensor = tensor.reshape(-1, channels)
        mean = np.mean(tensor, axis=0)
        var = np.var(tensor, axis=0)
        return mean, var

    def test_trainable(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        self.assertTrue(layer.trainable)

    def test_default_phase(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        self.assertFalse(layer.test_phase)

    def _test_forward_linear_shape(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward_shape_convolutional(self):
        layer = BatchNormalization.BatchNorm2d(self.channels)
        output = layer.forward(self.input_tensor_conv)

        self.assertEqual(output.shape, self.input_tensor_conv.shape)

    def _test_forward_linear(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        self.assertAlmostEqual(
            np.sum(np.square(mean - np.zeros(mean.shape[0]))), 0)
        self.assertAlmostEqual(
            np.sum(np.square(var - np.ones(var.shape[0]))), 0)

    def test_reformat_image2vec(self):
        layer = BatchNormalization.BatchNorm2d(3)
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reshape_tensor_for_input(image_tensor)
        np.testing.assert_equal(vec_tensor.shape, (120, 3))
        self.assertEqual(np.sum(vec_tensor, 1)[0], 72)
        self.assertEqual(np.sum(vec_tensor, 0)[0], 18660)

    def test_reformat_vec2image(self):
        layer = BatchNormalization.BatchNorm2d(3)
        layer.forward(np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4))
        vec_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(120, 3)
        image_tensor = layer.reshape_tensor_for_output(vec_tensor)
        np.testing.assert_equal(image_tensor.shape, (5, 3, 6, 4))
        self.assertEqual(np.sum(image_tensor, (0, 1, 2))[0], 15750)
        self.assertEqual(np.sum(image_tensor, (0, 2, 3))[0], 21420)

    def test_reformat(self):
        layer = BatchNormalization.BatchNorm2d(3)
        layer.forward(np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4))
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reshape_tensor_for_input(image_tensor)
        image_tensor2 = layer.reshape_tensor_for_output(vec_tensor)
        np.testing.assert_allclose(image_tensor, image_tensor2)

    def test_forward_convolutional(self):
        from Layers import BatchNormalization
        layer = BatchNormalization.BatchNorm2d(self.channels)
        output = layer.forward(self.input_tensor_conv)
        mean, var = TestBatchNorm2d._channel_moments(output, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean)), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones_like(var))), 0)

    def _test_forward_linear_train_phase(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertNotEqual(
            np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_train_phase_convolutional(self):
        layer = BatchNormalization.BatchNorm2d(self.channels)
        layer.forward(self.input_tensor_conv)

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm2d._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm2d._channel_moments(
            self.input_tensor_conv, self.channels)

        self.assertNotEqual(
            np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def _test_forward_linear_test_phase(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)
        layer.testing_phase = True

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertAlmostEqual(
            np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def test_forward_test_phase_convolutional(self):
        layer = BatchNormalization.BatchNorm2d(self.channels)
        layer.forward(self.input_tensor_conv)
        layer.test_phase = True

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm2d._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm2d._channel_moments(
            self.input_tensor_conv, self.channels)

        self.assertAlmostEqual(
            np.sum(np.square(mean + (mean_input / np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)

    def _test_linear_gradient(self):
        self.layers[0] = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        difference = Helpers.gradient_check(
            self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def _test_linear_gradient_weights(self):
        self.layers[0] = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        self.layers[0].forward(self.input_tensor)
        difference = Helpers.gradient_check_weights(
            self.layers, self.input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def _test_linear_gradient_bias(self):
        self.layers[0] = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        self.layers[0].forward(self.input_tensor)
        difference = Helpers.gradient_check_weights(
            self.layers, self.input_tensor, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNorm2d(self.channels)
        difference = Helpers.gradient_check(
            self.layers, self.input_tensor_conv, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_gradient_weights_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNorm2d(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = Helpers.gradient_check_weights(
            self.layers, self.input_tensor_conv, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias_convolutional(self):
        self.layers[0] = BatchNormalization.BatchNorm2d(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = Helpers.gradient_check_weights(
            self.layers, self.input_tensor_conv, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def _test_linear_update(self):
        layer = BatchNormalization.BatchNorm2d(
            self.input_tensor.shape[-1])
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros_like(self.input_tensor)
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)),
                            np.sum(np.power(new_output_tensor, 2)))


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10
        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_trainable(self):
        layer = Dropout.Dropout(0.25)
        self.assertFalse(layer.trainable)

    def test_default_phase(self):
        drop_layer = Dropout.Dropout(0.25)
        self.assertFalse(drop_layer.testing_phase)

    def test_forward_trainTime(self):
        drop_layer = Dropout.Dropout(0.25)
        output = drop_layer.forward(self.input_tensor)
        self.assertEqual(np.max(output), 4)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean/self.input_size, 1., places=1)

    def test_position_preservation(self):
        drop_layer = Dropout.Dropout(0.5)
        output = drop_layer.forward(self.input_tensor)
        error_prev = drop_layer.backward(self.input_tensor)
        np.testing.assert_almost_equal(
            np.where(output == 0.), np.where(error_prev == 0.))

    def test_forward_testTime(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.testing_phase = True
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size)

    def test_backward(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)
        self.assertEqual(np.max(output), 2)
        self.assertEqual(np.min(output), 0)

    def test_gradient(self):
        batch_size = 10
        input_size = 10
        input_tensor = np.ones((batch_size, input_size))
        label_tensor = np.zeros([batch_size, input_size])
        for i in range(batch_size):
            label_tensor[i, np.random.randint(0, input_size)] = 1
        layers = list()
        layers.append(Dropout.Dropout(0.5))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(
            layers, input_tensor, label_tensor, seed=1337)
        self.assertLessEqual(np.sum(difference), 1e-5)


class TestNeuralNetwork(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'
    iterations = 100

    def test_append_layer(self):
        # this test checks if your network actually appends layers, whether it copies the optimizer to these layers, and
        # whether it handles the initialization of the layer's weights
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1),
                                          Initializers.Constant(0.123),
                                          Initializers.Constant(0.123))
        fcl_1 = FullyConnected.Linear(1, 1)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected.Linear(1, 1)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2)
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer)
        self.assertTrue(np.all(net.layers[0].weights == 0.123))

    def test_data_access(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-4),
                                          )
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.Linear(input_size, categories, weights_initializer=Initializers.UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories, Initializers.UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2)

    def test_iris_data(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3),
                                          )
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.Linear(input_size, categories, Initializers.UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories,Initializers.UniformRandom() )
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(4000)
        if TestNeuralNetwork.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def _test_regularization_loss(self):
        '''
        This test checks if the regularization loss is calculated for the fc and rnn layer and tracked in the
        NeuralNetwork class
        '''
        import random
        fcl = FullyConnected.FullyConnected(4, 3)
        rnn = RNN.RNN(4, 4, 3)

        for layer in [fcl, rnn]:
            loss = []
            for reg in [False, True]:
                opt = Optimizers.Sgd(1e-3)
                if reg:
                    opt.add_regularizer(Constraints.L1_Regularizer(8e-2))
                net = NeuralNetwork.NeuralNetwork(opt, Initializers.Constant(0.5),
                                                  Initializers.Constant(0.1))

                net.data_layer = Helpers.IrisData(100, random=False)
                net.loss_layer = Loss.CrossEntropyLoss()
                net.append_layer(layer)
                net.append_layer(SoftMax.SoftMax())
                net.train(1)
                loss.append(np.sum(net.loss))

            self.assertNotEqual(loss[0], loss[1], "Regularization Loss is not calculated and added to the overall loss "
                                                  "for " + layer.__class__.__name__)

    def test_iris_data_with_momentum(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.SgdWithMomentum(1e-3, 0.8),)
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.Linear(input_size, categories, Initializers.UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories, Initializers.UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Momentum.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_adam(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-3, 0.9, 0.999),)
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.Linear(input_size, categories, Initializers.UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories, Initializers.UniformRandom())
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(3000)
        if TestNeuralNetwork.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_ADAM.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' +
                  str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_batchnorm(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),)
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_layer(BatchNormalization.BatchNormalization(input_size))
        fcl_1 = FullyConnected.Linear(input_size, categories, Initializers.UniformRandom())
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories,Initializers.UniformRandom() )
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using Batchnorm')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Batchnorm.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        results_next_run = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Batchnorm, we achieve an accuracy of: ' +
                  str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.8)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_iris_data_with_dropout(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.Linear(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout.Dropout(0.3))
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork.plot:
            fig = plt.figure(
                'Loss function for a Neural Net on the Iris dataset using Dropout')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Dropout.pdf"),
                        transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)

        results_next_run = net.test(data)

        with open(self.log, 'a') as f:
            print('On the Iris dataset using Dropout, we achieve an accuracy of: ' +
                  str(accuracy * 100.) + '%', file=f)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_layer_phases(self):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_layer(BatchNormalization.BatchNormalization(input_size))
        fcl_1 = FullyConnected.Linear(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.Linear(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout.Dropout(0.3))
        net.append_layer(SoftMax.SoftMax())

        net.train(100)

        data, labels = net.data_layer.get_test_set()
        results = net.test(data)

        bn_phase = net.layers[0].testing_phase
        drop_phase = net.layers[4].testing_phase

        self.assertTrue(bn_phase)
        self.assertTrue(drop_phase)

    def test_digit_data2(self):
        adam = Optimizers.Adam(5e-3, 0.98, 0.999)
        self._perform_test(
            adam, TestNeuralNetwork.iterations, 'ADAM', False, False)
        
    def test_digit_data(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(5e-3, 0.98, 0.999),
                                          )
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 200
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        cl_1 = Conv.Conv2d(stride=conv_stride_shape, kernel_size=3, in_channels=1, out_channels=num_kernels, padding="same")
        net.append_layer(cl_1)
        cl_1_output_shape = (*input_image_shape[1:], num_kernels)
        net.append_layer(ReLU.ReLU())

        pool = Pooling.MaxPool2d((2, 2), (2, 2))
        pool_output_shape = (4, 4, 4)
        net.append_layer(pool)
        fcl_1_input_size = np.prod(pool_output_shape)

        net.append_layer(Flatten.Flatten())

        fcl_1 = FullyConnected.Linear(fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        net.append_layer(ReLU.ReLU())

        fcl_2 = FullyConnected.Linear(int(fcl_1_input_size/2.), categories)
        net.append_layer(fcl_2)

        net.append_layer(SoftMax.SoftMax())

        net.train(200)

        if TestNeuralNetwork.plot:
            description = 'on_digit_data'
            fig = plt.figure('Loss function for training a Convnet on the Digit dataset')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestConvNet_" + description + ".pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        print('\nOn the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
        self.assertGreater(accuracy, 0.5)

    def _test_digit_data_L2_Regularizer(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(
            sgd_with_l2, TestNeuralNetwork3.iterations, 'L2_regularizer', False, False)

    def _test_digit_data_L1_Regularizer(self):
        sgd_with_l1 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l1.add_regularizer(Constraints.L1_Regularizer(8e-2))
        self._perform_test(
            sgd_with_l1, TestNeuralNetwork3.iterations, 'L1_regularizer', False, False)

    def test_digit_data_dropout(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        # sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))
        self._perform_test(
            sgd_with_l2, TestNeuralNetwork.iterations, 'Dropout', True, False)

    def test_digit_batch_norm(self):
        adam = Optimizers.Adam(1e-2, 0.98, 0.999)
        self._perform_test(adam, TestNeuralNetwork.iterations,
                           'Batch_norm', False, True)

    def _test_all(self):
        sgd_with_l2 = Optimizers.Adam(1e-2, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(
            sgd_with_l2, TestNeuralNetwork.iterations, 'Batch_norm and L2', False, True)

    def _perform_test(self, optimizer, iterations, description, dropout, batch_norm):
        np.random.seed(None)
        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          )
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 200
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNorm2d(1))

        cl_1 = Conv.Conv2d(stride=conv_stride_shape, in_channels=1, kernel_size=3, out_channels=num_kernels, padding="same")
        net.append_layer(cl_1)
        cl_1_output_shape = (num_kernels, *input_image_shape[1:])

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNorm2d(num_kernels))

        net.append_layer(ReLU.ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        net.append_layer(Flatten.Flatten())

        fcl_1 = FullyConnected.Linear(
            fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        if dropout:
            net.append_layer(Dropout.Dropout(0.3))

        net.append_layer(ReLU.ReLU())

        fcl_2 = FullyConnected.Linear(
            int(fcl_1_input_size / 2), int(fcl_1_input_size / 3))
        net.append_layer(fcl_2)

        net.append_layer(ReLU.ReLU())

        fcl_3 = FullyConnected.Linear(int(fcl_1_input_size / 3), categories)
        net.append_layer(fcl_3)

        net.append_layer(SoftMax.SoftMax())

        net.train(iterations)
        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(
                description, accuracy * 100.), file=f)
        print('\nOn the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.))
        self.assertGreater(accuracy, 0.3)


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = SoftMax.SoftMax()
        self.assertFalse(layer.trainable)

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax.SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)))

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10)

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12)

    def test_regression_backward_high_loss_w_CrossEntropy(self):
        input_tensor = self.label_tensor - 1
        input_tensor *= -10.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertAlmostEqual(element, 1/3, places=3)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1, places=3)

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.)

    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0)

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax.SoftMax())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.categories, self.batch_size))
        # print(input_tensor)
        layer = SoftMax.SoftMax()
        prediction = layer.forward(input_tensor.T)
        # print(prediction)
        expected_values = np.array([[0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724,
                                     0.21732724, 0.21732724],
                                    [0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387,
                                     0.23779387, 0.23779387],
                                    [0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794,
                                     0.26018794, 0.26018794],
                                    [0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095,
                                     0.28469095, 0.28469095]])
        # print(expected_values)
        # print(prediction)
        np.testing.assert_almost_equal(expected_values, prediction.T)


class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


if __name__ == "__main__":

    import sys
    if sys.argv[-1] == "Bonus":
        # sys.argv.pop()
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestOptimizers2, TestInitializers, TestFlatten,
                 TestConv2d, TestMaxPool2d, TestLinear, TestNeuralNetwork2, TestBatchNorm2d]
        percentages = [8, 5, 2, 45, 15, 2, 23]
        total_points = 0
        for t, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(t)).wasSuccessful():
                bonus_points.update({t.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({t.__name__: ["FAIL", p]})

        import time
        time.sleep(1)
        print("=========================== Statistics ===============================")
        exam_percentage = 3
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p),
                          "{:.3f} / 10 (%)".format(p / 100 * exam_percentage)])
        table.append([])
        table.append(["Ex2", "Total Achieved", "", "{} / 100 (%)".format(total_points),
                      "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])

        print(tabulate.tabulate(table, headers=[
              'Pos', 'Test', "Result", 'Percent in Exercise', 'Percent in Exam'], tablefmt="github"))
    else:
        unittest.main()
