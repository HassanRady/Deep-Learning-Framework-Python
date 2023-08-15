# TODO: fix _test_forward_test_phase_convolutional.

import unittest
import numpy as np

from DeepStorm.Layers.batch_normalization import BatchNorm2d


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
        layer = BatchNorm2d(
            self.input_tensor.shape[-1])
        self.assertTrue(layer.trainable)

    def test_default_phase(self):
        layer = BatchNorm2d(
            self.input_tensor.shape[-1])
        self.assertTrue(layer.training)

    def _test_forward_linear_shape(self):
        layer = BatchNorm2d(
            self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)

        self.assertEqual(output.shape[0], self.input_tensor.shape[0])
        self.assertEqual(output.shape[1], self.input_tensor.shape[1])

    def test_forward_shape_convolutional(self):
        layer = BatchNorm2d(self.channels)
        output = layer.forward(self.input_tensor_conv)

        self.assertEqual(output.shape, self.input_tensor_conv.shape)

    def _test_forward_linear(self):
        layer = BatchNorm2d(
            self.input_tensor.shape[-1])
        output = layer.forward(self.input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        self.assertAlmostEqual(
            np.sum(np.square(mean - np.zeros(mean.shape[0]))), 0)
        self.assertAlmostEqual(
            np.sum(np.square(var - np.ones(var.shape[0]))), 0)

    def _test_reformat_image2vec(self):
        layer = BatchNorm2d(3)
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reshape_tensor_for_input(image_tensor)
        np.testing.assert_equal(vec_tensor.shape, (120, 3))
        self.assertEqual(np.sum(vec_tensor, 1)[0], 72)
        self.assertEqual(np.sum(vec_tensor, 0)[0], 18660)

    def _test_reformat_vec2image(self):
        layer = BatchNorm2d(3)
        layer.forward(np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4))
        vec_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(120, 3)
        image_tensor = layer.reshape_tensor_for_output(vec_tensor)
        np.testing.assert_equal(image_tensor.shape, (5, 3, 6, 4))
        self.assertEqual(np.sum(image_tensor, (0, 1, 2))[0], 15750)
        self.assertEqual(np.sum(image_tensor, (0, 2, 3))[0], 21420)

    def _test_reformat(self):
        layer = BatchNorm2d(3)
        layer.forward(np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4))
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reshape_tensor_for_input(image_tensor)
        image_tensor2 = layer.reshape_tensor_for_output(vec_tensor)
        np.testing.assert_allclose(image_tensor, image_tensor2)

    def test_forward_convolutional(self):
        layer = BatchNorm2d(self.channels)
        output = layer.forward(self.input_tensor_conv)
        mean, var = TestBatchNorm2d._channel_moments(output, self.channels)

        self.assertAlmostEqual(np.sum(np.square(mean)), 0)
        self.assertAlmostEqual(np.sum(np.square(var - np.ones_like(var))), 0)

    def _test_forward_linear_train_phase(self):
        layer = BatchNorm2d(
            self.input_tensor.shape[-1])
        layer.forward(self.input_tensor)

        output = layer.forward((np.zeros_like(self.input_tensor)))

        mean = np.mean(output, axis=0)

        mean_input = np.mean(self.input_tensor, axis=0)
        var_input = np.var(self.input_tensor, axis=0)

        self.assertNotEqual(
            np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def test_forward_train_phase_convolutional(self):
        layer = BatchNorm2d(self.channels)
        layer.forward(self.input_tensor_conv)

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm2d._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm2d._channel_moments(
            self.input_tensor_conv, self.channels)

        self.assertNotEqual(
            np.sum(np.square(mean + (mean_input/np.sqrt(var_input)))), 0)

    def _test_forward_linear_test_phase(self):
        layer = BatchNorm2d(
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

    def _test_forward_test_phase_convolutional(self):
        layer = BatchNorm2d(self.channels, eps=1e-11)
        layer.forward(self.input_tensor_conv)
        layer.training = False

        output = layer.forward((np.zeros_like(self.input_tensor_conv)))

        mean, var = TestBatchNorm2d._channel_moments(output, self.channels)
        mean_input, var_input = TestBatchNorm2d._channel_moments(
            self.input_tensor_conv, self.channels)

        self.assertAlmostEqual(
            np.sum(np.square(mean + (mean_input / np.sqrt(var_input)))), 0)
        self.assertAlmostEqual(np.sum(np.square(var)), 0)


    # def _test_linear_update(self):
    #     layer = BatchNorm2d(
    #         self.input_tensor.shape[-1])
    #     layer.optimizer = Sgd(1)
    #     for _ in range(10):
    #         output_tensor = layer.forward(self.input_tensor)
    #         error_tensor = np.zeros_like(self.input_tensor)
    #         error_tensor -= output_tensor
    #         layer.backward(error_tensor)
    #         new_output_tensor = layer.forward(self.input_tensor)
    #         self.assertLess(np.sum(np.power(output_tensor, 2)),
    #                         np.sum(np.power(new_output_tensor, 2)))