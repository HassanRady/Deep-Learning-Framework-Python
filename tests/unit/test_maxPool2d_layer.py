import unittest
import numpy as np
import matplotlib.pyplot as plt

from tests import helpers

from DeepStorm.Layers.pooling import MaxPool2d


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

    def test_trainable(self):
        layer = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.assertFalse(layer.trainable)

    def test_shape(self):
        layer = MaxPool2d(stride=(2, 2), kernel_size=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(
            np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_overlapping_shape(self):
        layer = MaxPool2d(stride=(2, 1), kernel_size=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(
            np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_subsampling_shape(self):
        layer = MaxPool2d(stride=(3, 2), kernel_size=(2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(
            np.sum(np.abs(np.array(result.shape) - expected_shape)), 0)

    def test_layout_preservation(self):
        pool = MaxPool2d(stride=(1, 1), kernel_size=(1, 1))
        input_tensor = np.array(
            range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.)

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = MaxPool2d(stride=(2, 2), kernel_size=(2, 2))
        batch_size = 2
        input_tensor = np.array(
            range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[4]]], [[[13]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0)

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = MaxPool2d(stride=2, kernel_size=2)
        batch_size = 2
        input_tensor = np.array(
            range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array(
            [[[[5.,  7.], [13., 15.]]], [[[21., 23.], [29., 31.]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0)