import unittest
import numpy as np

from DeepStorm.Layers.linear import Linear


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

    def test_trainable(self):
        layer = Linear(
            self.input_size, self.output_size)
        self.assertTrue(layer.trainable)

    def test_forward_size(self):
        layer = Linear(
            self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = Linear(
            self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = Linear(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

