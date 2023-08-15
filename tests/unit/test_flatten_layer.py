import unittest
import numpy as np

from DeepStorm.Layers.flatten import Flatten


class TestFlatten(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_shape = (3, 4, 11)
        self.input_tensor = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        self.input_tensor = self.input_tensor.reshape(
            self.batch_size, *self.input_shape)

    def test_trainable(self):
        layer = Flatten()
        self.assertFalse(layer.trainable)

    def test_flatten_forward(self):
        flatten = Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        input_vector = np.array(
            range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_vector = input_vector.reshape(
            self.batch_size, np.prod(self.input_shape))
        self.assertLessEqual(np.sum(np.abs(output_tensor-input_vector)), 1e-9)

    def test_flatten_backward(self):
        flatten = Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        backward_tensor = flatten.backward(output_tensor)
        self.assertLessEqual(
            np.sum(np.abs(self.input_tensor - backward_tensor)), 1e-9)
