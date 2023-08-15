import unittest
import numpy as np


class BatchNorm2dBaseTestCase(unittest.TestCase):
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