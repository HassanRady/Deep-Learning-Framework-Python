import unittest
import numpy as np


class maxPool2dBaseTestCase(unittest.TestCase):
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

        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

