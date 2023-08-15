import unittest
import numpy as np


class DropoutBaseTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10
        self.input_tensor = np.ones((self.batch_size, self.input_size))