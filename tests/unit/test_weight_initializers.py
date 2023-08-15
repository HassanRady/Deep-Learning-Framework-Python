import unittest
import numpy as np

from scipy import stats

from DeepStorm.Initializers.constant import Constant
from DeepStorm.Initializers.uniform_random import UniformRandom
from DeepStorm.Initializers.he import He
from DeepStorm.Initializers.xavier import Xavier

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
        layer = self.DummyLayer(self.input_size, self.output_size)
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return layer.shape, weights_after_init

    def test_uniform_shape(self):
        weights_shape, weights_after_init = self._performInitialization(
            UniformRandom())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_uniform_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(
            UniformRandom())

        p_value = stats.kstest(weights_after_init.flat,
                               'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_xavier_shape(self):
        weights_shape, weights_after_init = self._performInitialization(
            Xavier())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_xavier_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(
            Xavier())

        scale = np.sqrt(2) / np.sqrt(self.input_size + self.output_size)
        p_value = stats.kstest(weights_after_init.flat,
                               'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_he_shape(self):
        weights_shape, weights_after_init = self._performInitialization(
            He())

        self.assertEqual(weights_shape, weights_after_init.shape)

    def test_he_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(
            He())

        scale = np.sqrt(2) / np.sqrt(self.input_size)
        p_value = stats.kstest(weights_after_init.flat,
                               'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)