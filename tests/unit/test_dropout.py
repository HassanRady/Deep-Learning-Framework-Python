import unittest
import numpy as np

from DeepStorm.Layers.dropout import Dropout


class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10
        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_trainable(self):
        layer = Dropout(0.25)
        self.assertFalse(layer.trainable)

    def test_default_phase(self):
        drop_layer = Dropout(0.25)
        self.assertTrue(drop_layer.training)

    def test_forward_trainTime(self):
        drop_layer = Dropout(0.25)
        output = drop_layer.forward(self.input_tensor)
        self.assertEqual(np.max(output), 4)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean/self.input_size, 1., places=1)

    def test_position_preservation(self):
        drop_layer = Dropout(0.5)
        output = drop_layer.forward(self.input_tensor)
        error_prev = drop_layer.backward(self.input_tensor)
        np.testing.assert_almost_equal(
            np.where(output == 0.), np.where(error_prev == 0.))

    def test_forward_testTime(self):
        drop_layer = Dropout(0.5)
        drop_layer.training = False
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size)

    def test_backward(self):
        drop_layer = Dropout(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)
        self.assertEqual(np.max(output), 2)
        self.assertEqual(np.min(output), 0)