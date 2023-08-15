import unittest
import numpy as np

from .set_up import LinearBaseTestCase

from DeepStorm.Layers.linear import Linear
from DeepStorm.Optimizers.sgd import Sgd


class TestLinearSgd(LinearBaseTestCase):
    def test_update(self):
        layer = Linear(
            self.input_size, self.output_size)
        layer.optimizer = Sgd(1)
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
        layer = Linear(
            self.input_size, self.output_size)
        layer.optimizer = Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)),
                            np.sum(np.power(new_output_tensor, 2)))