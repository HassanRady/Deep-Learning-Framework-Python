import unittest
import numpy as np

from tests import helpers
from .set_up import LinearBaseTestCase

from DeepStorm.Layers.linear import Linear
from DeepStorm.Losses.l2 import L2Loss


class TestWithL2Loss(LinearBaseTestCase):
    def test_gradient(self):
        input_tensor = np.abs(np.random.random(
            (self.batch_size, self.input_size)))
        layers = [Linear(self.input_size, self.categories)]
        layers.append(L2Loss())
        difference = helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random(
            (self.batch_size, self.input_size)))
        layers = [Linear(self.input_size, self.categories)]
        layers.append(L2Loss())
        difference = helpers.gradient_check_weights(
            layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)