import numpy as np

from .set_up import ReluBaseTestCase
from tests import helpers

from DeepStorm.Activations.relu import ReLU
from DeepStorm.Losses.l2 import L2Loss

class TestL2Loss(ReluBaseTestCase):
    def test_gradient(self):
        input_tensor = np.abs(np.random.random(
            (self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = [ReLU()]
        layers.append(L2Loss())
        difference = helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)
