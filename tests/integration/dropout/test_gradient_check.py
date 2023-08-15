import numpy as np

from .set_up import DropoutBaseTestCase
from tests import helpers

from DeepStorm.Layers.dropout import Dropout
from DeepStorm.Losses.l2 import L2Loss


class TestWithL2Loss(DropoutBaseTestCase):
    def test_gradient(self):
        batch_size = 10
        input_size = 10
        input_tensor = np.ones((batch_size, input_size))
        label_tensor = np.zeros([batch_size, input_size])
        for i in range(batch_size):
            label_tensor[i, np.random.randint(0, input_size)] = 1
        layers = [Dropout(0.5), L2Loss()]
        difference = helpers.gradient_check(
            layers, input_tensor, label_tensor, seed=1337)
        self.assertLessEqual(np.sum(difference), 1e-5)