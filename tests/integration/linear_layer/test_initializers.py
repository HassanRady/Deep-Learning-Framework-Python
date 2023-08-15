import numpy as np

from .set_up import LinearBaseTestCase

from DeepStorm.Layers.linear import Linear
from DeepStorm.Initializers.constant import Constant


class TestConstant(LinearBaseTestCase):
    class DummyWeightInitializer:
        def __init__(self):
            super().__init__()
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

    def test_initialization(self):
        init = self.DummyWeightInitializer()
        layer = Linear(self.input_size, self.categories,
                       init, Constant(0.5))
        self.assertEqual(init.fan_out, self.categories)
        self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5)
