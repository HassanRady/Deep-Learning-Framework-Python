import numpy as np

from .set_up import Conv2dBaseTestCase
from tests import helpers

from DeepStorm.Layers.conv import Conv2d
from DeepStorm.Initializers.constant import Constant


class TestConstant(Conv2dBaseTestCase):
    def test_weights_init(self):
        # simply checks whether you have not initialized everything with zeros
        conv = Conv2d(in_channels=100, out_channels=150,
                      kernel_size=10, stride=1, padding='same', weights_initializer=Constant(0.1), bias_initializer=Constant(0.1))
        self.assertGreater(np.mean(np.abs(conv.weights)), 1e-3)

    def test_bias_init(self):
        conv = Conv2d(in_channels=1, out_channels=150 *
                      100 * 10 * 10, kernel_size=1, stride=1, padding='same', weights_initializer=Constant(0.0), bias_initializer=Constant(0.1))
        self.assertGreater(np.mean(np.abs(conv.bias)), 1e-3)

    def test_initialization(self):
        init = Constant(0.1)
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                      kernel_size=self.kernel_shape, stride=(1, 1), padding='same', weights_initializer=init, bias_initializer=Constant(0.1))
        self.assertEqual(
            init.fan_in, self.kernel_shape[0] * self.kernel_shape[1] * self.in_channels)

        self.assertEqual(init.fan_out, np.prod(
            self.kernel_shape[:]) * self.num_kernels)
