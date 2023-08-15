import numpy as np

from .set_up import Conv2dBaseTestCase

from DeepStorm.Layers.conv import Conv2d
from DeepStorm.Optimizers.sgd import Sgd

class TestSgd(Conv2dBaseTestCase):
    def test_update(self):
        input_tensor = np.random.uniform(-1, 1,
                                         (self.batch_size, *self.input_shape))
        conv = Conv2d(in_channels=self.in_channels, out_channels=self.num_kernels,
                      kernel_size=self.kernel_shape, stride=(3, 2), padding='same', weights_initializer=self.TestInitializer())
        conv.optimizer = Sgd(1)
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)),
                            np.sum(np.power(new_output_tensor, 2)))