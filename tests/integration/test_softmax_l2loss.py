import numpy as np

from .set_up import SoftmaxBaseTestCase

from DeepStorm.Activations.softmax import SoftMax
from DeepStorm.Losses.l2 import L2Loss


class TestSoftMaxL2LossForward(SoftmaxBaseTestCase):
    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10)

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12)

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        self.assertGreater(float(loss), 0.)

