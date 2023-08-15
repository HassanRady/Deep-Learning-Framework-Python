# TODO: fix test_gradient_conv.

import numpy as np

from .set_up import BatchNorm2dBaseTestCase
from tests import helpers

from DeepStorm.Layers.batch_normalization import BatchNorm2d
from DeepStorm.Layers.linear import Linear
from DeepStorm.Layers.flatten import Flatten
from DeepStorm.Losses.l2 import L2Loss


class TestWithLinearAndL2Loss(BatchNorm2dBaseTestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        super().setUp()
        self.layers = [None, Flatten()]
        self.layers.append(Linear(
            self.input_size, self.categories))
        self.layers.append(L2Loss())

    # tests for batchNorm1d
    # def _test_linear_gradient(self):
    #     self.layers[0] = BatchNorm2d(
    #         self.input_tensor.shape[-1])
    #     difference = helpers.gradient_check(
    #         self.layers, self.input_tensor, self.label_tensor)
    #     self.assertLessEqual(np.sum(difference), 1e-4)

    # def _test_linear_gradient_weights(self):
    #     self.layers[0] = BatchNorm2d(
    #         self.input_tensor.shape[-1])
    #     self.layers[0].forward(self.input_tensor)
    #     difference = helpers.gradient_check_weights(
    #         self.layers, self.input_tensor, self.label_tensor, False)
    #     self.assertLessEqual(np.sum(difference), 1e-6)

    # def _test_linear_gradient_bias(self):
    #     self.layers[0] = BatchNorm2d(
    #         self.input_tensor.shape[-1])
    #     self.layers[0].forward(self.input_tensor)
    #     difference = helpers.gradient_check_weights(
    #         self.layers, self.input_tensor, self.label_tensor, True)
    #     self.assertLessEqual(np.sum(difference), 1e-6)

    def _test_gradient_convolutional(self):
        self.layers[0] = BatchNorm2d(self.channels)
        difference = helpers.gradient_check(
            self.layers, self.input_tensor_conv, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-3)

    def test_gradient_weights_convolutional(self):
        self.layers[0] = BatchNorm2d(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = helpers.gradient_check_weights(
            self.layers, self.input_tensor_conv, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_bias_convolutional(self):
        self.layers[0] = BatchNorm2d(self.channels)
        self.layers[0].forward(self.input_tensor_conv)
        difference = helpers.gradient_check_weights(
            self.layers, self.input_tensor_conv, self.label_tensor, True)
        self.assertLessEqual(np.sum(difference), 1e-6)