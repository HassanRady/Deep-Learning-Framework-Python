import numpy as np

from .set_up import maxPool2dBaseTestCase
from tests import helpers

from DeepStorm.Layers.pooling import MaxPool2d
from DeepStorm.Layers.flatten import Flatten
from DeepStorm.Losses.l2 import L2Loss


class TestWithL2Loss(maxPool2dBaseTestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.layers = [None, Flatten()]
        self.layers.append(L2Loss())

    def test_gradient_stride(self):
        self.layers[0] = MaxPool2d(stride=(2, 2), kernel_size=(2, 2))
        difference = helpers.gradient_check(
            self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_overlapping_stride(self):
        # sourcery skip: class-extract-method
        label_tensor = np.random.random((self.batch_size, 24))
        self.layers[0] = MaxPool2d(stride=(2, 1), kernel_size=(2, 2))
        difference = helpers.gradient_check(
            self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)

    def test_gradient_subsampling_stride(self):
        label_tensor = np.random.random((self.batch_size, 6))
        self.layers[0] = MaxPool2d(stride=(3, 2), kernel_size=(2, 2))
        difference = helpers.gradient_check(
            self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6)