"""Gradient check for the Conv2d layer with L2Loss and Flatten layer."""

import numpy as np

from .set_up import Conv2dBaseTestCase
from tests import helpers

from DeepStorm.Layers.conv import Conv2d
from DeepStorm.Layers.flatten import Flatten
from DeepStorm.Losses.l2 import L2Loss


class TestWithL2Loss(Conv2dBaseTestCase):
    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = [
            Conv2d(
                in_channels=3,
                out_channels=self.hidden_channels,
                kernel_size=3,
                stride=1,
                padding='same',
            )
        ]
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = helpers.gradient_check(
            layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 5e-2)

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = [
            Conv2d(
                in_channels=3,
                out_channels=self.hidden_channels,
                kernel_size=3,
                stride=1,
                padding='same',
            )
        ]
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = helpers.gradient_check_weights(
            layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 36])
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = [
            Conv2d(
                in_channels=3,
                out_channels=self.hidden_channels,
                kernel_size=3,
                stride=2,
                padding='same',
            )
        ]
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = helpers.gradient_check_weights(
            layers, input_tensor, label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = [
            Conv2d(
                in_channels=3,
                out_channels=self.hidden_channels,
                kernel_size=3,
                stride=1,
                padding='same',
            )
        ]
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = helpers.gradient_check_weights(
            layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_stride(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 35])
        input_tensor = np.abs(np.random.random((2, 6, 5, 14)))
        layers = [
            Conv2d(
                in_channels=6,
                out_channels=1,
                kernel_size=3,
                stride=(1, 2),
                padding='same',
            )
        ]
        layers.append(Flatten())
        layers.append(L2Loss())
        difference = helpers.gradient_check(layers, input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)