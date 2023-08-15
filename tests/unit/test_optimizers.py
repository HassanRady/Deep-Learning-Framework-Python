import unittest
import numpy as np

from DeepStorm.Optimizers.adam import Adam
from DeepStorm.Optimizers.sgd import Sgd, SgdWithMomentum


class TestSgd(unittest.TestCase):
    def test_sgd(self):
        optimizer = Sgd(1.)
        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))
        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]))


class TestSgdWithMomentum(unittest.TestCase):
    def test_sgd_with_momentum(self):
        optimizer = SgdWithMomentum(1., 0.9)
        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))
        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.9]))


class TestAdam(unittest.TestCase):
    def test_adam(self):
        optimizer = Adam(1., 0.01, 0.02, 1e-8)
        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]))
        result = optimizer.calculate_update(result, .5)
        np.testing.assert_almost_equal(result, np.array([-0.9814473195614205]))
