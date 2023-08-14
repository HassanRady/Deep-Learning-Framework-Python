import unittest
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import tabulate


from DeepStorm.Activations.relu import ReLU
from DeepStorm.Activations.sigmoid import Sigmoid
from DeepStorm.Activations.softmax import SoftMax


from tests import Helpers

ID = 2  # identifier for dispatcher
BATCH_SIZE = 32
METRICS = ['accuracy']