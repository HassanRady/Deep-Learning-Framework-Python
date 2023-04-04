from Layers.Initializers import Xavier, He, UniformRandom, Constant
from Layers.Conv import Conv
from Layers.Pooling import Pooling
from Layers.Flatten import Flatten
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Adam, SgdWithMomentum, Sgd


model = [
    Conv(in_channels=, out_channels=, kernel=, stride=, padding=),
    ReLU(),
    Pooling(kernel=None, stride=None),
    Flatten(),
    FullConnected(input_size=, output_size=),
    FullConnected(input_size=, output_size=),
    SoftMax()
]

