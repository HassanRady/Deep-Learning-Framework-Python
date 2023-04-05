from Layers.Initializers import Xavier, He, UniformRandom, Constant
from Layers.Conv import Conv2d
from Layers.Pooling import MaxPool2d
from Layers.Flatten import Flatten
from Layers.FullyConnected import Linear
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Adam, SgdWithMomentum, Sgd


model = [
    Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding='same'),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(in_features=, out_features=),
    SoftMax(),
]

