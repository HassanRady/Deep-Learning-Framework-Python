from Layers.Initializers import Xavier, He, UniformRandom, Constant
from Layers.Conv import Conv2d
from Layers.BatchNormalization import BatchNorm2d
from Layers.Pooling import MaxPool2d
from Layers.Flatten import Flatten
from Layers.FullyConnected import Linear
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Adam, SgdWithMomentum, Sgd

from train import Trainer
from Layers.Helpers import DigitData

model = [
    Conv2d(in_channels=1, out_channels=4,
           kernel_size=3, stride=1, padding='same'),
    ReLU(),
    BatchNorm2d(8*8*4),
    MaxPool2d(kernel_size=2, stride=2),

    Conv2d(in_channels=4, out_channels=4,
           kernel_size=3, stride=1, padding='same'),
    ReLU(),
    BatchNorm2d(4*4*4),
    # MaxPool2d(kernel_size=2, stride=2),

    Flatten(),
    Linear(in_features=4*4*4, out_features=32),
    ReLU(),
    Linear(32, 10),
    SoftMax(),
]

x, y = DigitData(200).next()

trainer = Trainer(model, Adam(5e-3, 0.98, 0.999),
                  He(),
                  Constant(0.1))

trainer.loss_layer = CrossEntropyLoss()

trainer.fit(200, [[x, y]])
