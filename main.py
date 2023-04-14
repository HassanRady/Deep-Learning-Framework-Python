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

from Train import Trainer
from Layers.Helpers import DigitData

from sklearn.model_selection import train_test_split
import pandas as pd

input_folder_path = "Data/"
train_df = pd.read_csv(input_folder_path+"train.csv")
test_df = pd.read_csv(input_folder_path+"test.csv")

train_labels = train_df['label'].values
train_images = (train_df.iloc[:,1:].values).astype('float32')
test_images = (test_df.iloc[:,:].values).astype('float32')

#Training and Validation Split
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                     stratify=train_labels, random_state=123,
                                                                     test_size=0.20)
train_images = train_images.reshape(train_images.shape[0], 28, 28)
val_images = val_images.reshape(val_images.shape[0], 28, 28)
test_images = test_images.reshape(test_images.shape[0], 28, 28)

train_images = train_images/255.0
val_images = val_images/255.0
test_images = test_images/255.0

    


model = [
    Conv2d(in_channels=1, out_channels=32,
           kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(32),
    ReLU(),

    Conv2d(in_channels=32, out_channels=64,
           kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(64),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),

    Conv2d(in_channels=64, out_channels=128,
           kernel_size=3, stride=1, padding='same'),
    BatchNorm2d(128),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),


    Flatten(),
    Linear(in_features=128*7*7, out_features=128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
    SoftMax(),
]

# x, y = DigitData(200).next()

# trainer = Trainer(model, Adam(5e-3, 0.98, 0.999),
#                   He(),
#                   Constant(0.1))

# trainer.loss_layer = CrossEntropyLoss()

# trainer.fit(200, [[x, y]])
