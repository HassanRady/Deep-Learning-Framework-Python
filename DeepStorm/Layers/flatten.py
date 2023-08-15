import numpy as np

from DeepStorm.Layers.base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
        self.initializable = False

    def forward(self, input_tensor: np.array):
        self.input_tensor = input_tensor
        return input_tensor.reshape(input_tensor.shape[0], -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_tensor.shape)