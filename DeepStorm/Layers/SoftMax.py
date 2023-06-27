import numpy as np
from DeepStorm.Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    def expand_softmax_values(self, x):
        size = x.shape[0]
        x = np.tile(x, size)
        x = x.reshape(size, size).T
        return x

    def forward(self, x):
        max_x = np.amax(x, 1).reshape(x.shape[0], 1)
        e_x = np.exp(x - max_x)
        self.softmax_value = e_x / e_x.sum(axis=1, keepdims=True)
        return self.softmax_value

    def backward(self, error_gradient):  # Y is NXM

        tmp = np.apply_along_axis(
            self.expand_softmax_values, axis=1, arr=self.softmax_value)  # NXMXM

        softmax_gradient = tmp * \
            (np.eye(tmp.shape[1]) - tmp.transpose(0, 2, 1))  # NXMXM

        # NXMXM * NXMX1
        v = softmax_gradient @ np.expand_dims(error_gradient, axis=2)
        v = np.squeeze(v)  # NXM
        return v
