import copy
from Layers.Base import BaseLayer
from logger import get_file_logger

import numpy as np

_logger = get_file_logger(__name__)

class Model(object):
    def __init__(self, model=None) -> None:
        self.model = []

        if isinstance(model, list):
            self.model = model

    def train_step(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        self.backward(y)
        _logger.debug(f"loss : {loss}")
        return loss
    
    def val_step(self, x, y):
        output = self.forward(x)
        loss = self.loss_layer.forward(output, y)
        return loss
    
    def train_epoch(self):
        loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_train, self.y_train):
            loss += self.train_step(x_batch, y_batch)
        epoch_loss = loss/len(self.x_train)
        return epoch_loss

    def eval_epoch(self):
        loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_val, self.y_val):
            loss += self.val_step(x_batch, y_batch)
        epoch_loss = loss/len(self.x_val)
        return epoch_loss

    def batcher(self, x, y):
        x = np.array_split(x, len(x)//self.batch_size)
        y = np.array_split(y, len(y)//self.batch_size)
        for x_batch, y_batch in zip(x, y):
            yield x_batch, y_batch
    
    def train(self, x, y):
        self.x_train = x
        self.y_train = y
        return self.train_epoch()

    def eval(self, x, y,):
        self.x_val = x
        self.y_val = y
        return self.eval_epoch()

    def append_layer(self, layer: BaseLayer):
        if isinstance(layer, BaseLayer):
            self.model.append(layer)

    def compile(self, optimizer, loss, batch_size):
        self.batch_size = batch_size
        self.loss = loss
        self.set_optimizer(optimizer)

    def set_optimizer(self, optimizer):
        for layer in self.model:
            if layer.trainable:
                layer.optimizer = copy.deepcopy(optimizer)

    def forward(self, x):
        for layer in self.model:
            output = layer.forward(x)
            x = output
        return output

    def backward(self, y):
        y = self.loss.backward(y)
        for layer in reversed(self.model):
            output = layer.backward(y)
            y = output
