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
        return loss
    
    def val_step(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        return loss
    
    def train_epoch(self):
        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_train, self.y_train):
            batch_loss, predictions = self.train_step(x_batch, y_batch)
            
            running_loss += batch_loss
            running_preds.append(predictions)

        epoch_loss = running_loss/self.data_len
        return epoch_loss

    def eval_epoch(self):
        loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_val, self.y_val):
            loss += self.val_step(x_batch, y_batch)
        epoch_loss = loss/self.data_len
        return epoch_loss

    def batcher(self, x, y):
        x = np.array_split(x, len(x)//self.batch_size)
        y = np.array_split(y, len(y)//self.batch_size)
        self.data_len = len(x)
        for x_batch, y_batch in zip(x, y):
            yield x_batch, y_batch
    
    def train(self, x, y):
        self.x_train = x
        self.y_train = y
        return self.train_epoch()

    def eval(self, x, y, epochs):
        self.x_val = x
        self.y_val = y

        val_losses = []
        val_preds = []

        for i in range(1, epochs+1):
            print(f"Epoch {i}: \n")

            loss, preds = self.eval_epoch()

            val_losses.append(loss)
            val_preds.append(preds)

        # metric = calc_metric(val_preds)
        return {"loss": val_losses, "preds": val_preds}

    def append_layer(self, layer: BaseLayer):
        if isinstance(layer, BaseLayer):
            self.model.append(layer)

    def compile(self, optimizer, loss, batch_size, metrics):
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
