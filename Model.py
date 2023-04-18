import copy
from Layers.Base import BaseLayer
from logger import get_file_logger

import numpy as np

_logger = get_file_logger(__name__)

class Model(object):
    def __init__(self, model=None) -> None:
        self.model = []
        self.train_output = {}
        self.eval_output = {}

        if isinstance(model, list):
            self.model = model

    def train_step(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        self.backward(y)
        return loss, output
    
    def val_step(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        return loss, output
    
    def train_epoch(self):
        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_train, self.y_train):
            batch_loss, predictions = self.train_step(x_batch, y_batch)
            
            running_loss += batch_loss
            running_preds.append(predictions)

        epoch_loss = running_loss/self.data_len
        return epoch_loss, running_preds

    def eval_epoch(self):
        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_val, self.y_val):
            batch_loss, predictions = self.val_step(x_batch, y_batch)

            running_loss += batch_loss
            running_preds.append(predictions)

        epoch_loss = batch_loss/self.data_len
        return epoch_loss, running_preds

    def batcher(self, x, y):
        x = np.array_split(x, len(x)//self.batch_size)
        y = np.array_split(y, len(y)//self.batch_size)
        self.data_len = len(x)
        for x_batch, y_batch in zip(x, y):
            yield x_batch, y_batch

    def fit(self, x_train, y_train, x_val, y_val, epochs):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        train_losses = []
        train_preds = []
        val_losses = []
        val_preds = []

        for i in range(1, epochs + 1):
            print(f"EPOCH {i}: ")

            train_loss, train_pred = self.train_epoch()
            val_loss, val_pred = self.eval_epoch()

            train_losses.append(train_loss)
            train_preds.append(train_pred)
            val_losses.append(val_loss)
            val_preds.append(val_pred)

            print(f"Train Loss: {train_loss:.2f}")
            print(f"Val Loss: {val_loss:.2f} \n")

        self.train_output['loss'] = train_losses
        self.train_output['predictions'] = train_preds
        self.eval_output['loss'] = val_losses
        self.eval_output['predictions'] = val_preds
        return self.train_output, self.eval_output
    
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

        self.eval_output['loss'] = val_losses
        self.eval_output['preds'] = val_preds
        return self.eval_output

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
