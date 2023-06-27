import copy
from DeepStorm.Layers.Base import BaseLayer
from DeepStorm.logger import get_file_logger

import numpy as np

_logger = get_file_logger(__name__, 'logs')


class Model(object):
    def __init__(self, model=None) -> None:
        self.model = []
        self.fit_output = {}

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
        _logger.info(f"Training")

        for layer in self.model:
            layer.train()

        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_train, self.y_train):
            batch_loss, preds = self.train_step(x_batch, y_batch)

            running_loss += batch_loss
            running_preds.append(preds)

        epoch_loss = running_loss/self.data_len

        running_preds = np.array(running_preds)
        running_preds = running_preds.reshape(
            running_preds.shape[0]*running_preds.shape[1], running_preds.shape[2])

        self.calc_metrics(running_preds, self.y_train)

        print(f"Train loss: {epoch_loss:.2f}")
        for metric, metric_output in self.metrics_output.items():
            self.fit_output[f"{metric}"].append(metric_output)
            print(f"Train {metric}: {metric_output}")

        return epoch_loss, running_preds

    def eval_epoch(self):
        _logger.info(f"Validation")

        for layer in self.model:
            layer.eval()

        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_val, self.y_val):
            batch_loss, preds = self.val_step(x_batch, y_batch)

            running_loss += batch_loss
            running_preds.append(preds)

        epoch_loss = running_loss/self.data_len

        running_preds = np.array(running_preds)
        running_preds = running_preds.reshape(
            running_preds.shape[0]*running_preds.shape[1], running_preds.shape[2])

        self.calc_metrics(running_preds, self.y_val)

        print(f"Val loss: {epoch_loss:.2f}")
        for metric, metric_output in self.metrics_output.items():
            self.fit_output[f"val_{metric}"].append(metric_output)
            print(f"Val {metric}: {metric_output}")

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

        for metric in self.metrics:
            self.fit_output[f"{metric}"] = []
            self.fit_output[f"val_{metric}"] = []

        for i in range(1, epochs + 1):
            print(f"Epoch {i}: ")
            _logger.info(f"Epoch: {i}")

            train_loss, train_pred = self.train_epoch()
            val_loss, val_pred = self.eval_epoch()

            train_losses.append(train_loss)
            train_preds.append(train_pred)
            val_losses.append(val_loss)
            val_preds.append(val_pred)

            print()

        self.fit_output['loss'] = train_losses
        self.fit_output['predictions'] = train_preds
        self.fit_output['val_loss'] = val_losses
        self.fit_output['val_predictions'] = val_preds

        return self.fit_output


    def append_layer(self, layer: BaseLayer):
        if isinstance(layer, BaseLayer):
            self.model.append(layer)

    def compile(self, optimizer, loss, batch_size, metrics: list):
        self.batch_size = batch_size
        self.loss = loss
        self.set_optimizer(optimizer)
        self.metrics = metrics

    def calc_metrics(self, preds, labels):
        self.metrics_output = {}
        for metric in self.metrics:
            if metric == "accuracy":
                self.metrics_output['accuracy'] = self.calc_accuracy(
                    preds, labels)

    def calc_accuracy(self, preds, labels):
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(labels, axis=1)
        accuracy = np.mean(preds == labels)
        return accuracy

    def set_optimizer(self, optimizer):
        for layer in self.model:
            if layer.trainable:
                layer.optimizer = copy.deepcopy(optimizer)

    def forward(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return x

    def backward(self, y):
        y = self.loss.backward(y)
        for layer in reversed(self.model):
            y = layer.backward(y)
    
    def train(self, x, y, epochs):
        self.x_train = x
        self.y_train = y

        train_losses = []
        train_preds = []

        for i in range(1, epochs+1):
            print(f"Epoch {i}: \n")

            loss, preds = self.train_epoch()

            train_losses.append(loss)
            train_preds.append(preds)

        self.fit_output['loss'] = train_losses
        self.fit_output['preds'] = train_preds
        return self.fit_output

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

        self.fit_output['val_loss'] = val_losses
        self.fit_output['val_preds'] = val_preds
        return self.fit_output
