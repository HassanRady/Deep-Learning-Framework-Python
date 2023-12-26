import copy
from DeepStorm.Layers.base import BaseLayer
from DeepStorm.Utils.metrics_calculator import calc_accuracy
from DeepStorm.logger import get_file_logger

import numpy as np

_logger = get_file_logger(__name__, 'logs')


class Model(object):
    def __init__(self, layers:list=None) -> None:
        self.layers = []
        self.fit_output = {}
        if isinstance(layers, list):
            self.layers = layers

    def append_layer(self, layer: BaseLayer):
        if isinstance(layer, BaseLayer):
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        y = self.loss.backward(y)
        for layer in reversed(self.layers):
            y = layer.backward(y)

    def train_batch(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        self.backward(y)
        return loss, output

    def val_batch(self, x, y):
        output = self.forward(x)
        loss = self.loss.forward(output, y)
        return loss, output

    def train_epoch(self):
        _logger.info("Training")

        for layer in self.layers:
            layer.train()

        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_train, self.y_train):
            batch_loss, preds = self.train_batch(x_batch, y_batch)

            running_loss += batch_loss
            running_preds.append(preds)

        epoch_loss = running_loss/self.num_batch

        running_preds = np.array(running_preds)
        running_preds = running_preds.reshape(
            running_preds.shape[0]*running_preds.shape[1], running_preds.shape[2])

        metrics_output = self.calc_metrics(running_preds, self.y_train)

        print(f"Train loss: {epoch_loss:.2f}")
        for metric, metric_output in metrics_output.items():
            self.fit_output[f"{metric}"].append(metric_output)
            print(f"Train {metric}: {metric_output:.2f}")

        return epoch_loss, running_preds

    def eval_epoch(self):
        _logger.info("Validation")

        for layer in self.layers:
            layer.eval()

        running_preds = []
        running_loss = 0.0
        for x_batch, y_batch in self.batcher(self.x_val, self.y_val):
            batch_loss, preds = self.val_batch(x_batch, y_batch)

            running_loss += batch_loss
            running_preds.append(preds)

        epoch_loss = running_loss/self.num_batch

        running_preds = np.array(running_preds)
        running_preds = running_preds.reshape(
            running_preds.shape[0]*running_preds.shape[1], running_preds.shape[2])

        metrics_output = self.calc_metrics(running_preds, self.y_val)

        print(f"Val loss: {epoch_loss:.2f}")
        for metric, metric_output in metrics_output.items():
            self.fit_output[f"val_{metric}"].append(metric_output)
            print(f"Val {metric}: {metric_output:.2f}")

        return epoch_loss, running_preds
    
    def batch_size_adjuster(self, x, y):
        limit = len(x) - len(x)%self.batch_size
        return x[:limit], y[:limit]

    def fit(self, x_train, y_train, x_val, y_val, epochs):
        self.x_train, self.y_train = self.batch_size_adjuster(x_train, y_train)
        self.x_val, self.y_val = self.batch_size_adjuster(x_val, y_val)

        train_losses = []
        train_preds = []
        val_losses = []
        val_preds = []

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

    def batcher(self, x, y):
        if len(x) < self.batch_size:
            raise ValueError(f"Batch size {self.batch_size} is greater than the number of samples {len(x)}")
        
        x_batches = np.split(x, len(x)//self.batch_size)
        y_batches = np.split(y, len(y)//self.batch_size)
        self.num_batch = len(x_batches)
        yield from zip(x_batches, y_batches)

    def compile(self, optimizer, loss, batch_size, metrics: list):
        self.batch_size = batch_size
        self.loss = loss
        self.set_optimizer(optimizer)
        self.metrics = metrics

        for metric in self.metrics:
            self.fit_output[f"{metric}"] = []
            self.fit_output[f"val_{metric}"] = []

    def calc_metrics(self, preds, labels):
        metrics_output = {}
        for metric in self.metrics:
            if metric == "accuracy":
                metrics_output['accuracy'] = calc_accuracy(
                    preds, labels)
        return metrics_output

    def set_optimizer(self, optimizer):
        for layer in self.layers:
            if layer.trainable:
                layer.optimizer = copy.deepcopy(optimizer)
    
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
