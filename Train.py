import copy

class Trainer:
    def __init__(self, x, y, batch_size) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def train_step(self, x, y):
        output = self.forward(x)
        loss = self.loss_layer.forward(output, y)
        self.backward(y)
        return loss

    def train_epoch(self):
        loss = 0.0
        for batch in self.batcher():
            x = batch[0]
            y = batch[1]
            loss += self.train_step(x, y)
        epoch_loss = loss/len(self.train_data)
        return epoch_loss

    def batcher(self):
        for i in range(len(self.x)):
            yield self.x[i:i+self.batch_size], self.y[i:i+self.batch_size]