import copy


class Trainer:
    def __init__(self, model, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.model = model
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.set_optimizer()

    def set_optimizer(self):
        for layer in self.model:
            if layer.trainable:
                layer.optimizer = copy.deepcopy(self.optimizer)
                layer.initialize(self.weights_initializer, self.bias_initializer)


    def forward(self, x):
        for layer in self.model:
            output = layer.forward(x)
            x = output
        return output

    def backward(self, y):
        y = self.loss_layer.backward(y)

        for layer in reversed(self.model):
            output = layer.backward(y)
            y = output

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.model.append(layer)

    # def fit(self, epoch, train_data, val_data):
    def fit(self, epoch, train_data, ):
        self.train_data = train_data
        for i in range(1, epoch+1):
            print(f"{'-'*50}Epoch {i}{'-'*50}")

            train_loss = self.train_epoch()
            # val_loss = self.eval_epoch()

            self.train_losses.append(train_loss)
            # self.val_losses.append(val_loss)

            print(f"train loss: {train_loss:.2f}")
            # print(f"val loss: {val_loss:.2f}")

    def train_step(self, x, y):
        output = self.forward(x)
        loss = self.loss_layer.forward(output, y)
        self.backward(y)
        return loss

    def eval_step(self, x, y):
        output = self.forward(x)
        loss = self.loss_layer.forward(output, y)
        return loss

    def train_epoch(self):
        loss = 0.0
        for batch in self.train_data:
            x = batch[0]
            y = batch[1]
            loss += self.train_step(x, y)
        epoch_loss = loss/len(self.train_data)
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def eval_epoch(self):
        loss = 0.0
        for x, y in self.data:
            loss += self.eval_step(x, y)
        epoch_loss = loss/len(self.data)
        return epoch_loss
