import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self, x, y):
        for layer in self.layers:
            output = layer.forward(x)
            x = output
        return output
    
    def backward(self, ):
        y = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            output = layer.backward(y)
            y = output
            
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def fit(self, epoch, train_data, val_data):
        pass


    
    def train_step(self, x, y):
        output = self.forward(x)
        loss = self.loss_layer.forward(output, y)
        self.backward()
        return loss
    
    def eval_step(self, x, y):
        output = self.forward(x)
        loss = self.loss_layer.forward(output, y)
        return loss
    
    def train_epoch(self):
        loss = 0.0
        for x, y in self.data:
            loss += self.train_step(x, y)
        epoch_loss = loss/len(self.data)
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def eval_epoch(self):
        loss = 0.0
        for x, y in self.data:
            loss += self.eval_step(x, y)
        epoch_loss = loss/len(self.data)
        self.val_losses.append(epoch_loss)
        return epoch_loss


