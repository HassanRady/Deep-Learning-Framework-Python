import copy

class Model(object):
    def __init__(self) -> None:
        self.model = None
        self.loss_layer = None

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
            layer.initialize(*self.initializer)
        self.model.append(layer)

    def compile(self, optimizer, loss):
        self.set_optimizer(optimizer)
        self.loss_layer = loss

    def set_optimizer(self, optimizer):
        for layer in self.model:
            if layer.trainable:
                layer.optimizer = copy.deepcopy(optimizer)
