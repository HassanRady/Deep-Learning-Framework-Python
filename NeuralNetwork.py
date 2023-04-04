import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self, ):
        x, y = self.data_layer.next()   
        self.label_tensor = y 

        for layer in self.layers:
            output = layer.forward(x)
            x = output

        return self.loss_layer.forward(output, y)
    
    def test_forward(self, x):
        for layer in self.layers:
            output = layer.forward(x)
            x = output
        return x
        

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

    def train(self, iterations):
        for i in range(1, iterations+1):
            print(f"{'-'*50}Epoch {i}{'-'*50}")
            loss = self.forward()
            self.loss.append(loss)
            print(f"Train Loss: {loss:.2f}")
            self.backward()
        return loss

    def test(self, x):
        return self.test_forward(x)

