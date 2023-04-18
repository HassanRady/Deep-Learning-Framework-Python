import copy

class NeuralNetwork:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self, ):
        x, y = self.data_layer.next()   
        self.label_tensor = y 
        for layer in self.layers:
            x = layer.forward(x)
        return self.loss_layer.forward(x, y)
    
    def test_forward(self, x):
        for layer in self.layers:
            output = layer.forward(x)
            x = output
        return x
        

    def backward(self, ):
        y = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            y = layer.backward(y)
            
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
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
