class BaseLayer:
    def __init__(self, ):
        self.trainable = False
        self.initializable = False
        self.training = False

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
