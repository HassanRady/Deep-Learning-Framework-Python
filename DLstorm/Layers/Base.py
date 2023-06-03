class BaseLayer:
    def __init__(self, ):
        self.trainable = False
        self.initializable = False
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

if __name__ == "__main__":
    pass
