from Interfaces import Optimizer_Interface

class SGD(Optimizer_Interface):
    def __init__(self, lr=0.1):
        super().__init__(learning_rate=lr)
        self.reset()
    
    def reset(self):
        pass

    def update(self, grad):
        return self.config["learning_rate"] * grad