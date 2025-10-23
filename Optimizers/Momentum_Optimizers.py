from Interfaces import Optimizer_Interface

class SGD_Momentum(Optimizer_Interface):
    def __init__(self, lr=0.1, k_momentum=0.5):
        super().__init__(learning_rate=lr)
        self.config["k_momentum"] = k_momentum
        self.reset()
    
    def reset(self):
        self.v = 0

    def update(self, grad):
        self.v = self.config["k_momentum"] * self.v + self.config["learning_rate"] * grad
        return self.v