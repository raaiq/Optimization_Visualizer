from Interfaces import Optimizer_Interface

class SGD_Momentum(Optimizer_Interface):
    def __init__(self, lr=0.1, k_momentum=0.5):
        self.lr = lr
        self.k_momentum = k_momentum
        self.reset()
    
    def reset(self):
        self.v = 0

    def update(self, grad):
        self.v = self.k_momentum * self.v + self.lr * grad
        return self.v