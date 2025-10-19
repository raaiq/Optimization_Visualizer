from Interfaces import Optimizer_Interface
import numpy as np

class RMSProp(Optimizer_Interface):
    def __init__(self, lr=0.1, beta=0.99, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.reset()
    
    def reset(self):
        self.v = 0

    def update(self, grad):
        self.v = self.beta * self.v + (1-self.beta)*(grad**2)
        update = self.lr * grad/(self.v+self.eps)**.5
        return update

class RProp(Optimizer_Interface):
    def __init__(self, lr=0.1, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50.0, delta_init=0.1):
        self.lr = lr
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_init = delta_init
        self.reset()
    
    def reset(self):
        self.prev_grad = 0
        self.delta = self.delta_init  # Initial step size

    def update(self, grad):
        sign = grad * self.prev_grad
        update = np.sign(grad)
        if sign > 0:
            self.delta = min(self.delta * self.eta_plus, self.delta_max)
        elif sign < 0:
            self.delta = max(self.delta * self.eta_minus, self.delta_min)
        
        update *= self.delta
        
        self.prev_grad = grad
        return update