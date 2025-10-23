from Interfaces import Optimizer_Interface
import numpy as np

class RMSProp(Optimizer_Interface):
    def __init__(self, lr=0.1, beta=0.99, eps=1e-8):
        super().__init__(learning_rate=lr)
        self.config["beta"] = beta
        self.config["eps"] = eps
        self.reset()
    
    def reset(self):
        self.v = 0

    def update(self, grad):
        beta = self.config["beta"]
        eps = self.config["eps"]
        lr = self.config["learning_rate"]

        self.v = beta * self.v + (1-beta)*(grad**2)
        update = lr * grad/(self.v+eps)**.5
        return update

class RProp(Optimizer_Interface):
    def __init__(self, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50.0, delta_init=0.1):
        super().__init__(learning_rate=0.0)  # Learning rate is not used in RProp
        self.config["eta_plus"] = eta_plus
        self.config["eta_minus"] = eta_minus
        self.config["delta_min"] = delta_min
        self.config["delta_max"] = delta_max
        self.config["delta_init"] = delta_init
        self.reset()
    
    def reset(self):
        self.prev_grad = 0
        self.delta = self.config["delta_init"]  # Initial step size

    def update(self, grad):
        # Extract config parameters for easier access
        eta_plus = self.config["eta_plus"]
        eta_minus = self.config["eta_minus"]
        delta_min = self.config["delta_min"]
        delta_max = self.config["delta_max"]

        sign = grad * self.prev_grad
        update = np.sign(grad)
        if sign > 0:
            self.delta = min(self.delta * eta_plus, delta_max)
        elif sign < 0:
            self.delta = max(self.delta * eta_minus, delta_min)
        
        update *= self.delta
        
        self.prev_grad = grad
        return update