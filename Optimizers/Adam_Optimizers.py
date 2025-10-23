from Interfaces import Optimizer_Interface  

class ADAM_Optimization(Optimizer_Interface): 
    def __init__(self, lr=0.3, mbeta= 0.9, vbeta=0.999, eps=1e-8):
        super().__init__(learning_rate=lr)
        self.config["mbeta"] = mbeta
        self.config["vbeta"] = vbeta
        self.config["eps"] = eps
        self.reset()

    def reset(self):
        self.t =0
        self.m = 0
        self.v= 0
    
    def update(self, grad):
        self.t += 1

        # Extract config parameters for easier access
        mbeta = self.config["mbeta"]
        vbeta = self.config["vbeta"]
        eps = self.config["eps"]
        lr = self.config["learning_rate"]

        self.m = mbeta*self.m + (1-mbeta)*grad
        self.v = vbeta*self.v + (1-vbeta)*(grad**2)

        m_hat = self.m/(1-mbeta**self.t)
        v_hat = self.v/(1-vbeta**self.t)

        update = lr * m_hat/(v_hat+eps)**.5
        return update


class Adam_No_Correction(ADAM_Optimization):
    def __init__(self, lr=0.3):
        super().__init__(lr=lr)

    def update(self, grad):
        self.t += 1

                # Extract config parameters for easier access
        mbeta = self.config["mbeta"]
        vbeta = self.config["vbeta"]
        eps = self.config["eps"]
        lr = self.config["learning_rate"]

        self.m = mbeta*self.m + (1-mbeta)*grad
        self.v = vbeta*self.v + (1-vbeta)*(grad**2)

        update = lr * self.m/(self.v+eps)**.5
        return update