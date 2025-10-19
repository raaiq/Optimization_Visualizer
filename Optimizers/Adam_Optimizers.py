from Interfaces import Optimizer_Interface  

class ADAM_Optimization(Optimizer_Interface): 
    def __init__(self, lr=0.3, mbeta= 0.9, vbeta=0.999, eps=1e-8):
        self.mbeta = mbeta
        self.vbeta = vbeta
        self.eps = eps
        self.lr = lr
        self.reset()

    def reset(self):
        self.t =0
        self.m = 0
        self.v= 0
    
    def update(self, grad):
        self.t += 1
        self.m = self.mbeta*self.m + (1-self.mbeta)*grad
        self.v = self.vbeta*self.v + (1-self.vbeta)*(grad**2)

        m_hat = self.m/(1-self.mbeta**self.t)
        v_hat = self.v/(1-self.vbeta**self.t)

        update = self.lr * m_hat/(v_hat+self.eps)**.5
        return update


class Adam_No_Correction(ADAM_Optimization):
    def __init__(self, lr=0.3):
        super().__init__(lr=lr)

    def update(self, grad):
        self.t += 1
        self.m = self.mbeta*self.m + (1-self.mbeta)*grad
        self.v = self.vbeta*self.v + (1-self.vbeta)*(grad**2)

        update = self.lr * self.m/(self.v+self.eps)**.5
        return update