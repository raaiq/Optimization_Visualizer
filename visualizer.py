import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class ADAM_Optimization: 
    def __init__(self):
        self.B1 = 0.99
        self.B2 = 0.99
        self.eps = 1e-8
        self.reset()

    def reset(self):
        self.t =0
        self.m = 0
        self.v= 0
    
    def update(self, grad, lr=0.3):
        self.t += 1
        self.m = self.B1*self.m + (1-self.B1)*grad
        self.v = self.B2*self.v + (1-self.B2)*(grad**2)

        m_hat = self.m#/(1-self.B1**self.t)
        v_hat = self.v#/(1-self.B2**self.t)

        update = lr * m_hat/(v_hat+self.eps)**.5
        return update

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def reset(self):
        pass

    def update(self, grad):
        return self.lr * grad
    
class RMSProp:
    def __init__(self, lr=0.1, beta=0.9, eps=1e-8):
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
class SGD_Momentum:
    def __init__(self, lr=0.1, k_momentum=0.9):
        self.lr = lr
        self.k_momentum = k_momentum
        self.reset()
    
    def reset(self):
        self.v = 0

    def update(self, grad):
        self.v = self.k_momentum * self.v + self.lr * grad
        return self.v

class Surface:
    def __init__(self):
        self.num_peaks = 50
        self.x_range = (-20, 20)
        self.y_range = (-10, 10)
        self.x_centers = np.random.uniform(self.x_range[0], self.x_range[1], self.num_peaks)
        self.y_centers = np.random.uniform(self.y_range[0], self.y_range[1], self.num_peaks)
        self.x_widths = np.random.uniform(0.5, 2, self.num_peaks)
        self.y_widths = np.random.uniform(0.5, 2, self.num_peaks)

        self.x_centers = torch.tensor(self.x_centers, dtype=torch.float32)
        self.y_centers = torch.tensor(self.y_centers, dtype=torch.float32)
        self.x_widths = torch.tensor(self.x_widths, dtype=torch.float32)
        self.y_widths = torch.tensor(self.y_widths, dtype=torch.float32)
    
    def getSurface(self,x,y):
        # Configure the parameters for the 
        # print(f'{x_centers.shape} , {y_centers.shape}, {x_widths.shape}, {y_widths.shape}, {x.shape}, {y.shape}')


        z = (x**2 + y**2) / 100  # Start with a parabolic component for variation
        for i in range(self.num_peaks):
            rand = torch.rand(1)-.5
            #(rand)*5*
            z += torch.exp(-( (x - self.x_centers[i])**2/(2*self.x_widths[i]**2) + (y - self.y_centers[i])**2 / (2 * self.y_widths[i]**2)))
        return z


def setup_animation(optimizer,surface, fig, ax,color ="black", no_of_frames=200):
    no_of_frames = 200
    x_start = torch.tensor([-19], dtype=torch.float32)
    y_start = torch.tensor([-9], dtype=torch.float32)
    z_start = surface.getSurface(x_start, y_start)

    line, = ax.plot(x_start.numpy(force=True), y_start.numpy(force=True), z_start.numpy(force=True), color=color, linewidth=2)

    def data_gen():
        for _ in range(no_of_frames):
            yield _

    def init():
        optimizer.reset()
        line.set_data_3d([x_start.numpy(force=True), y_start.numpy(force=True), z_start.numpy(force=True)])
        return line,
    def run(data):
        nonlocal line
        X_line, Y_line, Z_line = line.get_data_3d()

        x_t = torch.tensor([X_line[-1]], dtype=torch.float32, requires_grad=True)
        y_t = torch.tensor([Y_line[-1]], dtype=torch.float32, requires_grad=True)
        z = surface.getSurface(x_t,y_t)
        z.backward()
        dx = x_t.grad.item()
        dy = y_t.grad.item()

        new_X = X_line[-1] - optimizer.update(dx)
        new_Y = Y_line[-1] - optimizer.update(dy)
        new_Z = surface.getSurface(torch.tensor([new_X]), torch.tensor([new_Y]))
        X_line =np.append(X_line, new_X) 
        Y_line= np.append(Y_line, new_Y) 
        Z_line= np.append(Z_line, new_Z.detach().numpy())
        line.set_data_3d([X_line, Y_line, Z_line])
        return line,
    ani = animation.FuncAnimation(fig, run, data_gen, interval=200, save_count=no_of_frames, init_func=init)
    
    return ani
        

# def main():

surface = Surface()
np_X, np_Y = np.meshgrid(np.arange(-20, 20, 0.1), np.arange(-10, 10, 0.1))
X= torch.tensor(np_X, dtype=torch.float32)
Y= torch.tensor(np_Y, dtype=torch.float32)
Z = surface.getSurface(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
surf = ax.plot_surface(np_X, np_Y, Z.numpy(), cmap='plasma', alpha=.5)

optimizer = ADAM_Optimization()
ani = setup_animation(optimizer, surface, fig, ax, no_of_frames=200)
ani2 = setup_animation(SGD(lr=0.1), surface, fig, ax, color='red', no_of_frames=200)
an3 = setup_animation(RMSProp(lr=0.1), surface, fig, ax, color='green', no_of_frames=200)
an4 = setup_animation(SGD_Momentum(lr=0.1), surface, fig, ax, color='blue', no_of_frames=200)

plt.show()

# main()
    
