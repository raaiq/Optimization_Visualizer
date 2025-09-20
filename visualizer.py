import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


class ADAM_Optimization: 
    def __init__(self, lr=0.1):
        self.B1 = 0.99
        self.B2 = 0.7
        self.eps = 1e-8
        self.lr = lr
        self.reset()

    def reset(self):
        self.t =0
        self.m = 0
        self.v= 0
    
    def update(self, grad):
        self.t += 1
        self.m = self.B1*self.m + (1-self.B1)*grad
        self.v = self.B2*self.v + (1-self.B2)*(grad**2)

        m_hat = self.m/(1-self.B1**self.t)
        v_hat = self.v/(1-self.B2**self.t)

        update = self.lr * m_hat/(v_hat+self.eps)**.5
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
        z = (x**2 + y**2) / 100  # parabolic component
        for i in range(self.num_peaks):
            z += torch.exp(-( (x - self.x_centers[i])**2/(2*self.x_widths[i]**2) + (y - self.y_centers[i])**2 / (2 * self.y_widths[i]**2)))
        return z


def simulate_optimizer(optimizer, surface, no_of_frames=200, x_start=-19.0, y_start=-9.0):
    # use a deepcopy so optimizer states don't interfere
    optx = copy.deepcopy(optimizer)
    opty = copy.deepcopy(optimizer)

    optx.reset()
    opty.reset()


    xs = [float(x_start)]
    ys = [float(y_start)]
    with torch.no_grad():
        z0 = surface.getSurface(torch.tensor([x_start]), torch.tensor([y_start]))
    zs = [float(z0.item())]

    for _ in range(no_of_frames):
        x_t = torch.tensor([xs[-1]], dtype=torch.float32, requires_grad=True)
        y_t = torch.tensor([ys[-1]], dtype=torch.float32, requires_grad=True)
        z = surface.getSurface(x_t,y_t)
        z.backward()
        dx = x_t.grad.item()
        dy = y_t.grad.item()

        upd_x = optx.update(dx)
        upd_y = opty.update(dy)

        new_x = xs[-1] - upd_x
        new_y = ys[-1] - upd_y
        new_z = surface.getSurface(torch.tensor([new_x]), torch.tensor([new_y]))

        xs.append(float(new_x))
        ys.append(float(new_y))
        zs.append(float(new_z.item()))
    return np.array(xs), np.array(ys), np.array(zs)


def build_matplotlib_animation(surface, optimizers, names, colors, no_of_frames=400):
    # Create surface mesh
    np_X, np_Y = np.meshgrid(np.arange(-20, 20, 0.2), np.arange(-10, 10, 0.2))
    X = torch.tensor(np_X, dtype=torch.float32)
    Y = torch.tensor(np_Y, dtype=torch.float32)
    Z = surface.getSurface(X, Y).numpy()

    # simulate each optimizer
    sim_results = []
    for opt in optimizers:
        xs, ys, zs = simulate_optimizer(opt, surface, no_of_frames=no_of_frames)
        sim_results.append((xs, ys, zs))

    # build matplotlib figure
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np_X, np_Y, Z, cmap='plasma', alpha=0.6, linewidth=0, antialiased=False)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 10)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = []
    points = []
    for i, (xs, ys, zs) in enumerate(sim_results):
        line, = ax.plot([], [], [], color=colors[i], lw=2, label=names[i])
        point, = ax.plot([], [], [], marker='o', color=colors[i], markersize=5)
        lines.append(line)
        points.append(point)
    ax.legend()

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return lines + points

    def update(frame):
        # frame ranges from 0..no_of_frames
        for idx, (xs, ys, zs) in enumerate(sim_results):
            end = min(frame, len(xs)-1)
            lines[idx].set_data(xs[:end+1], ys[:end+1])
            lines[idx].set_3d_properties(zs[:end+1])
            points[idx].set_data(xs[end:end+1], ys[end:end+1])
            points[idx].set_3d_properties(zs[end:end+1])
        return lines + points

    frames_total = no_of_frames + 1
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames_total, interval=50, blit=False)

    plt.show()
    return anim


if __name__ == '__main__':
    surface = Surface()
    no_of_frames = 200
    optimizers = [ADAM_Optimization(lr=0.3), SGD(lr=0.3), RMSProp(lr=0.3), SGD_Momentum(lr=0.3)]
    names = ['ADAM', 'SGD', 'RMSProp', 'SGD_Mom']
    colors = ['black', 'red', 'green', 'blue']

    anim = build_matplotlib_animation(surface, optimizers, names, colors, no_of_frames=no_of_frames)

