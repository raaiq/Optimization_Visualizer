import copy
import torch
import numpy as np
import plotly.graph_objects as go


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

        m_hat = self.m/(1-self.B1**self.t)
        v_hat = self.v/(1-self.B2**self.t)

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
        z = (x**2 + y**2) / 100  # parabolic component
        for i in range(self.num_peaks):
            z += torch.exp(-( (x - self.x_centers[i])**2/(2*self.x_widths[i]**2) + (y - self.y_centers[i])**2 / (2 * self.y_widths[i]**2)))
        return z


def simulate_optimizer(optimizer, surface, no_of_frames=200, x_start=-19.0, y_start=-9.0):
    # use a deepcopy so optimizer states don't interfere
    opt = copy.deepcopy(optimizer)
    opt.reset()

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

        upd_x = opt.update(dx)
        upd_y = opt.update(dy)

        new_x = xs[-1] - upd_x
        new_y = ys[-1] - upd_y
        new_z = surface.getSurface(torch.tensor([new_x]), torch.tensor([new_y]))

        xs.append(float(new_x))
        ys.append(float(new_y))
        zs.append(float(new_z.item()))
    return np.array(xs), np.array(ys), np.array(zs)


def build_plotly_figure(surface, optimizers, names, colors, no_of_frames=200):
    # Create surface mesh
    np_X, np_Y = np.meshgrid(np.arange(-20, 20, 0.1), np.arange(-10, 10, 0.1))
    X = torch.tensor(np_X, dtype=torch.float32)
    Y = torch.tensor(np_Y, dtype=torch.float32)
    Z = surface.getSurface(X, Y).numpy()

    # simulate each optimizer
    sim_results = []
    for opt in optimizers:
        xs, ys, zs = simulate_optimizer(opt, surface, no_of_frames=no_of_frames)
        sim_results.append((xs, ys, zs))

    # Build initial figure: surface + one trace per optimizer
    fig = go.Figure()
    fig.add_trace(go.Surface(x=np_X, y=np_Y, z=Z, colorscale='Plasma', opacity=0.6, showscale=False, name='surface'))

    # add scatter traces (initial single point)
    for i, (xs, ys, zs) in enumerate(sim_results):
        fig.add_trace(go.Scatter3d(x=[xs[0]], y=[ys[0]], z=[zs[0]], mode='lines+markers',
                                   line=dict(color=colors[i], width=4), marker=dict(size=3), name=names[i]))

    # frames: update the scatter traces (trace indices 1..n)
    frames = []
    trace_indices = list(range(1, 1 + len(sim_results)))
    for k in range(1, no_of_frames + 1):
        data = []
        for xs, ys, zs in sim_results:
            data.append(go.Scatter3d(x=xs[:k+1], y=ys[:k+1], z=zs[:k+1], mode='lines+markers',
                                     line=dict(width=4)))
        frames.append(go.Frame(data=data, name=str(k), traces=trace_indices))

    fig.frames = frames

    # slider and play button
    steps = []
    for k in range(1, no_of_frames + 1):
        step = dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))], label=str(k))
        steps.append(step)

    sliders = [dict(active=0, pad=dict(t=50), steps=steps)]

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      width=900, height=700,
                      updatemenus=[dict(type='buttons', showactive=False, y=0.05, x=0.1,
                                        xanchor='right', yanchor='top',
                                        pad=dict(t=45, r=10),
                                        buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                                                 dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])])],
                      sliders=sliders)
    return fig


if __name__ == '__main__':
    surface = Surface()
    no_of_frames = 200
    optimizers = [ADAM_Optimization(), SGD(lr=0.1), RMSProp(lr=0.1), SGD_Momentum(lr=0.1)]
    names = ['ADAM', 'SGD', 'RMSProp', 'SGD_Mom']
    colors = ['black', 'red', 'green', 'blue']

    fig = build_plotly_figure(surface, optimizers, names, colors, no_of_frames=no_of_frames)
    fig.write_html("visualizer_output.html", auto_open=True)

