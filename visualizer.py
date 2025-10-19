import copy
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import importlib.util
import tkinter as tk
from Interfaces import Optimizer_Interface, Surface_Interface


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

class Optimizer_Path:
    """ Class to keep track of the path taken by an optimizer on a surface"""

    #TODO: Generalize to n-dimensions?
    def __init__(self, optimizer_proto: Optimizer_Interface, surface:Surface_Interface, x_start=-19.0, y_start=-9.0):
            # each axis needs its own optimizer state -> deep copy
            self.optx = copy.deepcopy(optimizer_proto)
            self.opty = copy.deepcopy(optimizer_proto)
            self.surface = surface
            self.optx.reset()
            self.opty.reset()
            self.xs = [float(x_start)]
            self.ys = [float(y_start)]
            with torch.no_grad():
                z0 = surface.getSurface(torch.tensor([x_start], dtype=torch.float32),
                                        torch.tensor([y_start], dtype=torch.float32))
            self.zs = [float(z0.item())]
    def step(self):
            # advance one step (compute gradients and update)
            x_t = torch.tensor([self.xs[-1]], dtype=torch.float32, requires_grad=True)
            y_t = torch.tensor([self.ys[-1]], dtype=torch.float32, requires_grad=True)
            z = self.surface.getSurface(x_t, y_t)

            # compute grads
            z.backward()
            dx = x_t.grad.item()
            dy = y_t.grad.item()

            # get updates separately for x and y
            upd_x = self.optx.update(dx)
            upd_y = self.opty.update(dy)
            new_x = self.xs[-1] - upd_x
            new_y = self.ys[-1] - upd_y
            with torch.no_grad():
                new_z = self.surface.getSurface(torch.tensor([new_x], dtype=torch.float32),
                                           torch.tensor([new_y], dtype=torch.float32))
            self.xs.append(float(new_x))
            self.ys.append(float(new_y))
            self.zs.append(float(new_z.item()))
    def history(self):
            return np.array(self.xs), np.array(self.ys), np.array(self.zs)


def build_matplotlib_animation(surface, optimizers, names, no_of_frames=400):
    """
    Build animation while computing optimizer steps on the fly (no full precomputation).
    """
    # Create surface mesh
    np_X, np_Y = np.meshgrid(np.arange(-20, 20, 0.2), np.arange(-10, 10, 0.2))
    X = torch.tensor(np_X, dtype=torch.float32)
    Y = torch.tensor(np_Y, dtype=torch.float32)
    Z = surface.getSurface(X, Y).numpy()

    # create simulator instances (do not precompute trajectories)
    sims = [Optimizer_Path(opt, surface) for opt in optimizers]

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
    for i in range(len(sims)):
        line, = ax.plot([], [], [], lw=2, label=names[i])
        point, = ax.plot([], [], [], marker='o', markersize=5)
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
        # advance each simulator by one step (unless it already reached no_of_frames)
        for s in sims:
            if len(s.history()[0]) < no_of_frames:
                s.step()
        # update plotted data from each simulator history
        for idx, s in enumerate(sims):
            xs, ys, zs = s.history()
            end = len(xs) - 1  # current last index
            lines[idx].set_data(xs[:end+1], ys[:end+1])
            lines[idx].set_3d_properties(zs[:end+1])
            points[idx].set_data(xs[end:end+1], ys[end:end+1])
            points[idx].set_3d_properties(zs[end:end+1])
        return lines + points

    frames_total = no_of_frames + 1
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames_total, interval=50, blit=False)
    plt.show()
    return anim
    # # embed into Tk
    # canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    # widget = canvas.get_tk_widget()
    # widget.pack(fill='both', expand=True)
    # toolbar = NavigationToolbar2Tk(canvas, parent_frame)
    # toolbar.update()
    # canvas._tkcanvas.pack(fill='x')

    # # keep reference to anim and canvas on the frame so GC does not stop the animation
    # parent_frame._anim = anim
    # parent_frame._canvas = canvas
    # return anim, canvas


# if __name__ == '__main__':
#     surface = Surface()
#     no_of_frames = 200
#     optimizers = [ADAM_Optimization(lr=0.3), SGD(lr=0.3), RMSProp(lr=0.3), SGD_Momentum(lr=0.3)]
#     names = ['ADAM', 'SGD', 'RMSProp', 'SGD_Mom']
#     colors = ['black', 'red', 'green', 'blue']

#     anim = build_matplotlib_animation(surface, optimizers, names, colors, no_of_frames=no_of_frames)



def load_custom_optimizers_from_file(path):
    """
    Load a user-provided Python file that should expose either:
      - get_optimizers() -> list of (instance, name, color)
      - CUSTOM_OPTIMIZERS -> list of (instance, name, color)
    Returns list of (instance, name, color) or raises Exception.
    """
    spec = importlib.util.spec_from_file_location("custom_optim_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "get_optimizers") and callable(mod.get_optimizers):
        items = mod.get_optimizers()
    elif hasattr(mod, "CUSTOM_OPTIMIZERS"):
        items = mod.CUSTOM_OPTIMIZERS
    else:
        raise ValueError("Module must define get_optimizers() or CUSTOM_OPTIMIZERS list")

    # Normalize and validate
    normalized = []
    for it in items:
        if isinstance(it, dict):
            inst = it.get("instance")
            name = it.get("name")
            color = it.get("color", "black")
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            inst = it[0]
            name = it[1]
            color = it[2] if len(it) > 2 else "black"
        else:
            raise ValueError("Each optimizer entry must be (instance, name[, color]) or dict")

        if not (hasattr(inst, "update") and callable(inst.update) and hasattr(inst, "reset")):
            raise ValueError(f"Optimizer {name} does not implement required API (reset, update)")

        normalized.append((inst, name, color))
    return normalized


def run_gui():
    surface = Surface()
    no_of_frames = 200

    # default optimizers
    initial = [
        (ADAM_Optimization(lr=0.3), 'ADAM', 'black'),
        (RMSProp(lr=0.3), 'RMSProp', 'green'),
        (SGD_Momentum(lr=0.3), 'SGD_Mom', 'blue'),
    ]
    current_opts = initial.copy()

    def on_load():
        path = filedialog.askopenfilename(
            title="Select optimizer Python file",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            loaded = load_custom_optimizers_from_file(path)
            # deep-copy instances so states don't carry over between runs
            current_opts.clear()
            for inst, name, color in loaded:
                current_opts.append((copy.deepcopy(inst), name, color))
            messagebox.showinfo("Loaded", f"Loaded {len(current_opts)} optimizer(s) from {path}")
        except Exception as e:
            messagebox.showerror("Error loading module", str(e))

    def on_run():
        if not current_opts:
            messagebox.showwarning("No optimizers", "No optimizers to run. Load one first or use defaults.")
            return
        # prepare lists
        optimizers = [copy.deepcopy(inst) for inst, _, _ in current_opts]
        names = [name for _, name, _ in current_opts]
        colors = [color for _, _, color in current_opts]

        # close GUI before running matplotlib (so event loops don't clash)
        root.destroy()
        build_matplotlib_animation(surface, optimizers, names, colors, no_of_frames=no_of_frames)

    def on_reset_defaults():
        current_opts.clear()
        for inst, name, color in initial:
            current_opts.append((copy.deepcopy(inst), name, color))
        messagebox.showinfo("Reset", "Reverted to default optimizers")

    root = tk.Tk()
    root.title("Optimizer Visualizer - Load Custom Optimizers")

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack()

    lbl = tk.Label(frm, text="Load a Python file that exposes get_optimizers() or CUSTOM_OPTIMIZERS")
    lbl.pack(pady=(0,8))

    btn_load = tk.Button(frm, text="Load optimizer file...", width=30, command=on_load)
    btn_load.pack(pady=4)

    btn_defaults = tk.Button(frm, text="Reset to defaults", width=30, command=on_reset_defaults)
    btn_defaults.pack(pady=4)

    btn_run = tk.Button(frm, text="Run animation", width=30, command=on_run)
    btn_run.pack(pady=(12,4))

    btn_quit = tk.Button(frm, text="Quit", width=30, command=root.destroy)
    btn_quit.pack(pady=4)

    root.mainloop()



if __name__ == '__main__':
    from Optimizers import optimizer_classes
    from Surfaces import surface_classes

    optimizers = [opt() for opt in optimizer_classes]
    surface = surface_classes[0]() 
    anim = build_matplotlib_animation(surface, optimizers, [opt.__name__ for opt in optimizer_classes], no_of_frames=400)
    anim
# ...existing