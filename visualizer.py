import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import importlib.util
import tkinter as tk
import customtkinter
from Interfaces import Optimizer_Interface, Surface_Interface

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


def build_matplotlib_animation(surface, optimizers, names, fig, no_of_frames=400):
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

    # build matplotlib Figure (avoid plt.figure)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(np_X, np_Y, Z, cmap='plasma', alpha=0.3, linewidth=0, antialiased=False)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 10)
    ax.set_zlim(Z.min(), Z.max())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = []
    points = []
    for i in range(len(sims)):
        line, = ax.plot([], [], [], alpha=1, lw=2, label=names[i])
        point, = ax.plot([], [], [], color=line.get_color(), alpha=1, marker='o', markersize=5)
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

    # create the animation after the canvas exists (if any)
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=frames_total, interval=25, blit=False)

    return anim


class Optimizer_Selector(customtkinter.CTkFrame):
    """ Frame for selecting optimizers to visualize """
    def __init__(self, master, optimizers, title="Select Optimizers"):
        super().__init__(master)
        self.optimizers = optimizers
        self.checkboxes = []
        self.title = customtkinter.CTkLabel(master=self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.grid_columnconfigure(0, weight=1)

        for i, name in enumerate(self.optimizers):
            checkbox = customtkinter.CTkCheckBox(master=self, text=name)
            checkbox.grid(row=i+1, column=0, padx=20, pady=(20, 0), sticky="w")
            self.checkboxes.append(checkbox)
    
    def get_selected_optimizers(self):
        selected_opts = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.get():
                selected_opts.append(self.optimizers[i])
        return selected_opts

class GUI(customtkinter.CTk):
    """ Main GUI class for the optimizer visualizer application """
    def __init__(self, optimizers, surface, names):
        super().__init__()
        self.title("Optimizer Visualizer")

        self._resize_timer = None
        self.animation_frame = customtkinter.CTkFrame(master=self)
        self.animation_frame.grid(row=0, column=0, sticky="nsew")
        # make a container frame for the animation so layout is stable
        fig = Figure(figsize=(9, 7))

        self.canvas = FigureCanvasTkAgg(fig, master=self.animation_frame)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.anim = build_matplotlib_animation(optimizers=optimizers, surface=surface, names=names, fig=fig)


        self.optimizer_selector = Optimizer_Selector(master=self, optimizers=names)
        self.optimizer_selector.grid(row=0, column=1, padx=10, pady=10, sticky="new")
        self.bind("<Configure>", self.on_resize)
    
    # Stop animation during resize to avoid performance issues
    def on_resize(self, event):
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        self.anim.pause()

        self._resize_timer = self.after(500, self.on_resize_complete)

    # Resume animation after resize is complete
    def on_resize_complete(self):
        self.anim.resume()
        self._resize_timer = None




if __name__ == '__main__':
    from Optimizers import optimizer_classes
    from Surfaces import surface_classes

    optimizers = [opt() for opt in optimizer_classes]
    surface = surface_classes[0]()

    gui = GUI(surface=surface, optimizers=optimizers, names=[opt.__name__ for opt in optimizer_classes])
    gui.mainloop()
# ...existing