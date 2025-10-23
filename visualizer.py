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
            self.reset(x_start, y_start)

    def reset(self, x_start=-19.0, y_start=-9.0):
            self.optx.reset()
            self.opty.reset()
            self.xs = [float(x_start)]
            self.ys = [float(y_start)]
            with torch.no_grad():
                z0 = self.surface.getSurface(torch.tensor([x_start], dtype=torch.float32),
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
    def __len__(self):
        return len(self.xs)

    def history(self):
            return np.array(self.xs), np.array(self.ys), np.array(self.zs)

class Animation_Controller:
    """ Class to hold the data for the animation (surface + optimizer paths) """
    def __init__(self, fig: Figure, surface:Surface_Interface, optimizers:list[Optimizer_Interface], x_lim =(-20,20), y_lim =(-10,10), total_frames=400):
        # store parameters
        self.surface = surface
        self.optimizers = {opt.__class__.__name__: opt for opt in optimizers}
        self.xlim = x_lim
        self.ylim = y_lim
        self.total_frames = total_frames
        self.fig = fig

        # create simulator instances (do not precompute trajectories)
        self.trajectories = [Optimizer_Path(opt, surface) for opt in optimizers]

        # render/build surface (avoid plt.figure)
        self.ax = fig.add_subplot(111, projection='3d')
        ax = self.ax
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self._plot_surface_()

        # Initialize lines and points for each optimizer
        self.lines = []
        self.points = []
        for label in self.optimizers.keys():
            line, = ax.plot([], [], [], alpha=1, lw=2, label=label)
            point, = ax.plot([], [], [], color=line.get_color(), alpha=1, marker='o', markersize=5)
            self.lines.append(line)
            self.points.append(point)
        ax.legend() #Not sure if I need to call this after adding new optimizers

        # Create the matplotlib animation
        self.anim = animation.FuncAnimation(fig, func = self._update_, init_func=self._reset_, frames=total_frames, interval=25, blit=True)

    def _plot_surface_(self):

        # Create surface mesh
        np_X, np_Y = np.meshgrid(np.arange(-20, 20, 0.2), np.arange(-10, 10, 0.2))
        X = torch.tensor(np_X, dtype=torch.float32)
        Y = torch.tensor(np_Y, dtype=torch.float32)
        Z = self.surface.getSurface(X, Y).numpy()

        # Add surface to plot
        ax = self.ax
        ax.plot_surface(np_X, np_Y, Z, cmap='plasma', alpha=0.3, linewidth=0, antialiased=False)
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        ax.set_zlim(Z.min(), Z.max())


    def _reset_(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return self.lines + self.points

    # This method allows the user to skip to any time frame ahead of time, since this method will
    # simulate all the steps up to that frame if not already done
    def _update_(self, frame):
        trajectories = self.trajectories
        lines = self.lines
        points = self.points

        for idx, traj in enumerate(trajectories):
            while len(traj) < frame:
                traj.step()
            xs, ys, zs = traj.history()
            pointStartIdx= frame-1 if frame>0 else 0

             # TODO: optimize by only updating new data if possible
            self.lines[idx].set_data(xs[:frame], ys[:frame])
            self.lines[idx].set_3d_properties(zs[:frame])
            self.points[idx].set_data(xs[pointStartIdx:frame], ys[pointStartIdx:frame])
            self.points[idx].set_3d_properties(zs[pointStartIdx:frame])
        return lines + points
    
    def set_range(self, xlim:tuple, ylim:tuple):
        self.xlim = xlim
        self.ylim = ylim
        self._plot_surface_()
    
    def add_optimizer(self, optimizer:Optimizer_Interface):
        name = optimizer.__class__.__name__
        self.optimizers[name] = optimizer
        self.trajectories.append(Optimizer_Path(optimizer, self.surface))

        line, = self.ax.plot([], [], [], alpha=1, lw=2, label=name)
        point, = self.ax.plot([], [], [], color=line.get_color(), alpha=1, marker='o', markersize=5)
        self.lines.append(line)
        self.points.append(point)
    
    def remove_optimizer(self, name:str):
        self.optimizers.pop(name)
        
        # find and remove corresponding trajectory
        for i, traj in enumerate(self.trajectories):
            if traj.optx.__class__.__name__ == name:
                self.trajectories.pop(i)
                break
    
    """ Pause the animation """
    def pause(self):
        self.anim.pause()

    """ 
        Resume the animation, if all the frames rendered is equal to total_frames, 
        then simulate/render more frames without indefinitly 
    """
    def resume(self):
        self.anim.resume()

    """ Set total number of frames to render """
    def set_total_frames(self, total_frames:int):
        self.total_frames = total_frames
    
    def seek_frame(self, frame:int):
        self.anim.pause()
        self._update_(frame)
        self.fig.canvas.draw_idle()
        


class Optimizer_Selector(customtkinter.CTkFrame):
    """Frame for selecting optimizers to visualize."""
    def __init__(self, master, optimizers, title="Select Optimizers", command=None):
        super().__init__(master)
        self.optimizers = list(optimizers)
        self._command = command
        self.checkboxes = []
        self.title = customtkinter.CTkLabel(master=self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.grid_columnconfigure(0, weight=1)

        for i, name in enumerate(self.optimizers):
            checkbox = customtkinter.CTkCheckBox(master=self, text=name, command=self._notify_selection_change)
            checkbox.grid(row=i+1, column=0, padx=20, pady=(20, 0), sticky="w")
            self.checkboxes.append(checkbox)

    def select_all(self, notify=True):
        for checkbox in self.checkboxes:
            checkbox.select()
        if notify:
            self._notify_selection_change()

    def get_selected_optimizers(self):
        selected_opts = []
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.get():
                selected_opts.append(self.optimizers[i])
        return selected_opts

    def _notify_selection_change(self):
        if self._command:
            self._command(self.get_selected_optimizers())

class GUI(customtkinter.CTk):
    """ Main GUI class for the optimizer visualizer application """
    def __init__(self, optimizer_classes, surface):
        super().__init__()
        self.title("Optimizer Visualizer")
        self.surface = surface
        self.optimizer_classes = list(optimizer_classes)
        self.optimizer_name_map = {opt.__name__: opt for opt in self.optimizer_classes}
        self._resize_timer = None
        self.anim = None
        self.animation_controller = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.animation_frame = customtkinter.CTkFrame(master=self)
        self.animation_frame.grid(row=0, column=0, sticky="nsew")
        # make a container frame for the animation so layout is stable
        self.animation_frame.grid_rowconfigure(0, weight=1)
        self.animation_frame.grid_columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(9, 7))

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.animation_frame)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")

        names = list(self.optimizer_name_map.keys())
        self.optimizer_selector = Optimizer_Selector(
            master=self,
            optimizers=names,
            command=self.on_optimizer_selection_change,
        )
        self.optimizer_selector.grid(row=0, column=1, padx=10, pady=10, sticky="new")
        self.optimizer_selector.select_all(notify=False)
        self._update_animation(names)

        self.bind("<Configure>", self.on_resize)

    # Stop animation during resize to avoid performance issues
    def on_resize(self, event):
        if self._resize_timer:
            self.after_cancel(self._resize_timer)
        if self.anim is not None:
            self.anim.pause()

        self._resize_timer = self.after(500, self.on_resize_complete)

    # Resume animation after resize is complete
    def on_resize_complete(self):
        if self.anim is not None:
            self.anim.resume()
        self._resize_timer = None

    def on_optimizer_selection_change(self, selected_names):
        self._update_animation(selected_names)

    def _update_animation(self, selected_names):
        if self.anim is not None:
            if hasattr(self.anim, "event_source") and self.anim.event_source:
                self.anim.event_source.stop()
            self.anim = None
        self.animation_controller = None

        self.figure.clf()

        if not selected_names:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Select at least one optimizer to visualize.", ha="center", va="center")
            ax.set_axis_off()
            self.canvas.draw()
            return

        optimizers = [self.optimizer_name_map[name]() for name in selected_names]
        self.animation_controller = Animation_Controller(
            fig=self.figure,
            surface=self.surface,
            optimizers=optimizers,
        )
        self.anim = self.animation_controller.anim
        self.canvas.draw_idle()



if __name__ == '__main__':
    from Optimizers import optimizer_classes
    from Surfaces import surface_classes

    surface = surface_classes[0]()

    gui = GUI(surface=surface, optimizer_classes=optimizer_classes)
    gui.mainloop()
