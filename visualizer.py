import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class Surface:
    def __init__(self, x_range, y_range, step=0.1, num_peaks=5):    
        # Generate random parameters for each Gaussian
        self.num_peaks = num_peaks
        self.amplitudes = np.random.uniform(4, 5, num_peaks)  # Varying amplitudes, some negative
        self.x_centers = np.random.uniform(x_range[0], x_range[1], num_peaks)
        self.y_centers = np.random.uniform(y_range[0], y_range[1], num_peaks)
        self.x_widths = np.random.uniform(0.5, 2, num_peaks)
        self.y_widths = np.random.uniform(0.5, 2, num_peaks)
        x = np.arange(x_range[0]*1.2, x_range[1]*1.2, step)
        y = np.arange(y_range[0]*1.2, y_range[1]*1.2, step)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.zeros_like(self.X)
        for i in range(num_peaks):
            self.Z += self.amplitudes[i] * np.exp(-( (self.X - self.x_centers[i])**2/(2*self.x_widths[i]**2) + (self.Y - self.y_centers[i])**2 / (2 * self.y_widths[i]**2)))
        self.Z += (self.X**2 + self.Y**2) / 100  # Adding a parabolic component for variation
    
    def get_Z(self, x,y):
        z =0
        for i in range(self.num_peaks):
            z += self.amplitudes[i] * np.exp(-( (x - self.x_centers[i])**2/(2*self.x_widths[i]**2) + (y - self.y_centers[i])**2 / (2 * self.y_widths[i]**2)))
        z+= (x**2 + y**2) / 100  # Adding a parabolic component for variation
        return z
    
    def get_slope(self, x, y):
        x_slope = 0
        y_slope = 0 

        for i in range(self.num_peaks):
            common_term = self.amplitudes[i] * np.exp(-( (x - self.x_centers[i])**2/(2*self.x_widths[i]**2) + (y - self.y_centers[i])**2 / (2 * self.y_widths[i]**2)))
            x_slope += common_term*(-(x - self.x_centers[i]) / (self.x_widths[i]**2))
            y_slope += common_term*(-(y - self.y_centers[i]) / (self.y_widths[i]**2))
        x_slope += x / 50  # Derivative of the parabolic component
        y_slope += y / 50  # Derivative of the parabolic component
        return x_slope, y_slope
    def get_starting_point(self):
        return self.x_centers[0], self.y_centers[0], self.get_Z(self.x_centers[0], self.y_centers[0])

    def get_surface(self):
        return self.X, self.Y, self.Z

    def animate(self, ax, func, frames=100, interval=100, cmap='viridis'):
        def update(frame):
            ax.cla()
            self.update_Z(lambda x, y: func(x, y, frame))
            self.plot_surface(ax, cmap=cmap)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

        ani = animation.FuncAnimation(plt.gcf(), update, frames=frames, interval=interval)
        return ani
def gaussian_sum_function(x, y, num_peaks=5):
    """
    Generates a function as a sum of 2D Gaussian functions
    with random amplitudes and centers.
    """
    z = np.zeros_like(x)
    
    # Generate random parameters for each Gaussian
    amplitudes = np.random.uniform(-5, 5, num_peaks)  # Varying amplitudes, some negative
    x_centers = np.random.uniform(np.min(x), np.max(x), num_peaks)
    y_centers = np.random.uniform(np.min(y), np.max(y), num_peaks)
    widths = np.random.uniform(0.5, 2, num_peaks)
    
    # Sum the Gaussian functions
    for i in range(num_peaks):
        z += amplitudes[i] * np.exp(-((x - x_centers[i])**2 + (y - y_centers[i])**2) / (2 * widths[i]**2))
    return z

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surface = Surface(x_range=(-20,20), y_range=(-10,10), step=0.1, num_peaks=70)
X, Y, Z = surface.get_surface()
# Z= Z+(X**2+Y**2)/100

# plt.tight_layout()

# Create a Matplotlib figure and 3D axes
surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=.5)


x_start, y_start, z_start = surface.get_starting_point()
mx =0
my =0
vx=0
vy=0
t =0
x_run = np.array([x_start])
y_run = np.array([y_start])
z_run = np.array([z_start])



line, = ax.plot(x_run, y_run, z_run, color='black', linewidth=2)
def data_gen():
    for _ in range(200):
        yield _
def run(data):
    global x_run, y_run, z_run, line, mx, my,vx, vy, t
    # update the data
    dx,dy = surface.get_slope(x_run[-1], y_run[-1])
    step_size = .1
    b2x=0.9
    b2y=0.9
    bx =0.8
    by =0.8

    t = t + 1
    mx = bx*mx + (1-bx)*dx
    my = by*my + (1-by)*dy
    # mx = mx/(1-bx**t)
    # my = my/(1-by**t)
    

    vx= vx*b2x + (1-b2x)*dx*dx
    vy= vy*b2y + (1-b2y)*dy*dy
    # vx = vx/(1-b2x**t)
    # vy = vy/(1-b2y**t)

    print(f"t: {t}, Vx: {vx}, Vy: {vy}, Mx: {mx}, My: {my}, Bias Vx: {(1-b2x**t)}, Bias Vy: {(1-b2y**t)} Bias Mx: {(1-bx**t)}, Bias My: {(1-by**t)}\n")
    
    new_x = x_run[-1] - step_size * mx/(np.sqrt(vx)+1e-6)
    new_y = y_run[-1] - step_size * my/(np.sqrt(vy)+1e-6)
    new_z = surface.get_Z(new_x, new_y)
    x_run = np.append(x_run, new_x)
    y_run = np.append(y_run, new_y)
    z_run = np.append(z_run, new_z)
    line.set_data_3d([x_run, y_run, z_run])
    return line,


def init():
    global x_run, y_run, z_run, line,vx, vy,mx,my, t

    mx=0
    my=0
    vx=0
    vy=0
    t=0
    x_run = np.array([x_start])
    y_run = np.array([y_start]) 
    z_run = np.array([z_start])
    line.set_data_3d([x_run, y_run, z_run])
    return line,
# ax.plot(X2, Y2, Z2, color='black', linewidth=2)
ani = animation.FuncAnimation(fig, run, data_gen,interval=200, save_count=200 ,init_func=init)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()
