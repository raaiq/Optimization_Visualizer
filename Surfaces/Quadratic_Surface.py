from Interfaces import Surface_Interface
import numpy as np
import torch

class Quadratic_Surface(Surface_Interface):
    """ A simple quadratic surface (z = x^2 + y^2) with multiple Gaussian peaks added to create many local minima """
    def __init__(self, xrange=(-10,10), yrange=(-10,10), no_of_peaks=50):
        self.num_peaks = no_of_peaks
        self.x_range = xrange
        self.y_range = yrange
        self.x_centers = np.random.uniform(self.x_range[0], self.x_range[1], self.num_peaks)
        self.y_centers = np.random.uniform(self.y_range[0], self.y_range[1], self.num_peaks)
        # Setting widths of the peaks
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