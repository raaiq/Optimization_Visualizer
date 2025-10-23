from abc import ABCMeta, abstractmethod

class Optimizer_Interface(metaclass = ABCMeta):

    def __init__(self, learning_rate: float):
        self.config = {
            "learning_rate": learning_rate
            }
    
    @abstractmethod
    def reset(self):
        """
        Resets the optimizer's internal state 
        """
        pass

    @abstractmethod
    def update(self, grad):
        """
        Optimizes the gradient descent step: 'grad'

        Paramaters:
        grad: The slope calculated for a particular axis at a particular time step

        Returns
        The updated gradient descent step
        """
        pass

    def get_config(self):
        """
        Returns the configuration dictionary of the optimizer.
        """
        return self.config

class Surface_Interface(metaclass =ABCMeta):
    @abstractmethod
    def getSurface(self, x, y):
        """
        Calculates the Z value of the surface at a given (x, y) coordinate.

        Parameters:
        x: The x-coordinate (can be a tensor or numpy array).
        y: The y-coordinate (can be a tensor or numpy array).

        Returns:
        The z-value(s) of the surface at (x, y).
        """
        pass

