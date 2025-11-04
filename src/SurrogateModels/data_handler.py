import numpy as np

class DataHandler:
    def __init__(self, data: np.array):
        self.min = data.min()
        self.max = data.max()
        self.ptp = self.max-self.min

    def scale(self, data):
        return 2*((data - self.min)/self.ptp) - 1.0
    
    def unscale(self, data):
        return ((data+1)*self.ptp)/2 + self.min