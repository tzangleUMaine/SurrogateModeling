import numpy as np

class DataHandler:
    def __init__(self, data: np.array):
        self.min = data.min()
        self.max = data.max()
        self.ptp = self.max-self.min

    def scale(self, data):
        return (data - self.min)/self.ptp
    
    def unscale(self, data):
        return data*self.ptp + self.min