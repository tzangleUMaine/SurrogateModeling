import numpy as np
import matplotlib.pyplot as plt
import model
from sklearn.neighbors import KNeighborsRegressor
import typing


class KNNRegression(model.Model):
    def __init__(
        self,
        x_mat: np.ndarray[float],
        y_vec: np.ndarray[float],
        n_neighbors: float = 1,
        scale_input=False,
        bounds=None,
    ):
        super().__init__(x_mat, y_vec, bounds)
        self.scale_input = scale_input
        if scale_input:
            self.x = self.x_scale.transform(self.x)

        # Generate the knnr model using the input dataset
        self.n_neighbors = n_neighbors
        self.knn_regressor = KNeighborsRegressor(self.n_neighbors, weights="distance")
        self.knn_regressor.fit(self.x, self.y)

    def evaluate(self, x_new, scale=False):
        # Scale x input if necessary
        if self.scale_input:
            x_new = self.x_scale.transform(x_new)
        return self.knn_regressor.predict(x_new)
