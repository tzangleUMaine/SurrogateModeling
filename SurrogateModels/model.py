import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self,x_mat,y_vec):
        # Expects input values of:
        # x_mat: matrix of explanatory variables
        # y_vec: vector of response variables
        self.x = x_mat
        self.y = y_vec

    def evaluate(self,x_in):
        raise NotImplementedError()

    def plot_data(self):
        plt.scatter(self.x, self.y, color = "black")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        _ = plt.title(
            "Input Data Set"
        )
        plt.show()