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

    def plot_data(self,name="Data"):
        plt.figure(name)
        plt.title(name)
        plt.scatter(self.x, self.y, color = "black")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")

    def plot_model(self,x_vals,name="Model"):
        y_hat = np.zeros(len(x_vals[:,0]))
        for i, x in enumerate(x_vals):
            y_hat[i] = self.evaluate(x)
        plt.figure(name)
        plt.title(name)
        plt.scatter(x_vals, y_hat, color = "black")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y_hat")