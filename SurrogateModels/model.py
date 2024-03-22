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

    def plot_data(self,name="Data",color="black",label = "model", marker = "o"):
        plt.figure(name)
        plt.title(name)
        plt.scatter(self.x, self.y, color = color, label = label, marker=marker)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")

    def plot_model(self,x_vals,name="Model", color="black",
                    label = "model", linestyle='solid'):
        y_hat = np.zeros(len(x_vals[:,0]))
        for i, x in enumerate(x_vals):
            y_hat[i] = self.evaluate(x)
        plt.figure(name)
        plt.title(name)
        plt.plot(x_vals, y_hat, color = color, label = label, linestyle = linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")