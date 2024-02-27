import numpy as np
import matplotlib.pyplot as plt
import model

class MFModel(model.Model):
    def __init__(self,x_lf,y_lf,x_hf,y_hf):
        # Expects input values of:
        # x_mat: matrix of explanatory variables
        # y_vec: vector of response variables
        super().__init__(x_hf,y_hf)
        self.x_lf = x_lf
        self.y_lf = y_lf

    def plot_data(self):
        plt.scatter(self.x, self.y, color = "black",label = "hf")
        plt.scatter(self.x_lf,self.y_lf,color = "blue",label = "lf")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        _ = plt.title(
            "Input Data Set"
        )
        plt.show()

    def plot_model(self,x_vals):
        y_hat = np.zeros(len(x_vals[:,0]))
        for i, x in enumerate(x_vals):
            y_hat[i] = self.evaluate(x)
        plt.scatter(x_vals, y_hat, color = "black")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y_hat")
        _ = plt.title(
            "Model Values"
        )
        plt.show()