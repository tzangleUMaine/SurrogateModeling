import numpy as np
import matplotlib.pyplot as plt
import model


class expensive_fxn(model.Model):
    def __init__(self, fxn_ind):
        self.fxn_ind = fxn_ind

    # Refactor to make this into a dictionary since its gross right now
    def evaluate(self, Xin):
        if self.fxn_ind == 1:
            y_samp = (6 * Xin - 2) ** 2 * np.sin(12 * Xin - 4)
        elif self.fxn_ind == 2:
            y_samp = np.sin(2 * np.pi * (Xin - 0.1)) + Xin**2
        elif self.fxn_ind == 3:
            y_samp = Xin * np.sin(Xin) / 10
        elif self.fxn_ind == 4:
            y_samp = np.cos(3.5 * np.pi * Xin) * np.exp(-1.4 * Xin)
        elif self.fxn_ind == 5:
            y_samp = np.zeros((Xin.shape[0], 1))
            y_samp[:, 0] = (
                4
                - 2.1 * Xin[:, 0] ** 4
                + (1.0 / 3.0) * Xin[:, 0] ** 6
                + Xin[:, 0] * Xin[:, 1]
                - 4 * Xin[:, 1] ** 2
                + 4.0 * Xin[:, 1] ** 4
            )
        elif self.fxn_ind == 6:
            y_samp = np.where(
                Xin < 0.5,
                2 * discontinuous_fxn(Xin) - 20 * Xin + 20,
                4 + 2 * discontinuous_fxn(Xin) - 20 * Xin + 20,
            )
        elif self.fxn_ind == 7:
            y_samp = (Xin - 1.414213) * np.square(periodic_fxn(Xin))
        elif self.fxn_ind == 8:
            y_samp = np.square(Xin) + np.square(np.sin(8 * np.pi * Xin + (np.pi / 10)))
        else:
            print("invalid function index")
        return y_samp

    def plot_data(self, name="Model", color="black", label="model", linestyle="solid"):
        # DRAGONS: not setup for higher dimensions
        dim, xmin, xmax = function_bounds(self.fxn_ind)
        x_vals = np.linspace(xmin, xmax, 300)
        y_hat = np.zeros(len(x_vals))
        for i, x in enumerate(x_vals):
            y_hat[i] = self.evaluate(x)
        plt.figure(name)
        plt.title(name)
        plt.plot(x_vals, y_hat, color=color, label=label, linestyle=linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")


# Calculate sample points from the specified "Cheap" test function
# This function example is from Forrester 2008
class cheap_fxn(model.Model):
    def __init__(self, fxn_ind):
        self.fxn_ind = fxn_ind

        # Note: I dont like this
        self.expensive_fxn = expensive_fxn(fxn_ind)

    def evaluate(self, Xin):
        if self.fxn_ind == 1:
            y_samp = 0.5 * self.expensive_fxn.evaluate(Xin) + 10 * (Xin - 0.5) - 5
        elif self.fxn_ind == 2:
            y_samp = np.sin(2 * np.pi * (Xin - 0.1))
        elif self.fxn_ind == 3:
            y_samp = Xin * np.sin(Xin) / 10 + Xin / 10
        elif self.fxn_ind == 4:
            y_samp = np.cos(3.5 * np.pi * Xin) * np.exp(-1.4 * Xin) + 0.75 * Xin**2
        elif self.fxn_ind == 5:
            y_samp = np.zeros(Xin.shape()[0], 1)
            y_samp[:, 0] = (
                4
                - 0.5 * Xin[:, 0] ** 4
                + (0.04) * Xin[:, 0] ** 6
                + 0.5 * Xin[:, 0] * Xin[:, 1]
                - 1.96 * Xin[:, 1] ** 2
                + 0.96 * Xin[:, 1] ** 4
            )
        # Meng Karniadikis functions
        # Discontinuous function
        elif self.fxn_ind == 6:
            y_samp = discontinuous_fxn(Xin)
        elif self.fxn_ind == 7:
            y_samp = periodic_fxn(Xin)
        elif self.fxn_ind == 8:
            y_samp = periodic_fxn(Xin)
        else:
            print("invalid function index")
        return y_samp

    def plot_data(self, name="Model", color="black", label="model", linestyle="solid"):
        # DRAGONS: not setup for higher dimensions
        dim, xmin, xmax = function_bounds(self.fxn_ind)
        x_vals = np.linspace(xmin, xmax, 300)
        y_hat = np.zeros(len(x_vals))
        for i, x in enumerate(x_vals):
            y_hat[i] = self.evaluate(x)
        plt.figure(name)
        plt.title(name)
        plt.plot(x_vals, y_hat, color=color, label=label, linestyle=linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")


# Function returns the bounds and input dimensions of the function of interest
def function_bounds(fxn_ind):
    if fxn_ind == 1:
        dim = 1
        xmin = 0
        xmax = 1
    elif fxn_ind == 2:
        dim = 1
        xmin = 0
        xmax = 1
    elif fxn_ind == 3:
        dim = 1
        xmin = 0
        xmax = 10
    elif fxn_ind == 4:
        dim = 1
        xmin = 0
        xmax = 1
    elif fxn_ind == 5:
        dim = 2
        xmin = np.zeros((dim, 1))
        xmax = np.zeros((dim, 1))
        xmin[0] = -2
        xmin[1] = -2
        xmax[0] = 2
        xmax[1] = 2
    elif fxn_ind == 6:
        dim = 1
        xmin = 0
        xmax = 1
    elif fxn_ind == 7:
        dim = 1
        xmin = 0
        xmax = 1
    elif fxn_ind == 8:
        dim = 1
        xmin = 0
        xmax = 1
    return dim, xmin, xmax


def discontinuous_fxn(Xin):
    y_samp = np.where(
        Xin < 0.5,
        (0.5 * (6 * Xin - 2) ** 2 * np.sin(12 * Xin - 4) + 10 * (Xin - 0.5) - 5),
        (3 + 0.5 * (6 * Xin - 2) ** 2 * np.sin(12 * Xin - 4) + 10 * (Xin - 0.5) - 5),
    )
    return y_samp


def periodic_fxn(Xin):
    return np.sin(8 * np.pi * Xin)
