import numpy as np
import matplotlib.pyplot as plt
import model


class MFModel(model.Model):
    def __init__(self, x_lf, y_lf, x_hf, y_hf):
        # Expects input values of:
        # x_mat: matrix of explanatory variables
        # y_vec: vector of response variables
        super().__init__(x_hf, y_hf)
        self.x_lf = x_lf
        self.y_lf = y_lf

    def plot_data(
        self,
        name="Data",
        color_lf="black",
        color_hf="blue",
        label_hf="hf",
        label_lf="lf",
        marker_hf="^",
        marker_lf="s",
        ax: plt.subplot = None,
    ):
        if self.d == 1:
            if ax == None:
                fig = plt.figure(name)
                ax = fig.add_subplot()
            ax.set_title(name)
            ax.legend()
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.scatter(self.x, self.y, color=color_hf, label=label_hf, marker=marker_hf)
            ax.scatter(
                self.x_lf, self.y_lf, color=color_lf, label=label_lf, marker=marker_lf
            )
            return ax
        elif self.d == 2:
            if self.d == 2:
                if ax == None:
                    fig = plt.figure(name)
                    ax = fig.add_subplot(projection="3d")
                ax.set_title(name)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_zlabel("y(x)")

                ax.scatter(
                    self.x[:, 0],
                    self.x[:, 1],
                    self.y[:, 0],
                    color=color_hf,
                    label=label_hf,
                )
                ax.scatter(
                    self.x_lf[:, 0],
                    self.x_lf[:, 1],
                    self.y_lf[:, 0],
                    color=color_lf,
                    label=label_lf,
                )
                return ax
        else:
            print("ERROR: higher dimensional plotting not supported")

    # def plot_model(
    #     self,
    #     bounds,
    #     name="Model",
    #     color="black",
    #     label="model",
    #     linestyle="solid",
    #     ax=None,
    #     n_samples=100,
    # ):
    #     if self.d == 1:
    #         # Read in bounds
    #         x_min = bounds[0][0]
    #         x_max = bounds[0][1]

    #         # Generate the sample point locations and evaluate the model
    #         x_test = np.linspace(x_min, x_max, n_samples, dtype=np.float32)
    #         x_test.reshape(-1, 1)
    #         y_hat = self.evaluate(x_test)

    #         # Plot the function
    #         if ax == None:
    #             fig = plt.figure(name)
    #             ax = fig.add_subplot()

    #         ax.set_title(name)
    #         ax.legend()
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("y(x)")

    #         ax.plot(x_test, y_hat, color=color, label=label, linestyle=linestyle)
    #         return ax
    #     elif self.d == 2:
    #         # Read in bounds
    #         x1_min = bounds[0][0]
    #         x1_max = bounds[0][1]
    #         x2_min = bounds[1][0]
    #         x2_max = bounds[1][1]

    #         # Generate the sample point locations and meshgrid then evaluate the model
    #         X1, X2 = model.Model.generate_grid(
    #             x1_min, x1_max, x2_min, x2_max, n_samples
    #         )

    #         x_sample = np.column_stack([X1.ravel(), X2.ravel()])
    #         y_hat = self.evaluate(x_sample)
    #         y_hat = y_hat.reshape(n_samples, n_samples)

    #         # Plot the function
    #         if ax == None:
    #             fig = plt.figure(name)
    #             ax = fig.add_subplot(projection="3d")

    #         ax.set_title(name)
    #         ax.legend()
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("y(x)")

    #         ax.plot(X1, X2, y_hat, color=color, label=label, linestyle=linestyle)
    #         return ax

    def plot_model_error(
        self,
        x_vals,
        ground_truth,
        name="Model",
        color="black",
        label="model",
        linestyle="solid",
    ):
        error = np.zeros(len(x_vals[:, 0]))
        for i, x in enumerate(x_vals):
            error[i] = np.sqrt((self.evaluate(x) - ground_truth(x)) ** 2)
        plt.figure(name)
        plt.title(name)
        plt.plot(x_vals, error, color=color, label=label, linestyle=linestyle)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y(x)")
