import numpy as np
import matplotlib.pyplot as plt
import cokriging
import mfs_functions

def main():
    # Start with a synthetic dataset
    rng = np.random.RandomState(0)
    fxn_ind = 1
    dim,xmin,xmax=mfs_functions.function_bounds(fxn_ind)

    n_lf = 11
    n_hf = 4

    x_lf = np.zeros((n_lf,dim), dtype=float)
    x_hf = np.zeros((n_hf,dim), dtype=float)

    x_lf[:,0] = np.linspace(xmin,xmax,n_lf)
    y_lf = mfs_functions.cheap_test_function(fxn_ind,x_lf)
    x_hf[:,0] = np.linspace(xmin,xmax,n_hf)
    y_hf = mfs_functions.expensive_test_function(fxn_ind,x_hf)
    model = cokriging.CoKriging(x_lf,y_lf,x_hf,y_hf)
    
    x_vals = np.ones((300,1))
    x_vals[:,0] = np.linspace(0,1,300)

    model.plot_data("Cokriging Model")
    model.plot_model(x_vals,"Cokriging Model")
    model.kriging_hf.plot_model(x_vals, "Cokriging Model")
    model.kriging_lf.plot_model(x_vals,"Cokriging Model")
    model.kriging_diff.plot_model(x_vals,"Diff Model")
    plt.show()

if __name__ == "__main__":
    main()