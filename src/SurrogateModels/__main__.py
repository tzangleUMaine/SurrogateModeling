# import numpy as np
# import matplotlib.pyplot as plt
# import cokriging
# import mfs_functions

# def main():
#     # Start with a synthetic dataset
#     rng = np.random.RandomState(0)
#     fxn_ind = 1
#     dim,xmin,xmax=mfs_functions.function_bounds(fxn_ind)

#     n_lf = 30
#     n_hf = 4

#     x_lf = np.zeros((n_lf,dim), dtype=float)
#     x_hf = np.zeros((n_hf,dim), dtype=float)

#     # Generate the analytical functions
#     fxn_ind = 4
#     hf_fxn = mfs_functions.expensive_fxn(fxn_ind)
#     lf_fxn = mfs_functions.cheap_fxn(fxn_ind)


#     x_lf[:,0] = np.linspace(xmin,xmax,n_lf)
#     y_lf = lf_fxn.evaluate(x_lf)
#     x_hf[:,0] = np.linspace(xmin,xmax,n_hf)
#     y_hf = hf_fxn.evaluate(x_hf)
#     model = cokriging.CoKriging(x_lf,y_lf,x_hf,y_hf)

#     x_vals = np.ones((300,1))
#     x_vals[:,0] = np.linspace(xmin,xmax,300)

#     #plot the model information
#     plot_name = "Analytical Models"
#     hf_fxn.plot_model(x_vals, plot_name, color = "blue", linestyle = 'solid', label = "hf_fxn")
#     lf_fxn.plot_model(x_vals, plot_name, color = "black", linestyle = 'dashed', label = "lf_fxn")

#     plot_name = "Cokriging Model"
#     hf_fxn.plot_model(x_vals,plot_name, color = "blue", linestyle = (0, (5, 10)), label = "hf_fxn")
#     lf_fxn.plot_model(x_vals,plot_name, color = "black", linestyle = (0, (5, 10)), label = "lf_fxn")
#     model.plot_data(plot_name, color_lf="black", color_hf="blue",
#                     label_hf = "HF", label_lf = "LF",
#                     marker_hf = '^',marker_lf="s")
#     model.plot_model(x_vals, name = plot_name, color="red",
#                     label = "Cokriging", linestyle='solid')
#     model.kriging_hf.plot_model(x_vals, name=plot_name, color="blue",
#                         label = "HF Kriging", linestyle='dotted')
#     model.kriging_lf.plot_model(x_vals, name=plot_name, color="black",
#                         label = "LF Kriging", linestyle='dashed')
#     #plt.show()
#     error = model.rms_error(hf_fxn,x_vals)
#     print(error)
# if __name__ == "__main__":
#     main()
