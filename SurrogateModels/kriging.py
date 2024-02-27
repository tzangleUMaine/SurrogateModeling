import numpy as np
import matplotlib.pyplot as plt
import model
import scipy.optimize

class Kriging(model.Model):
    def __init__(self,x_mat,y_vec):
        # Expects input values of:

        super().__init__(x_mat,y_vec)

        #self.plot_data()

        self.n, self.d = np.shape(self.x)

        self.pre_psi = self.calculate_pre_psi_matrix()

        self.theta = self.calculate_hyper_params()

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.inv_psi,self.mu)

        self.b = self.calculate_b()

    def calculate_hyper_params(self):
        fun = lambda theta: (-self.calculate_mle(theta))
        theta0=np.ones((self.d,1), dtype=float)
        theta_opt = scipy.optimize.fmin(fun, theta0)
        return theta_opt

    def calculate_pre_psi_matrix(self):
        pre_psi_ij = np.zeros((self.n,self.n),dtype=float)
        for i in range(0,self.n):
            for j in range(0,self.n):
                diffX = self.x[i,:] - self.x[j,:]
                pre_psi_ij[i,j] = np.dot(diffX,diffX)
        return pre_psi_ij
    
    def calculate_kriging_mean(self,inv_psi):
        F = np.ones((self.n,1))
        mu = np.transpose(F) @ inv_psi @ self.y
        mu = mu/(np.transpose(F) @ inv_psi @ F)
        return mu
    
    def calculate_sigma(self,inv_psi,mu):
        F = np.ones((self.n,1))
        sigma = (1.0/self.n) * np.transpose(self.y-F*mu) @ inv_psi @ (self.y-F*mu)
        return sigma

    def calculate_b(self):
        b = self.inv_psi @ (self.y-np.ones((self.n,1))*self.mu)
        return b

    def calculate_psi_matrix(self, theta):
        psi_ij = np.exp(-theta * self.pre_psi)
        inv_psi_ij = np.linalg.inv(psi_ij)
        return psi_ij,inv_psi_ij

    def calculate_mle(self,theta):
        psi,inv_psi = self.calculate_psi_matrix(theta)
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi,mu_hat)
        mle = -0.5*(self.n*np.log(sigma)+np.log(np.linalg.det(psi)))
        return mle

    def evaluate(self, x_new):
        F = np.ones((self.n,1))
        x_diff = F*x_new-self.x
        y_pred = np.exp(-self.theta*(x_diff*x_diff))
        y_pred = y_pred * self.b
        sum_y = np.sum(y_pred)
        return np.sum(y_pred)+self.mu

class DiffKriging(Kriging):    
    def __init__(self,x_mat,yh, yl):
        self.x = x_mat

        # Store the size of the inputs
        self.n, self.d = np.shape(self.x)

        # Initialize hyperparameters
        self.scale_factor = 1.0
        self.theta = np.ones((self.d,1), dtype=float)

        # Set the high fidelity and low fidelity data
        self.yh = yh        
        self.yl = yl

        # Set y vector to be difference of two data streams
        self.y = self.yh - self.scale_factor * self.yl

        self.pre_psi = self.calculate_pre_psi_matrix()

        self.theta, self.scale_factor = self.calculate_hyper_params(self.theta,self.scale_factor)

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.inv_psi,self.mu)

        self.b = self.calculate_b()

    def fun(self, params):
            theta = params[0:-1]
            scale_factor = params[-1]
            return (-self.calculate_mle(theta,scale_factor))
    
    def calculate_hyper_params(self,theta0,scale_factor0):
        
        init_vars = np.vstack((theta0,scale_factor0))
        result = scipy.optimize.fmin(self.fun, init_vars)
        theta_opt, scale_factor_opt = result
        return theta_opt, scale_factor_opt
    
    def calculate_mle(self,theta,scale_factor):
        psi,inv_psi = self.calculate_psi_matrix(theta)
        self.y = self.yh - scale_factor * self.yl
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi,mu_hat)
        mle = -0.5*(self.n*np.log(sigma)+np.log(np.linalg.det(psi)))
        return mle
    
def near_zero_fix(array):
    for val in array:
        if(val<1e-3):
            val = 1e-3
    return array