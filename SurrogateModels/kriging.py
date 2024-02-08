import numpy as np
import matplotlib.pyplot as plt
import model
import scipy.optimize

class Kriging(model.Model):
    def __init__(self,x_mat,y_vec):
        # Expects input values of:

        super().__init__(x_mat,y_vec)

        self.plot_data()

        self.d, self.n = np.shape(self.x)

        self.pre_psi = self.calculate_pre_psi_matrix()

        self.theta = self.calculate_hyper_params()

        self.psi, self.inv_psi = self.calculate_psi_matrix(self.theta)

        self.mu = self.calculate_kriging_mean(self.inv_psi)

        self.sigma = self.calculate_sigma(self.mu,self.inv_psi)

        self.b = self.calculate_b()

    def calculate_hyper_params(self):
        fun: lambda theta: (-self.calculate_mle(theta))
        theta_opt = scipy.optimize.fmin(func=fun,theta0=np.ones(self.d,1))
        return theta_opt

    def calculate_pre_psi_matrix(self):
        pre_psi_ij = np.zeros(self.n)
        for i in range(1,self.n):
            for j in range(1,self.n):
                diffX = self.x[i,:] - self.x[j,:]
                pre_psi_ij[i,j] = np.dot(diffX,diffX)
        return pre_psi_ij
    
    def calculate_kriging_mean(self,inv_psi):
        F = np.ones(self.n,1)
        mu = np.transpose(F)*self.inv_psi*self.y
        mu = mu/(np.transpose(F)*self.inv_psi*F)
        return mu
    
    def calculate_sigma(self,mu,inv_psi):
        F = np.ones(self.n,1)
        sigma = (1.0/self.n)*np.transpose(self.y-F*mu)*inv_psi*(self.y-F*mu)
        return sigma

    def calculate_b(self):
        b = self.psi_inv*(self.y-np.ones(self.n,1)*self.mu)
        return b

    def calculate_psi_matrix(self, theta):
        psi_ij = np.exp(-theta * self.pre_psi)
        inv_psi_ij = np.invert(psi_ij)
        return psi_ij,inv_psi_ij

    def calculate_mle(self,theta):
        psi,inv_psi = self.calculate_psi_matrix(theta)
        mu_hat = self.calculate_kriging_mean(inv_psi)
        sigma = self.calculate_sigma(inv_psi,mu_hat)
        mle = -0.5*(self.n*np.log(sigma)+np.log(np.linalg.det(psi)))
        return mle

    def evaluate(self, x_new):
        x_diff = x_new-self.x
        y_pred = self.b @ np.exp(-self.theta*(np.dot(x_diff,x_diff))) + self.mu
        return y_pred
