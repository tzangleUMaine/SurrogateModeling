import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import mf_model
import kriging

class CoKriging(mf_model.MFModel):
    def __init__(self,x_lf,y_lf,x_hf,y_hf,scale_fixed = False):

        # Init the multifidelity model data
        super().__init__(x_lf,y_lf,x_hf,y_hf)

        # Store the shapes of the input matrices
        self.n_hf, self.d = np.shape(self.x)
        self.n_lf, __ = np.shape(self.x_lf)
        self.n = self.n_lf + self.n_hf

        # Set up the low fidelity kriging model
        self.kriging_lf = kriging.Kriging(self.x_lf,self.y_lf)

        # Match find values for the low fidelity data set at exactly the same 
        # locations as the high fidelity data
        y_lf_hat = np.zeros((self.n_hf,1),float)
        for i,x in enumerate(self.x):
            y_lf_hat[i,0] = self.kriging_lf.evaluate(x)

        # Set up kriging model of the difference between the low and
        # high fidelity data set
        self.kriging_diff = kriging.DiffKriging(self.x,self.y,y_lf_hat,scale_fixed)

        # Find a kriging model for the high fidelity data set
        # This is not completely necessary, may be removed later for efficiency
        self.kriging_hf = kriging.Kriging(self.x,self.y)

        # Set up the cokriging covariance matrix
        psi_ec, psi_ce = self.calculate_psi_matrix()
        psic_e = self.calculate_pre_psi_matrix(self.x)
        psic_e = np.exp(-self.kriging_lf.theta*psic_e)

        self.C, self.inv_C = self.calculate_covariance(psi_ce,psi_ec,psic_e)
        
        # Assemble the combined y vector of cheap and expensive values
        self.yck = np.vstack((self.y_lf, self.y))

        # Find the mean
        self.mean = self.calculate_mean()

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
    
    def calculate_pre_psi_matrix(self,X):
        n,d = np.shape(X)
        pre_psi_ij = np.zeros((n,n),dtype=float)
        for i in range(n):
            for j in range(n):
                diffX = X[i,:] - X[j,:]
                pre_psi_ij[i,j] = np.dot(diffX,diffX)
        return pre_psi_ij
    
    def calculate_mean(self):
        F = np.ones((self.n,1))
        mu = np.transpose(F) @ self.inv_C @ self.yck
        mu = mu/(np.transpose(F) @ self.inv_C @ F)
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

        F = np.ones((self.n_lf,self.d))
        diff_norm = x_new*F - self.x_lf
        diff_norm_dot = diff_norm * diff_norm
        psic_new = np.exp(-self.kriging_lf.theta*diff_norm_dot)

        F = np.ones((self.n_hf,self.d))
        diff_norm = x_new*F - self.x
        diff_norm_dot = diff_norm * diff_norm
        psie_new = np.exp(-self.kriging_lf.theta*diff_norm_dot)
        psid_new = np.exp(-self.kriging_diff.theta*diff_norm_dot)

        c_vec = np.zeros((self.n,1))
        c_vec[0 : self.n_lf] = self.kriging_diff.scale_factor * self.kriging_lf.sigma * psic_new
        c_vec[self.n_lf : self.n] = self.kriging_diff.scale_factor**2 * self.kriging_lf.sigma * psie_new
        c_vec[self.n_lf : self.n] += self.kriging_diff.sigma*psid_new

        F = np.ones((self.n,self.d))
        y_pred = self.mean + np.transpose(c_vec) @ self.inv_C @ (self.yck-F*self.mean)
        return y_pred

    # Calculate the distance relation between multi-fidelity datasets
    def calculate_psi_matrix(self):
        psi_ec = np.zeros((self.n_hf,self.n_lf))
        for i in range(self.n_hf):
            for j in range(self.n_lf):
                diffX = self.x[i,:]-self.x_lf[j,:]
                psi_ec[i,j] = np.exp(-self.kriging_lf.theta*np.dot(diffX,diffX))                
        psi_ce = np.transpose(psi_ec)
        return psi_ec, psi_ce
    
    def calculate_covariance(self,psic_ce, psic_ec, psic_e):
        n = self.n_hf+self.n_lf
        C_mat = np.zeros((n,n))
        C_mat[0:self.n_lf,0:self.n_lf]=self.kriging_lf.sigma*self.kriging_lf.psi
        C_mat[0:self.n_lf,self.n_lf:n] = self.kriging_diff.scale_factor*self.kriging_lf.sigma*psic_ce
        C_mat[self.n_lf:n,0:self.n_lf] = self.kriging_diff.scale_factor*self.kriging_lf.sigma*psic_ec
        val1 = self.kriging_diff.scale_factor**2 * self.kriging_lf.sigma * psic_e
        val2 = self.kriging_diff.sigma*self.kriging_diff.psi
        C_mat[self.n_lf:n,self.n_lf:n] = self.kriging_diff.scale_factor**2 * self.kriging_lf.sigma * psic_e
        C_mat[self.n_lf:n,self.n_lf:n] += self.kriging_diff.sigma*self.kriging_diff.psi
        mat = np.matrix(C_mat)
        with open('CMATOUT.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%.2f')
        inv_C_mat = np.linalg.inv(C_mat)
        return C_mat,inv_C_mat
    
    