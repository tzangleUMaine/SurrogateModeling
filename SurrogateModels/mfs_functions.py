import numpy as np
import model

class expensive_fxn(model.Model):
    def __init__(self, fxn_ind):
        self.fxn_ind = fxn_ind

    def evaluate(self, Xin):
        if self.fxn_ind == 1:
            y_samp = (6*Xin-2)**2*np.sin(12*Xin-4)
        elif self.fxn_ind == 2:
            y_samp = np.sin(2*np.pi*(Xin-0.1))+Xin**2
        elif self.fxn_ind == 3:
            y_samp = Xin*np.sin(Xin)/10
        elif self.fxn_ind == 4:
            y_samp = np.cos(3.5*np.pi*Xin)*np.exp(-1.4*Xin)
        elif self.fxn_ind == 5:
            y_samp = 4-2.1*Xin[:,1]^4+(1.0/3.0)*Xin[:,1]^6+Xin[:,1]*Xin[:,2]-4*Xin[:,2]**2+4.*Xin[:,2]**4
        else:
            print('invalid function index')
        return y_samp

# Calculate sample points from the specified "Cheap" test function
# This function example is from Forrester 2008
class cheap_fxn(model.Model):    
    def __init__(self, fxn_ind):
        self.fxn_ind = fxn_ind

        # Note: I dont like this
        self.expensive_fxn = expensive_fxn(fxn_ind)

    def evaluate(self,Xin):
        if self.fxn_ind == 1:
            y_samp = 0.5*self.expensive_fxn.evaluate(Xin)+10*(Xin-0.5)-5
        elif self.fxn_ind == 2:
            y_samp = np.sin(2*np.pi*(Xin-0.1))
        elif self.fxn_ind == 3:
            y_samp = Xin*np.sin(Xin)/10 + Xin/10
        elif self.fxn_ind == 4:
            y_samp = np.cos(3.5*np.pi*Xin)*np.exp(-1.4*Xin) + 0.75*Xin**2
        elif self.fxn_ind == 5:
            y_samp = 4-0.5*Xin[:,1]^4+(0.04)*Xin[:,1]**6+0.5*Xin[:,1]*Xin[:,2]-1.96*Xin[:,2]**2+0.96*Xin[:,2]**4
        else:
            print('invalid function index')
        return y_samp

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
        xmin = np.zeros((dim,1))
        xmax = np.zeros((dim,1))
        xmin[0] = -2
        xmin[1] = -2
        xmax[0] = 2
        xmax[1] = 2
    
    return dim,xmin,xmax