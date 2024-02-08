import numpy as np

def expensive_test_function(fxn_ind,Xin):
    if fxn_ind == 1:
        y_samp = (6*Xin-2)**2*np.sin(12*Xin-4)
    elif fxn_ind == 2:
        y_samp = np.sin(2*np.pi*(Xin-0.1))+Xin**2
    elif fxn_ind == 3:
        y_samp = Xin*np.sin(Xin)/10
    elif fxn_ind == 4:
        y_samp = np.cos(3.5*np.pi*Xin)*np.exp(-1.4*Xin)
    elif fxn_ind == 5:
        y_samp = 4-2.1*Xin[:,1]^4+(1.0/3.0)*Xin[:,1]^6+Xin[:,1]*Xin[:,2]-4*Xin[:,2]**2+4.*Xin[:,2]**4
    else:
        print('invalid function index')
    return y_samp

# Calculate sample points from the specified "Cheap" test function
# This function example is from Forrester 2008
def cheap_test_function(fxn_ind,Xin):
    if fxn_ind == 1:
        y_samp = 0.5*expensive_test_function(fxn_ind,Xin)+10*(Xin-0.5)-5
    elif fxn_ind == 2:
        y_samp = np.sin(2*np.pi*(Xin-0.1))
    elif fxn_ind == 3:
        y_samp = Xin*np.sin(Xin)/10 + Xin/10
    elif fxn_ind == 4:
        y_samp = np.cos(3.5*np.pi*Xin)*np.exp(-1.4*Xin) + 0.75*Xin**2
    elif fxn_ind == 5:
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
        xmin = np.zeros(2,1)
        dim = 2
        xmin[1] = -2
        xmin[2] = -2
        xmax[1] = 2
        xmax[2] = 2
    
    return dim,xmin,xmax