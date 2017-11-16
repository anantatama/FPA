import numpy as np
import random
import scipy
from scipy import special

def u(beta):
    var_u = np.random.normal(0, omega_u(beta)**2)
    return var_u

def v():
    return 1

def omega_u(beta):
    om_u = ((scipy.special.gamma(1+beta) * np.sin(np.pi*beta/2)) / (scipy.special.gamma((1+beta)/2) * beta * 2**(beta-1)/2))**1/beta
    return om_u

def stepsize(beta):
    s = u(beta) / (abs(v())**1/beta)
    return s
