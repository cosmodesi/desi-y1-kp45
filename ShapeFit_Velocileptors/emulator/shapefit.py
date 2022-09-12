import numpy as np

def shapefit_factor(kvec, m, a = 0.6, kp = 0.03, n = 0):
    
    return m/a * np.tanh(a * np.log(kvec/kp) ) + n * np.log(kvec/kp)