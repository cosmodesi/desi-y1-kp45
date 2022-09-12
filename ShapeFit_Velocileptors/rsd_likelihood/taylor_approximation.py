import numpy as np
from findiff import FinDiff
from scipy.special import factorial

def taylor_approximate(pars, x0s, derivs, order=None):
    '''
    Computes the nth order Taylor series approximation to a function given 
    - derivs: a list of n derivatives
    - x0s: the point about which we are expanding
    - pars: the coordinates at which we want to expand the function
    If order is specified it will go to that order instead of using the full list of derivatives.
    '''
    Nparams = len(x0s)
    output_shape = derivs[0].shape
    
    if order is None:
        order = len(derivs) - 1
        print("Taking the order %d Taylor series."%(order))

    Fapprox = np.zeros(output_shape)
    diffs =  [ (pars[ii] - x0s[ii]) for ii in range(Nparams) ]

    for oo in range(order+1):
        if oo == 0:
            Fapprox += derivs[oo]
        else:
            param_inds = np.meshgrid( * (np.arange(Nparams),)*oo, indexing='ij')
            PInds = [ind.flatten() for ind in param_inds] 
        
            term = 0
        
            for iis in zip(*PInds):
                term += derivs[oo][iis] * np.prod([ diffs[ii] for ii in iis ])
            
            Fapprox += 1/factorial(oo) * term
    
    return Fapprox


def compute_derivatives(Fs, dxs, center_ii, order):
    '''
    Computes all the partial derivatives up to order 'order' given a function on a grid Fs
    The grid separation is given by dxs, and the derivatives are computed at the grid point 'center_ii.'
    
    Assumes that Fs is gridded in the standard matrix way (i.e. indexing = 'ij' in numpy) as
    opposed to Cartesian indexing that one uses for plotting.
    '''
    
    # assume the structure of the input is
    # [Npoints,]*Nparams + output_shape, where Nparams is also the length of dxs
    
    Nparams = len(dxs)
    output_shape = Fs.shape[len(dxs):]
    
    derivs = []

    for oo in range(order+1):
        if oo == 0:
            derivs += [Fs[center_ii]]
        else:
            dnFs = np.zeros( (Nparams,)*oo + output_shape)
        
            # Want to get a list of all the possible d/dx_i dx_j dx_k ...
            param_inds = np.meshgrid( * (np.arange(Nparams),)*oo, indexing='ij')
            PInds = [ind.flatten() for ind in param_inds] 
        
            for iis in zip(*PInds):
                # build a string of (xk, dxk, 1) for taking the d/dxk derivative in sequence
                deriv_tuple = []
                for ii in iis:
                    deriv_tuple += [(ii, dxs[ii],1),]
            
                dndx = FinDiff(*deriv_tuple)
            
                dnFs[iis] += dndx(Fs)[center_ii]
        
            derivs += [dnFs]
            
    return derivs

