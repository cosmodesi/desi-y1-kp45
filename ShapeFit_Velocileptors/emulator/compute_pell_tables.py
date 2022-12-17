import numpy as np

from classy import Class
from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from shapefit import shapefit_factor

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )


def compute_pell_tables(pars,pkclass,z ,h ,fid_dists= None, ap_off=False ):
    
    # OmegaM, h, sigma8 = pars
    Hzfid, chizfid = fid_dists
    f_sig8,apar,aperp,m = pars


    
    if ap_off:
        apar, aperp = 1.0, 1.0

    sig8_z = pkclass.sigma(8,z,h_units=True)
    f = f_sig8 / sig8_z
    
    # Calculate and renormalize power spectrum
    ki = np.logspace(-3.0,1.0,400)
    pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] ) * np.exp( shapefit_factor(ki,m) )
    # pi = (sig8_z/pkclass.sigma8())**2 * pi
    
    # Now do the RSD
    modPT = LPT_RSD(ki, pi, kIR=0.2,\
                cutoff=10, extrap_min = -5, extrap_max = 3, N = 4000, threads=1, jn=5)
    modPT.make_pltable(f, kv=kvec, apar=apar, aperp=aperp, ngauss=3)
    
    return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable

