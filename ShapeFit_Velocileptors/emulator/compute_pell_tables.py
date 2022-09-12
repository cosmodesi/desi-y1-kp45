import numpy as np

from classy import Class
from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from shapefit import shapefit_factor

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )

# Reference Cosmology:
# z = 0.61
Omega_M = 0.31
fb = 0.1571
h = 0.6766
ns = 0.9665
speed_of_light = 2.99792458e5


pkparams = {
    'output': 'mPk',
    'P_k_max_h/Mpc': 20.,
    'z_pk': '0.0,10',
    'A_s': np.exp(3.040)*1e-10,
    'n_s': 0.9665,
    'h': h,
    'N_ur': 3.046,
    'N_ncdm': 0,#1,
    #'m_ncdm': 0,
    'tau_reio': 0.0568,
    'omega_b': h**2 * fb * Omega_M,
    'omega_cdm': h**2 * (1-fb) * Omega_M}

import time
t1 = time.time()
pkclass = Class()
pkclass.set(pkparams)
pkclass.compute()


def compute_pell_tables(pars,pkclass, z=0.59,fid_dists= None, ap_off=False ):
    
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

