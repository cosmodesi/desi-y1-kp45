"""
Cosmo Blind Module
------------------
This module provides functionality to get blind cosmological parameters.
"""

import numpy as np
from cosmoprimo.fiducial import DESI

# Constants
TP2Z = {'LRG': 0.8, 'ELG': 1.1, 'QSO': 1.6, 'BGS': 0.25}
TP2BIAS = {'LRG': 2., 'ELG': 1.3, 'QSO': 2.3, 'BGS': 1.8}

def compute_f_blind(tracer_type, w0_blind, wa_blind, fiducial_f):
    """
    Computes the f_blind value.
    
    Parameters:
    - tracer_type (str): Type of tracer ('LRG', 'ELG', 'QSO', or 'BGS').
    - w0_blind (float): w0_blind value
    - wa_blind (float): wa_blind value
    - fiducial_f (float): fiducial_f value
    
    Returns:
    float: The computed f_blind value.
    """
    if tracer_type not in TP2Z:
        raise ValueError(f"Invalid tracer_type '{tracer_type}'. Expected one of {list(TP2Z.keys())}.")
    
    ztp = TP2Z[tracer_type[:3]]
    bias = TP2BIAS[tracer_type[:3]]
    
    cosmo_fid = DESI()
    cosmo_shift = cosmo_fid.clone(w0_fld=w0_blind, wa_fld=wa_blind)
    DM_fid = cosmo_fid.comoving_angular_distance(ztp)
    DH_fid = 1. / cosmo_fid.hubble_function(ztp)
    DM_shift = cosmo_shift.comoving_angular_distance(ztp)
    DH_shift = 1. / cosmo_shift.hubble_function(ztp)
    vol_fac = (DM_shift**2 * DH_shift) / (DM_fid**2 * DH_fid)
    
    a = 0.2 / bias**2
    b = 2 / (3 * bias)
    c = 1 - (1 + 0.2 * (fiducial_f / bias)**2. + 2/3 * fiducial_f / bias) / vol_fac
    f_blind = (-b + np.sqrt(b**2. - 4.*a*c)) / (2*a)
    
    dfper = (f_blind - fiducial_f) / fiducial_f
    maxfper = 0.1
    if abs(dfper) > maxfper:
        dfper = maxfper*dfper/abs(dfper)
        f_blind = (1+dfper) * fiducial_f
    
    return f_blind

def get_blind_cosmo(tracer_type, zmin, zmax, w0_blind=-0.90, wa_blind=0.26, fiducial_f=0.8):
    """
    Gets the blind cosmological parameters.
    
    Parameters:
    - tracer_type (str): Type of tracer ('LRG', 'ELG', 'QSO', or 'BGS').
    - zmin (float): Minimum z value
    - zmax (float): Maximum z value
    - w0_blind (float, optional): w0_blind value. Default is -0.90.
    - wa_blind (float, optional): wa_blind value. Default is 0.26.
    - fiducial_f (float, optional): fiducial_f value. Default is 0.8.
    
    Returns:
    dict: Dictionary with the blind cosmological parameters.
    """
    f_blind = compute_f_blind(tracer_type, w0_blind, wa_blind, fiducial_f)
    
    cosmo = fiducial = DESI()
    ztp = (zmax + zmin) / 2.
    print("=====================================")
    print(f"ztp: {ztp}")
    cosmo = fiducial.clone(w0_fld=w0_blind, wa_fld=wa_blind)
    qpar = cosmo.efunc(ztp) / fiducial.efunc(ztp)
    qper = fiducial.comoving_angular_distance(ztp) / cosmo.comoving_angular_distance(ztp)
    qiso = (qpar * qper**2)**(1/3)
    qap = qpar / qper
    df = f_blind / fiducial_f

    return dict(qpar=qpar, qper=qper, df=df, dm=0., qiso=qiso, qap=qap)

if __name__ == "__main__":
    # Sample execution code or testing can go here
    pass
