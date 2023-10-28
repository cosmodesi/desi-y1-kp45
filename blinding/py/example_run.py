import numpy as np

import thecov.covariance
import thecov.geometry
import thecov.base
import thecov.utils

from cosmoprimo.fiducial import DESI
from mockfactory import Catalog, utils
from pypower import CatalogFFTPower

import os
import logging

logger = logging.getLogger('GenerateCovariance')
logging.basicConfig(filename='covariance.log', level=logging.DEBUG)

cosmo = DESI()

# --------- LOADING REFERENCE P(K) ------------

basedir = '/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Post/forero/fiducial_settings/dk0.005/z0.800'
pypower = CatalogFFTPower.load(f'{basedir}/saiojsaoijs.npy')
pks = pk.poles.get_power(remove_shotnoise=True, complex=False)[:,:80]
shotnoise = pypowers[0].poles.shotnoise

P0, P2, P4 = np.mean(pks, axis=0)

# mock_cov = thecov.base.MultipoleCovariance.from_array(np.cov([pk.flatten() for pk in pks], rowvar=False))

# --------- LOADING CATALOGS AND RANDOMS ------------

# Catalog is needed only to calculate alpha. Randoms are used for window calculations

def read_npy(filename):
    data = np.load(filename)
    labels = ['RA', 'DEC', 'Z', 'NZ']
    return Catalog({l: data[:,i] for i,l in enumerate(labels)})

def get_maskbit(main, nz, Y5, sv3):
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

def select_maskbit(catalog, main=1, nz=0, Y5=1, sv3=0):
    maskbit = get_maskbit(main, nz, Y5, sv3)
    catalog = catalog[(catalog['STATUS'] & maskbit == maskbit)]
    return catalog

def select_region(catalog, region):
    '''This function selects a region of the sky and returns the catalog of galaxies in that region.'''
    
    if 'NGC' in region:
        mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
    elif 'SGC' in region:
        mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
    elif 'NGCSGCcomb':
        return catalog

    return catalog[mask]

def select_redshift(catalog, zmin, zmax):
    mask = (catalog['Z'] > zmin) & (catalog['Z'] < zmax)
    return catalog[mask]

def normalize_and_concatenate(list_data, list_randoms):
    '''This function concatenates data and random catalogs, renormalizing random weights before concatenation.'''
    wsums_data = [data['WEIGHT'].sum().compute() for data in list_data]
    wsums_randoms = [randoms['WEIGHT'].sum().compute() for randoms in list_randoms]
    alpha = sum(wsums_data) / sum(wsums_randoms)
    alphas = [wsum_data / wsum_randoms / alpha for wsum_data, wsum_randoms in zip(wsums_data, wsums_randoms)]
    
    logger.info(f'Renormalizing randoms weights by {alphas} before concatenation.')
    for randoms, a in zip(list_randoms, alphas):
        randoms['WEIGHT'] *= a

    logger.info(f'Estimated alpha is {alpha}')
    return Catalog.concatenate(list_randoms), alpha

def get_alpha(data, randoms):
    # alpha =  (data['WEIGHT'].sum() / randoms['WEIGHT'].sum()).compute()
    alpha = len(data)/len(randoms)
    logger.info(f'Estimated alpha is {alpha}')
    return alpha

zmin, zmax = 0.8, 1.1

catalog = Catalog.read(catalog_filename)
catalog = select_redshift(catalog, zmin, zmax)

randoms = Catalog.read(randoms_filename)
# randoms = randoms[np.random.rand(randoms.size) < 0.05]
randoms = select_redshift(randoms, zmin, zmax)

# WEIGHT and WEIGHT_FKP columns should be present
# randoms['WEIGHT'] = randoms['NZ']**0
# randoms['WEIGHT_FKP'] = 1./(1. + 1e4*randoms['NZ'])

randoms['POSITION'] = utils.sky_to_cartesian(cosmo.comoving_radial_distance(randoms['Z']), randoms['RA'], randoms['DEC'], degree=True)

alpha = get_alpha(data=catalog, randoms=randoms)
# alpha_pypower = pypowers[0].poles.attrs['sum_data_weights1']/pypowers[0].poles.attrs['sum_randoms_weights1']

# --------- COMPUTING COVARIANCE ------------

geometry = thecov.geometry.SurveyGeometry(randoms, nmesh=32, boxpad=1.2, alpha=alpha, kmodes_sampled=2000)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(0, 0.4, 0.005)

covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)

covariance.set_shotnoise(shotnoise)

# Saving the window kernel calculations help to speed up the covariance re-calculation when
# only changing the input power spectrum
window_kernel_filename = 'window_kernel.npz'

if os.path.exists(window_kernel_filename):
    logger.info(f'Loading window kernels from {window_kernel_filename}...')
    geometry.load_window_kernels(window_kernel_filename)

covariance.compute_covariance()

if not os.path.exists(window_kernel_filename):
    geometry.save_window_kernels(window_kernel_filename)

# covariance.symmetrize()

covariance.savetxt(f'covariance.txt')
