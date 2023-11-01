import yaml
import numpy as np
import thecov.covariance
import thecov.geometry
import thecov.base
import thecov.utils
from cosmoprimo.fiducial import DESI
from mockfactory import Catalog, utils
from pypower import CatalogFFTPower, PowerSpectrumStatistics
import os
import logging
import time

# Setting up the logger
logger = logging.getLogger('GenerateCovariance')
logging.basicConfig(filename='covariance.log', level=logging.DEBUG)

# ----------- DATA PROCESSING FUNCTIONS --------------

def read_npy(filename):
    """Read data from a NPY file and return a Catalog object."""
    data = np.load(filename)
    labels = ['RA', 'DEC', 'Z', 'NZ']
    return Catalog({l: data[:,i] for i,l in enumerate(labels)})

def get_maskbit(main, nz, Y5, sv3):
    """Calculate and return the maskbit for the catalog."""
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

def select_maskbit(catalog, main=1, nz=0, Y5=1, sv3=0):
    """Filter the catalog based on the maskbit."""
    maskbit = get_maskbit(main, nz, Y5, sv3)
    return catalog[(catalog['STATUS'] & maskbit == maskbit)]

def select_region(catalog, region):
    """Filter the catalog based on a specified sky region."""
    if 'NGC' in region:
        mask = (catalog['RA'] > 88) & (catalog['RA'] < 303)
    elif 'SGC' in region:
        mask = (catalog['RA'] < 88) | (catalog['RA'] > 303)
    elif 'NGCSGCcomb' in region:
        return catalog
    return catalog[mask]

def select_redshift(catalog, zmin, zmax):
    """Filter the catalog based on redshift range."""
    return catalog[(catalog['Z'] > zmin) & (catalog['Z'] < zmax)]

def normalize_and_concatenate(list_data, list_randoms):
    """Normalize and concatenate data and random catalogs."""
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
    alpha = len(data)/len(randoms)
    logger.info(f'Estimated alpha is {alpha}')
    return alpha

start_time = time.time()

# Load configuration from YAML file
try:
    with open("py/config_covariance.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        logger.info('Successfully loaded configuration from config_covariance.yaml')
except Exception as e:
    logger.error(f'Error loading configuration: {str(e)}')
    raise

cosmo = DESI()

pk_dir = config['pk_dir']
catalog_dir = config['catalog_dir']
randoms_dir = config['randoms_dir']
tracer = config['tracer']
zmin = config['zmin']
zmax = config['zmax']
window_kernel = config['window_kernel'].format(tracer=tracer, zmin=zmin, zmax=zmax)
output_covariance = config['output_covariance'].format(tracer=tracer, zmin=zmin, zmax=zmax)

pk = PowerSpectrumStatistics.load(os.path.join(pk_dir, f'pkpoles_{tracer}_NGC_{zmin}_{zmax}_default_FKP_lin.npy'))
pk = pk[:400:5]
shotnoise = pk.shotnoise
P = pk(ell=[0, 2, 4], remove_shotnoise=True, complex=False)
P0, P2, P4 = P[0, :], P[1, :], P[2, :]

catalog = Catalog.read(os.path.join(catalog_dir, f'{tracer}_NGC_clustering.dat.fits'))
catalog = select_redshift(catalog, zmin, zmax)

randoms = Catalog.read(os.path.join(randoms_dir, f'{tracer}_NGC_0_clustering.ran.fits'))
randoms = select_redshift(randoms, zmin, zmax)

randoms['POSITION'] = utils.sky_to_cartesian(cosmo.comoving_radial_distance(randoms['Z']), randoms['RA'], randoms['DEC'], degree=True)
alpha = get_alpha(data=catalog, randoms=randoms)

# --------- COMPUTING COVARIANCE --------------

geometry = thecov.geometry.SurveyGeometry(randoms, nmesh=32, boxpad=1.2, alpha=alpha, kmodes_sampled=2000)

covariance = thecov.covariance.GaussianCovariance(geometry)
covariance.set_kbins(0, 0.4, 0.005)
covariance.set_pk(P0, 0, has_shotnoise=False)
covariance.set_pk(P2, 2)
covariance.set_pk(P4, 4)
covariance.set_shotnoise(shotnoise)

if os.path.exists(window_kernel):
    logger.info(f'Loading window kernels from {window_kernel}...')
    geometry.load_window_kernels(window_kernel)

covariance.compute_covariance()

if not os.path.exists(window_kernel):
    geometry.save_window_kernels(window_kernel)

covariance.savetxt(output_covariance)

end_time = time.time()
elapsed_time = (end_time - start_time) / 60.

if __name__ == "__main__":
    print(f"Script executed in {elapsed_time:.2f} minutes.")
    print(f"Results saved to {output_covariance}")
