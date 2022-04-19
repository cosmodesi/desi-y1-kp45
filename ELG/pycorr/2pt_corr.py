################################################################### STAGE 1: LOAD PACKAGES ###################################################################
import time
import os

# standard scientific libraries
import numpy as np

# loading module for 2pt calculationgs
from pycorr import TwoPointCorrelationFunction, utils, setup_logging

# loading astropy to inspect fits files
from astropy.io import fits

# load cosmoprimo for redshift-distance relation
import cosmoprimo
from cosmoprimo.fiducial import DESI

# to read ini file
import sys
import argparse
import configparser
import ast

def mask(main=0, nz=0, Y5=0, sv3=0):
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

print('....................................................... STAGE 1 COMPLETED: Modules loaded.\n')
###################################################################### STAGE 2: SETTINGS ######################################################################
# To activate logging
setup_logging()

# set parser
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = sys.argv[1]
inifile = dir_path + f'/{filename}'
CONFIG = configparser.RawConfigParser()
CONFIG.read(inifile)

# set catalogues path
path_cat = CONFIG.get('path', 'path_gal_cat')
path_ran = CONFIG.get('path', 'path_ran_cat')

# set labels for right ascension, declination and redshift, used in fits files
ra_label = CONFIG.get('labels', 'ra_label')
dec_label = CONFIG.get('labels', 'dec_label')
z_label = CONFIG.get('labels', 'z_label')

# set fiducial cosmology
h_fid = float(CONFIG.get('fiducial_cosmology', 'h_fid'))
omm_fid = float(CONFIG.get('fiducial_cosmology', 'omm_fid'))
sigma8_fid = float(CONFIG.get('fiducial_cosmology', 'sigma8_fid'))

# set redshift bins
z_low = float(CONFIG.get('analysis_settings', 'z_low'))
z_high = float(CONFIG.get('analysis_settings', 'z_high'))
P_0 = float(CONFIG.get('analysis_settings', 'P_0'))

# set other analysis settings
boltzmann_code = CONFIG.get('analysis_settings', 'boltzmann_code')
two_pt_mode = CONFIG.get('analysis_settings', 'two_pt_mode')
two_pt_engine = CONFIG.get('analysis_settings', 'two_pt_engine')
num_of_threads = int(CONFIG.get('analysis_settings', 'num_of_threads'))
kmin = float(CONFIG.get('analysis_settings', 'kmin'))
kmax = float(CONFIG.get('analysis_settings', 'kmax'))
nk = int(CONFIG.get('analysis_settings', 'nk'))
mumax = float(CONFIG.get('analysis_settings', 'mumax'))
nmu = int(CONFIG.get('analysis_settings', 'nmu'))

# set output path
output_path = CONFIG.get('path', 'output_path')

# set mpi (optional)
if(CONFIG.get('analysis_settings', 'use_mpi') == 'True'):
    from mpi4py import MPI
    #from pycorr import mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size_comms = comm.size
    print('Succesfully launched the process of rank №', rank,' of total size of comms being  ', size_comms)

# set fiducial cosmology with cosmoprimo for redshift-distance relationship
cosmo = DESI()
cosmo.set_engine(boltzmann_code)

print('....................................................... STAGE 2 COMPLETED: Settings loaded. Cosmology set. \n')
#################################################################### STAGE 3: GET POSITIONS FOR GALAXY CATALOGUES ####################################################################

start = time.time()

# Read fits files for galaxies
hdul = fits.open(path_cat, mode='readonly')
print("GALAXY CATALOGUE:")
print("---------------------------------------------------------------------------------------------------")
print("FITS info:")
print("---------------------------------------------------------------------------------------------------")
hdul.info()

# Read galaxy catalogue data
bin_cat = (hdul[""].data[z_label]>z_low)&(hdul[""].data[z_label]<z_high)
RA_cat = hdul[""].data[ra_label]
DEC_cat = hdul[""].data[dec_label]
z_cat = hdul[""].data[z_label]
STATUS_cat = hdul[""].data['STATUS']
nz_cat = hdul[""].data['NZ']

# close fits file
hdul.close()

# APPLY MASKS FOR DESI-Y5 AND REDSHIFT BIN

### Array of indices
idx = np.arange(len(STATUS_cat))

### Choose the Y5 footprint and downsampled to input n(z)
mask_Y5 = mask(main=0, nz=1, Y5=1, sv3=0)
idx_Y5 = idx[(STATUS_cat & (mask_Y5))==mask_Y5]

### Apply masking for DESI-Y5
bin_cat = bin_cat[idx_Y5]
RA_cat = RA_cat[idx_Y5]
DEC_cat = DEC_cat[idx_Y5]
z_cat = z_cat[idx_Y5]
nz_cat = nz_cat[idx_Y5]

### Apply masking for redshift-bin
RA_cat = RA_cat[bin_cat]
DEC_cat = DEC_cat[bin_cat]
z_cat = z_cat[bin_cat]
nz_cat = nz_cat[bin_cat]

print("Number of galaxies:", len(RA_cat))

# convert redshift to distance
comoving_distance_cat = cosmo.comoving_radial_distance(z_cat)

# convert sky to cartesian coordinates
pos_cat = utils.sky_to_cartesian(np.array([RA_cat, DEC_cat, comoving_distance_cat]))
pos_cat = np.asarray(pos_cat)
pos_cat = pos_cat.astype('float32')
#print(pos_cat)
#print(type(pos_cat))
#print(np.shape(pos_cat))
#print(pos_cat[0].dtype)

# get FKP weights
w_cat = 1. / (1. + P_0*nz_cat)

print('....................................................... STAGE 3 COMPLETED: Galaxies ready. Time elapsed:', time.time() - start, '\n')
############################################################## STAGE 4: GET POSITIONS FOR RANDONM CATALOGUES ##############################################################

start = time.time()

pos_ran = np.load(CONFIG.get('path', 'path_ran_positions'))
w_ran = np.load(CONFIG.get('path', 'path_ran_weights'))
    
print("Randoms loaded!")
print("Total number of randoms:", len(pos_ran[0]))
print('....................................................... STAGE 4 COMPLETED: Randoms ready. Time elapsed:', time.time() - start)
#################################################################### STAGE 5: RUN 2PT ####################################################################
start = time.time()
print(np.shape(pos_cat), np.shape(pos_ran))

# Define edges
edges = (np.linspace(kmin, kmax, nk), np.linspace(-mumax, mumax, nmu))

### Run Two point correlation function analysis
if(CONFIG.get('analysis_settings', 'use_mpi') == 'True'):
    if(CONFIG.get('analysis_settings', 'pre_computed_RR') == 'True'):
        # Use MPI and precomputed R1R2
        R1R2_path = CONFIG.get('path', 'precomputed_R1R2_path')
        precomputed = TwoPointCorrelationFunction.load(R1R2_path)
        result = TwoPointCorrelationFunction(two_pt_mode, edges, 
                                             data_positions1 = pos_cat, data_weights1=w_cat,
                                             randoms_positions1 = pos_ran, randoms_weights1 = w_ran, 
                                             position_type='xyz', engine = two_pt_engine, R1R2=precomputed.R1R2, dtype = 'float32', 
                                             nthreads = num_of_threads, mpicomm = comm, mpiroot = 0)
    else:
        # Use MPI and compute R1R2
        result = TwoPointCorrelationFunction(two_pt_mode, edges, 
                                             data_positions1 = pos_cat, data_weights1=w_cat,
                                             randoms_positions1 = pos_ran, randoms_weights1 = w_ran, 
                                             position_type='xyz', engine = two_pt_engine, dtype = 'float32', 
                                             nthreads = num_of_threads, mpicomm = comm, mpiroot = 0)
else:
    if(CONFIG.get('analysis_settings', 'pre_computed_RR') == 'True'):
        # Use precomputed R1R2 and don't use MPI
        R1R2_path = CONFIG.get('path', 'precomputed_R1R2_path')
        precomputed = TwoPointCorrelationFunction.load(R1R2_path)
        result = TwoPointCorrelationFunction(two_pt_mode, edges, 
                                             data_positions1 = pos_cat, data_weights1=w_cat,
                                             randoms_positions1 = pos_ran, randoms_weights1 = w_ran,
                                             position_type='xyz', engine = two_pt_engine, R1R2=precomputed.R1R2, dtype = 'float32',
                                             nthreads = num_of_threads)
    else:
        # Compute R1R2 and don't use MPI
        result = TwoPointCorrelationFunction(two_pt_mode, edges, 
                                             data_positions1 = pos_cat, data_weights1=w_cat,
                                             randoms_positions1 = pos_ran, randoms_weights1 = w_ran,
                                             position_type='xyz', engine = two_pt_engine, dtype = 'float32',
                                             nthreads = num_of_threads)

print('....................................................... STAGE 5 COMPLETED: Results ready! Time elapsed:', time.time() - start)
############################################################### STAGE 6: SAVE RESULTS ###############################################################
result.save(output_path)

settings_paths = {'cat': path_cat, 'ran': path_ran, 'output': output_path}
settings_labels = {'RA': ra_label, 'DEC': dec_label, 'Z': z_label}
settings_cosmology = {'h': h_fid, 'omegam': omm_fid, 'sigma_8': sigma8_fid}
settings_analysis = {'boltzmann': boltzmann_code, 'zlow': z_low, 'zhigh': z_high,
                     'mode': two_pt_mode, 'engine': two_pt_engine, 'threads': num_of_threads,
                     'mpi': CONFIG.get('analysis_settings', 'use_mpi'), 'P_0_FKP': P_0, 
                     'precomputed_RR': CONFIG.get('analysis_settings', 'pre_computed_RR'),
                     'kmin': kmin, 'kmax': kmax, 'nk': nk, 'mumax': mumax, 'nmu': nmu}
settings = {
    'paths': settings_paths,
    'labels': settings_labels,
    'cosmology': settings_cosmology,
    'analysis': settings_analysis
}
      
if(CONFIG.get('analysis_settings', 'pre_computed_RR') == 'True'):
    settings['paths']['precomputed_RR'] = R1R2_path
      
with open(output_path + '_settings.txt', 'w') as f:
    print(settings, file=f)
f.close()
      
print('....................................................... CODE COMPLETED. Results stored at:', output_path)
