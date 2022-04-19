################################################################### STAGE 1: LOAD PACKAGES ###################################################################
import time
import os
from mpi4py import MPI

# standard scientific libraries
import numpy as np
import math

# loading module for 2pt calculationgs
from pypower import CatalogFFTPower, utils, setup_logging

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
print(inifile)
CONFIG = configparser.RawConfigParser()
CONFIG.read(inifile)

# set catalogues path
path_cat = CONFIG.get('path', 'path_gal_cat')
seed = CONFIG.get('path', 'seed')

# set labels 
x_label = CONFIG.get('labels', 'x_label')
y_label = CONFIG.get('labels', 'y_label')
z_label = CONFIG.get('labels', 'z_label')
vx_label = CONFIG.get('labels', 'vx_label')
vy_label = CONFIG.get('labels', 'vy_label')
vz_label = CONFIG.get('labels', 'vz_label')

# set other analysis settings
boltzmann_code = CONFIG.get('analysis_settings', 'boltzmann_code')
interlacing = int(CONFIG.get('analysis_settings', 'interlacing'))
boxsize = float(CONFIG.get('analysis_settings', 'boxsize'))
nmesh = int(CONFIG.get('analysis_settings', 'nmesh'))
ells = tuple([int(item) for item in CONFIG.get('analysis_settings', 'ells').split()])
los = CONFIG.get('analysis_settings', 'los')
resampler = CONFIG.get('analysis_settings', 'resampler')
P_0 = float(CONFIG.get('analysis_settings', 'P_0'))

# set output path
output_path = CONFIG.get('path', 'output_path')

# MPI
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

# use Leonel lines of code to concatenate sub-boxes
x = np.empty(0)
y = np.empty(0)
z = np.empty(0)
vx = np.empty(0)
vy = np.empty(0)
vz = np.empty(0)
for i in np.arange(64):
    print("Reading sub-box number {}...".format(i))
    hdul = fits.open(path_cat + '/EZmock_B2000G512Z1.1N24000470_b0.345d1.45r40c0.05_seed'+ seed +
                     '/seed' + seed + '.sub{}.fits.gz'.format(i))
    data = hdul[1].data
    x_temp = data[x_label]
    y_temp = data[y_label]
    z_temp = data[z_label]
    vx_temp = data[vx_label]
    vy_temp = data[vy_label]
    vz_temp = data[vz_label]
    hdul.close()
    x = np.hstack((x, x_temp))
    y = np.hstack((y, y_temp))
    z = np.hstack((z, z_temp))
    vx = np.hstack((vx, vx_temp))
    vy = np.hstack((vy, vy_temp))
    vz = np.hstack((vz, vz_temp))
    
pos_cat = np.array([x,y,z]).T
vel_cat = np.array([vx,vy,vz]).T
    
# apply RSD (redshift for ELG is z=1.1)
redshift = 1.1
rsd_factor = (1+redshift) / (100 * cosmo.efunc(redshift))
pos_cat += rsd_factor * vel_cat * [0, 0, 1] # only apply to the z-coordinate
# apply periodic condition
pos_cat = (pos_cat + boxsize) % boxsize

# remove nans (one single data-point was giving troubles, so remove it) 
print("shape of pos_cat array:", np.shape(pos_cat), "and vel_cat:", np.shape(vel_cat))
mask = np.ones(pos_cat.shape, bool)
mask[np.where(np.isnan(pos_cat))[0]] = False
pos_cat = pos_cat[mask[:,0]]
vel_cat = vel_cat[mask[:,0]]
mask = None
pos_cat = pos_cat.astype('float32') 
print("new shape of pos_cat array:",np.shape(pos_cat), "and vel_cat:", np.shape(vel_cat))

# get FKP weights
N_gal = pos_cat.shape[0]
print("Num of galaxies:", N_gal)
nbar_gal = N_gal / boxsize**3.0   # boxsize 2Gpc/h       
w_cat = 1.0/(1.0 + nbar_gal * P_0) * np.ones(N_gal) # need to be an array

print('....................................................... STAGE 3 COMPLETED: Galaxies ready. Time elapsed:', time.time() - start, '\n')
############################################################## STAGE 4: GET POSITIONS FOR RANDONM CATALOGUES ##############################################################

start = time.time()
#make randoms (using lines of code from Enrique Pailas)
nbar_rand = nbar_gal
N_rand = int(nbar_rand * boxsize**3.0)
np.random.seed(44)
alpha = 20         # the ratio of number of random points over particles
pos_ran = np.random.rand(N_rand*alpha, 3) * boxsize 
pos_ran = pos_ran.astype('float32') 

# get FKP weights
nbar_ran = (N_rand*alpha) / boxsize**3.0   # boxsize 2Gpc/h       
w_ran = 1.0/(1.0 + nbar_ran * P_0) * np.ones(N_rand*alpha) # need to be an array

print("Randoms generated!")
print("Total number of randoms:", len(pos_ran.T[0]))
print('....................................................... STAGE 4 COMPLETED: Randoms ready. Time elapsed:', time.time() - start)

#################################################################### STAGE 5: RUN 2PT ####################################################################
start = time.time()

# Define edges
kmin = 0.0
kmax = np.pi*nmesh/boxsize
delta_k = 0.005
nk = math.ceil((kmax-kmin)/delta_k)
kedges = np.linspace(kmin, kmax, nk)

### Run Two point correlation function analysis
result = CatalogFFTPower(data_positions1 = pos_cat, data_weights1=w_cat,
                         randoms_positions1 = pos_ran, randoms_weights1 = w_ran, 
                         edges = kedges, ells=ells, interlacing = interlacing,
                         boxsize = boxsize, #boxcenter=[500., 500., 500.],
                         nmesh = nmesh, resampler = resampler,  dtype = 'float32',
                         los = los, position_type = 'pos', mpicomm=comm, mpiroot = 0)

print('....................................................... STAGE 5 COMPLETED: Results ready! Time elapsed:', time.time() - start)
############################################################### STAGE 6: SAVE RESULTS ###############################################################
result.save(output_path)

settings_paths = {'cat': path_cat, 'output': output_path}
settings_cosmology = {'cosmo': cosmo.get_default_parameters()}
settings_analysis = {'boltzmann': boltzmann_code,
                     'interlacing': interlacing, 'boxsize': boxsize, 'nmesh': nmesh,
                     'los': los, 'resampler': resampler, 'ells': ells,
                     'P_0_FKP': P_0, 'kmin': kmin, 'kmax': kmax, 'nk': nk}
settings = {
    'paths': settings_paths,
    'cosmology': settings_cosmology,
    'analysis': settings_analysis
}
      
with open(output_path + '_settings.txt', 'w') as f:
    print(settings, file=f)
f.close()
      
print('....................................................... CODE COMPLETED. Results stored at:', output_path)
