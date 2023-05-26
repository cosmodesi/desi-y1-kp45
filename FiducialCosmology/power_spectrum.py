import os, sys
import tempfile
import fitsio
import numpy as np
from cosmoprimo import fiducial
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging, mpi
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction
from astropy.table import Table
from time import time
setup_logging()
import argparse
from read_data import *

out_parent_dir = '/global/cfs/projectdirs/desi/users/alexpzfz/kp4/Testing_fiducial_cosmo/'
input_parent_dir = '/pscratch/sd/a/alexpzfz/KP4/Testing_fiducial_cosmo/' # for postrec catalogs


# power spectrum settings
step = 0.001
cellsize = 2000./512 # for consistency


parser = argparse.ArgumentParser(description='This is my help')
parser.add_argument('mocktype', choices=['cubicbox', 'cutsky'],
                    help='The kind of mock.')
parser.add_argument('tracer', choices=['lrg', 'elg', 'qso'],
                    help='Tracer.')
parser.add_argument('whichmocks', choices=['firstgen', 'sv3'],
                    help='FirstGen mocks or mocks with new HOD based on sv3.')
parser.add_argument('ph', choices=range(25), type=int, help='Phase')
parser.add_argument('zbin', choices=range(3), type=int, nargs='?')
parser.add_argument('-r', '--rectype', choices=['reciso', 'recsym'],
                    help='Type of reconstruction.', default=None)
parser.add_argument('-ct', '--cosmo_true', choices=['000', '003', '004'],
                    default='000')
parser.add_argument('-cg', '--cosmo_grid', choices=[f'00{i}' for i in range(5)],
                    default='000')
args = parser.parse_args()

mocktype = args.mocktype
tracer = args.tracer.upper()
whichmocks = args.whichmocks
ph = f'{args.ph:03d}'
rectype = args.rectype
ncosmo_true = args.cosmo_true
ncosmo_grid = args.cosmo_grid
if args.zbin:
    nzbin=args.zbin

print0('\nPower spectrum computation:')
reclabel=rectype.upper() if rectype else 'PRE-REC'
print0(f'{mocktype.upper()} {tracer} {whichmocks.upper()} {ph} {reclabel}') 
print0(f'True cosmology={ncosmo_true}, Grid cosmology={ncosmo_grid}')

# set cosmologies 
cosmo_true = fiducial.AbacusSummit(name=ncosmo_true, engine='class')
cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')

# set style for reading data
if whichmocks=='firstgen':
    style ='firstgen'
    whereto = f'AbacusSummit_base_c{ncosmo_true}_FirstGen/'
elif whichmocks=='sv3':
    whereto = f'AbacusSummit_base_c{ncosmo_true}_SV3/'
    if tracer=='LRG':
        style = 'sandy'
    elif tracer=='ELG':
        style = 'antoine'


if mocktype=='cubicbox':
    # set output directory
    output_dir = out_parent_dir + f'CubicBox/Pk/{tracer}/' + whereto
    if not os.path.exists(output_dir):
        if mpicomm.rank==0: # avoid error if multiple tasks try to create the directory
            os.makedirs(output_dir)
        
    boxsize = 2000.0
    boxcenter = 1000.0
    los = 'z'
    zbox = settings_cubic[tracer]['zbox']
    smoothing = settings_cubic[tracer]['smoothing']
    bias = settings_cubic[tracer]['bias']
    snap = settings_cubic[tracer]['snap']
    
    # Calculate Alcock Paczynski dilation parameters
    h_grid, h_true = cosmo_grid.efunc(zbox).item(), cosmo_true.efunc(zbox).item()
    da_grid, da_true = cosmo_grid.angular_diameter_distance(zbox).item(), cosmo_true.angular_diameter_distance(zbox).item()
    q_par = h_grid / h_true
    q_perp = da_true / da_grid
    print0('\nRescaling the box with:')
    print0('q_par =', q_par)
    print0('q_perp =', q_perp, '\n')
    boxsize_ap = np.array([boxsize]*3, dtype='f4')
    boxsize_ap /= np.array([q_perp, q_perp, q_par], dtype='f4')
    boxcenter_ap = np.array([boxcenter]*3, dtype='f4')
    boxcenter_ap /= np.array([q_perp, q_perp, q_par], dtype='f4')
    
    
    if not rectype:
        data = read_data(mocktype, tracer, ncosmo_true, ph, style=style)
        data_positions = data.T
        data_positions /= np.array([q_perp, q_perp, q_par], dtype='f4')
        
        shifted_positions = None
        
        ofile = output_dir + f'Pk_{tracer}_snap{snap}_Grid{ncosmo_grid}_ph{ph}.npy'
        
    else:
        rec_settings = {'nmesh': 512, 'recalg': 'multigrid'}
        data, file_name = read_data(mocktype, tracer, ncosmo_true, ph, rectype=rectype,
                                    style=style, path=input_parent_dir, ncosmo_grid=ncosmo_grid,
                                    rec_settings=rec_settings)
        data_positions = data.T
        shifted_list = read_randoms(mocktype, tracer, rectype=rectype, path=input_parent_dir,
                                    style=style, ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid,
                                    ph=ph, rec_settings=rec_settings)
            
        shifted_list = [randoms.T for randoms in shifted_list]
        shifted_positions = np.concatenate(shifted_list)
        
        ofile = output_dir + ('Pk_' + '_'.join(file_name.replace('displaced_', '').split('_')[:-2]) +
                              f'_{rectype}_Grid{ncosmo_grid}_ph{ph}.npy')
        
        
    # Calculate power spectrum multipoles
    poles = CatalogFFTPower(data_positions1=data_positions, shifted_positions1=shifted_positions,
                            boxsize=boxsize_ap, boxcenter=boxcenter_ap, cellsize=cellsize, resampler='tsc',
                            interlacing=2, ells=(0, 2, 4), los=los, edges={'step':step}, position_type='pos', 
                            mpicomm=mpicomm, mpiroot=mpiroot, dtype='f4').poles
    
elif mocktype=='cutsky':
    raise NotImplementedError


poles.save(ofile)
if ph == '00':
    poles.save_txt(ofile.replace('npy', 'txt'), complex=False)
