import os, sys
import tempfile
import fitsio
import numpy as np
from cosmoprimo import fiducial
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging, mpi, PowerSpectrumMultipoles
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction
from astropy.table import Table
from time import time
import argparse
from read_data import *
setup_logging()

out_parent_dir = '/pscratch/sd/a/alexpzfz/KP4/Testing_fiducial_cosmo/'

# settings for reconstruction
ReconstructionAlgorithm = MultiGridReconstruction
recalg = 'multigrid' # to name the files
nmesh = 512 # for cubic boxes
cellsize = 2000./nmesh # for cutsky


parser = argparse.ArgumentParser(description='This is my help')
parser.add_argument('mocktype', choices=['cubicbox', 'cutsky'],
                    help='The kind of mock.')
parser.add_argument('tracer', choices=['lrg', 'elg', 'qso'],
                    help='Tracer.')
parser.add_argument('whichmocks', choices=['firstgen', 'sv3'],
                    help='FirstGen mocks or mocks with new HOD based on sv3.')
parser.add_argument('ph', choices=range(25), type=int, help='Phase')
parser.add_argument('zbin', choices=range(3), type=int, nargs='?')
parser.add_argument('-ct', '--cosmo_true', choices=['000', '003', '004'],
                    default='000')
parser.add_argument('-cg', '--cosmo_grid', choices=[f'00{i}' for i in range(5)],
                    default='000')
args = parser.parse_args()

mocktype = args.mocktype
tracer = args.tracer.upper()
whichmocks = args.whichmocks
ph = f'{args.ph:03d}'
ncosmo_true = args.cosmo_true
ncosmo_grid = args.cosmo_grid
if args.zbin:
    nzbin=args.zbin

print('\nReconstrution computation:')
print(f'{mocktype.upper()} {tracer} {whichmocks.upper()} {ph}') 
print(f'True cosmology={ncosmo_true}, Grid cosmology={ncosmo_grid}\n')

# set style for reading data
if whichmocks=='firstgen':
    style ='firstgen'
    whereto = f'AbacusSummit_base_c{ncosmo_true}_FirstGen_ph{ph}/'
elif whichmocks=='sv3':
    whereto = f'AbacusSummit_base_c{ncosmo_true}_SV3_ph{ph}/'
    if tracer=='LRG':
        style = 'sandy'
    elif tracer=='ELG':
        style = 'antoine'


# seeds for randoms
seeds = list(range(100, 2100, 100))
   
# set cosmologies 
cosmo_true = fiducial.AbacusSummit(name=ncosmo_true, engine='class')
cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')

# read data
data = {}
data['Positions'] = read_data(mocktype, tracer, ncosmo_true,
                              ph, rectype=None, style=style, path=None)

if mocktype=='cubicbox':
    # set output directory
    output_dir = out_parent_dir + f'CubicBox/{tracer}/' + whereto
    if not os.path.exists(output_dir + '/randoms/'):
        os.makedirs(output_dir + '/randoms/')

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
    print('\nRescaling the box with:')
    print('q_par =', q_par)
    print('q_perp =', q_perp, '\n')
    boxsize_ap = np.array([boxsize]*3, dtype='f4')
    boxsize_ap /= np.array([q_perp, q_perp, q_par], dtype='f4')
    boxcenter_ap = np.array([boxcenter]*3, dtype='f4')
    boxcenter_ap /= np.array([q_perp, q_perp, q_par], dtype='f4')
    
    data['Positions'] = data['Positions'].T
    data['Positions'] /= np.array([q_perp, q_perp, q_par], dtype='f4')
    

    # read randoms
    if style=='firstgen':
        r_list = read_randoms(mocktype=mocktype, tracer=tracer)
    else:
        print(data['Positions'].shape)
        r_list = generate_randoms(seeds, size=data['Positions'].T.shape)
        print0(f'Succesfully generated {len(r_list)} boxes of {len(r_list[0][0])} randoms')
        
    randoms_list = [randoms.T / np.array([q_perp, q_perp, q_par], dtype='f4') for randoms in r_list] 
            
    # RECONSTRUCTION
    # calculate growth rate with grid cosmology
    f = cosmo_grid.growth_rate(zbox)
    print('Begining reconstruction')
    recon = ReconstructionAlgorithm(f=f, bias=bias, los=los, nmesh=nmesh, boxsize=boxsize_ap, dtype='f4',
                                    boxcenter=boxcenter_ap, fft_engine='fftw', fft_plan='estimate')

    recon.assign_data(data['Positions'])
    recon.set_density_contrast(smoothing_radius=smoothing)
    recon.run()

    # data positions post-rec
    data['Positions_rec'] = data['Positions'] - recon.read_shifts(data['Positions'], field='disp+rsd')
    data['Positions_rec'] = data['Positions_rec'] % boxsize_ap
    d = Table(data['Positions_rec'], names=('x', 'y', 'z'))
    ofile_data = ( output_dir + 
                  f'{tracer}_snap{snap}_displaced_{recalg}_nmesh{nmesh}_sm{int(smoothing):0>2d}_'
                  f'f{f:.3f}_b{bias:.2f}_Grid{ncosmo_grid}.fits' )
    print('Saving displaced field: \n', ofile_data, '\n')
    d.write(ofile_data, format='fits')


    # Randoms
    for r, seed in zip(randoms_list, seeds):
        randoms = {}
        randoms['Positions'] = r
        for convention, field in zip(['recsym','reciso'], ['disp+rsd','disp']):
            randoms['Positions_rec'] = randoms['Positions'] - recon.read_shifts(randoms['Positions'], field=field)
            randoms['Positions_rec'] = randoms['Positions_rec'] % boxsize_ap
            ofile = ( output_dir + 
                     f'randoms/{tracer}_snap{snap}_shifted_{recalg}_nmesh{nmesh}_sm{int(smoothing):0>2d}'
                     f'_f{f:.3f}_b{bias:.2f}_{convention}_Grid{ncosmo_grid}_S{seed}.fits' )
            randoms_table = Table(randoms['Positions_rec'], names=('x', 'y', 'z'))
            print('Saving shifted randoms: \n', ofile)
            randoms_table.write(ofile, format='fits')
    print('\n')
