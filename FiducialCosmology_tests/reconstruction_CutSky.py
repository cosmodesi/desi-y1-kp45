import os, sys
import tempfile
import fitsio
import numpy as np
from cosmoprimo import fiducial, Background, Cosmology
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, setup_logging, mpi, PowerSpectrumMultipoles
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction, utils
from astropy.table import Table
from time import time
setup_logging()

# Function for applying binary mask
def mask(main=0, nz=0, Y5=0, sv3=0):
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

# Function to calculate nmesh that works properly with MultigridRec
def determine_nmesh(positions, smoothing, boxpad=1.5, epsilon=2.):
    xmin, xmax = positions[:,0].min(), positions[:,0].max()
    ymin, ymax = positions[:,1].min(), positions[:,1].max()
    zmin, zmax = positions[:,2].min(), positions[:,2].max()
    lx = xmax - xmin
    ly = ymax - ymin
    lz = zmax - zmin
    l = np.max([lx, ly, lz])
    boxsize = l * boxpad
    
    nmesh = 2
    cellsize = boxsize / nmesh
    while cellsize + epsilon >= smoothing:
        nmesh *= 2
        cellsize = boxsize / nmesh
    return nmesh


# settings dictionary
settings = {'LRG': {'zbox': 0.8,
                    'snap': 20,
                    'bias': 1.99,
                    'smoothing': 10,
                    'zbins': [{'zmin': 0.4, 'zmin':0.6},
                              {'zmin': 0.6, 'zmax': 0.8},
                              {'zmin': 0.8, 'zmax': 1.1}]},
            'ELG': {'zbox': 1.1,
                    'snap': 16,
                    'bias' 1.2,
                    'smoothing': 10,
                    'zbins': [{'zmin': 0.6, 'zmax': 0.8},
                              {'zmin': 0.8, 'zmax': 1.1},
                              {'zmin': 1.1, 'zmax': 1.6}]},
            'QSO': {'zbox': 1.4,
                    'snap': 12,
                    'bias': 2.07,
                    'smoothing': 15,
                    'zbins': [{'zmin': 0.8, 'zmax': 1.6},
                              {'zmin': 1.6, 'zmax': 2.1},
                              {'zmin': 2.1, 'zmax': 3.5}]}}


# input tracer as an argument when running the script
tracer = sys.argv[1].upper()
zbox = settings[tracer]['zbox']
bias = settings[tracer]['bias']
smoothing = settings[tracer]['smoothing']
zbins = settings[tracer]['zbins']

if tracer == 'LRG':
    mask_y5 = mask(main=1, nz=0, Y5=1, sv3=0)
    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ_MAIN']
else:
    mask_y5 = mask(main=0, nz=1, Y5=1, sv3=0)
    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ']

# additional settings for reconstruction
RecAlgorithm = MultiGridReconstruction

# directories
input_dir = f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CutSky/{tracer}/z{zbox:.3f}/'
phases = [f'{i:0>2}' for i in range(25)]
randoms_files = [input_dir + f'cutsky_{tracer}_random_S{seed}_1X.fits' for seed in range(100, 2100,100)]
output_dir = f'/pscratch/sd/a/alexpzfz/KP4/Testing_fiducial_cosmo/CutSky/{tracer}/'

# cosmologies
cosmo_desi = fiducial.DESI()
names = ['000', '001', '002', '003', '004']


for name in names:
    # Setting cosmology
    cosmo = fiducial.AbacusSummit(name=name, engine='class')
    cosmo_name = f'AbacusSummit_c{name}'
    dtoredshift = utils.DistanceToRedshift(cosmo.comoving_radial_distance)

    for zbin in zbins:
        # masking settings
        zmin = zbin['zmin']
        zmax = zbin['zmax']
        zmid = 0.5 * (zmin + zmax)

        # Randoms
        randoms_list = []
        positions = {}
        for file in randoms_files:
            randoms = fitsio.read(file, ext=1, columns=columns)
            status = randoms['STATUS']
            idx = np.arange(len(status))
            idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
            randoms = randoms[idx_Y5]
            mask_z = (randoms['Z']>=zmin) & (randoms['Z']<=zmax)
            randoms = randoms[mask_z]
            print(f'Succesfully read randoms:', file, sep='\n')
            randoms_list.append(randoms)
        randoms = np.concatenate(randoms_list)
        dist = cosmo.comoving_radial_distance(randoms['Z'])
        positions['randoms'] = utils.sky_to_cartesian(dist, randoms['RA'], randoms['DEC'])
        
        # reconstruction settings 
        f = cosmo.growth_rate(zbox).item()
        f *= cosmo.efunc(zbox) / cosmo.efunc(zmid)
        f *= (1 + zmid) / (1 + zbox)
        nmesh = determine_nmesh(positions['randoms'], smoothing)
        
        # Iterating over AbacusSummit boxes
        for ph in phases:
            odir = output_dir + f'AbacusSummit_base_c000_ph0{ph}/'
            positions_rec = {}

            # Data
            file = input_dir + f'cutsky_{tracer}_z{zbox:.3f}_AbacusSummit_base_c000_ph0{ph}.fits'
            data = fitsio.read(file, ext=1, columns=columns)
            status = data['STATUS']
            idx = np.arange(len(status))
            idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
            data = data[idx_Y5]
            mask_z = (data['Z']>=zmin) & (data['Z']<=zmax)
            data = data[mask_z]
            dist = cosmo.comoving_radial_distance(data['Z'])
            positions['data'] = utils.sky_to_cartesian(dist, data['RA'], data['DEC'])
            print(f'Succesfully read cutsky mock:', file, sep='\n') 


            # Reconstruction
            print('Beginning reconstruction')
            recon = RecAlgorithm(f=f, bias=bias, los='local', positions=positions['randoms'], dtype='f4',
                                 nmesh=nmesh, fft_engine='fftw', fft_plan='estimate')
            t0 = time()
            recon.assign_data(positions['data'])
            recon.assign_randoms(positions['randoms'])
            recon.set_density_contrast(smoothing_radius=smoothing)
            recon.run()
            t1 = time()
            print('\nElapsed time: ', t1-t0, 's\n')


            # Post-recon positions
            positions_rec['data'] = positions['data'] - recon.read_shifts(positions['data'], field='disp+rsd')
            distance, ra, dec = utils.cartesian_to_sky(positions_rec['data'])
            z = dtoredshift(distance)
            d = Table(data)
            d.add_columns([ra, dec, z], names=('RA_REC', 'DEC_REC', 'Z_REC'))
            ofile = odir + (f'cutsky_{tracer}_zmin{zmin:.1f}_zmax{zmax:.1f}_displaced_multigrid_nmesh{nmesh}_'
                            f'sm{int(smoothing):0>2d}_f{f:.3f}_b{bias:.2f}_{cosmo_name}.fits')
            d.write(ofile, format='fits')
            print('\nDisplaced data saved:', ofile, sep='\n')

            for convention, field in zip(['recsym','reciso'], ['disp+rsd','disp']):
                positions_rec['randoms'] = positions['randoms'] - recon.read_shifts(positions['randoms'], field=field)
                distance, ra, dec = utils.cartesian_to_sky(positions_rec['randoms'])
                z = dtoredshift(distance)
                r = Table(randoms)
                r.add_columns([ra, dec, z], names=('RA_REC', 'DEC_REC', 'Z_REC'))
                ofile = odir + (f'cutsky_{tracer}_zmin{zmin:.1f}_zmax{zmax:.1f}_randomsx20_shifted_multigrid_nmesh{nmesh}_'
                                f'sm{int(smoothing):0>2d}_f{f:.3f}_b{bias:.2f}_{convention}_{cosmo_name}.fits')
                r.write(ofile, format='fits')
                print('Shifted randoms saved:', ofile, sep='\n')
                print('\n\n')

