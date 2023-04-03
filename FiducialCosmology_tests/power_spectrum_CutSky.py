import os, sys
import tempfile
import fitsio
import numpy as np
from cosmoprimo import fiducial
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, setup_logging, mpi
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction, utils
from astropy.table import Table
from time import time
setup_logging()

os.environ['NUMEXPR_MAX_THREADS'] = '32'
mpicomm = mpi.COMM_WORLD
mpiroot = None

# read fits file
def read(fn, columns=('x', 'y', 'z'), ext=1):
    gsize = fitsio.FITS(fn)[ext].get_nrows()
    start, stop = mpicomm.rank * gsize // mpicomm.size, (mpicomm.rank + 1) * gsize // mpicomm.size
    tmp = fitsio.read(fn, ext=ext, columns=columns, rows=range(start, stop))
    return tmp

# FKP weights
def w_fkp(nz, P0=1e4):
    return 1 / (1 + nz * P0)


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
    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ_MAIN', 'RA_REC', 'DEC_REC', 'Z_REC']
else:
    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ', 'RA_REC', 'DEC_REC', 'Z_REC']

# additional settings
nmesh_rec = 1024
nmesh_pk = 1024
step = 0.001

# directories 
paren_dir = f'/pscratch/sd/a/alexpzfz/KP4/Testing_fiducial_cosmo/CutSky/{tracer}/'
phases = [f'{i:0>2}' for i in range(25)]
odir = f'/global/cfs/projectdirs/desi/users/alexpzfz/kp4/Testing_fiducial_cosmo/CutSky/{tracer}/'

# cosmologies
cosmo_desi = fiducial.DESI()
names = ['000', '001', '002', '003', '004']

for name in names:
    cosmo = fiducial.AbacusSummit(name=name, engine='class')
    cosmoname = f'AbacusSummit_c{name}'

    for zbin in zbins:
        zmin = zbin['zmin']
        zmax = zbin['zmax']
        zmid = 0.5 * (zmin + zmax)

        # growth rate for file names
        f = cosmo.growth_rate(zbox).item()
        f *= cosmo.efunc(zbox)/cosmo.efunc(zmid)
        f *= (1 + zmid)/(1 + zbox)

        # loop over Abacus CutSky mocks
        for ph in phases:
            t0 = time()
            base_dir = paren_dir + f'AbacusSummit_base_c000_ph0{ph}/'
            file_name = (f'cutsky_{tracer}_zmin{zmin:.1f}_zmax{zmax:.1f}_displaced_multigrid_nmesh{nmesh_rec}_'
                            f'sm{int(smoothing):0>2d}_f{f:.3f}_b{bias:.2f}_{cosmoname}.fits')
            
            data_file = base_dir + file_name
            data = read(data_file, columns=columns)

            print('Loaded data file:\n', data_file)
            
            
            #for convention in ['recsym', 'reciso']:
            for convention in ['recsym']:
                randoms_file_name = (f'cutsky_{tracer}_zmin{zmin:.1f}_zmax{zmax:.1f}_randomsx20_shifted_multigrid_nmesh{nmesh}_'
                                     f'sm{int(smoothing):0>2d}_f{f:.3f}_b{bias:.2f}_{convention}_{cosmoname}.fits')
                randoms_file = base_dir + randoms_file_name
                randoms = read(randoms_file, columns=columns)
                print('Loaded randoms file:\n', randoms_file)
                
                positions = {}
                weights = {}
                
                print('Transforming sky coordinates to Cartesian coordinates')
                dist = cosmo.comoving_radial_distance(data['Z_REC'])
                positions['data'] = utils.sky_to_cartesian(dist, data['RA_REC'], data['DEC_REC'])
                dist = cosmo.comoving_radial_distance(randoms['Z_REC'])
                positions['randoms_shift'] = utils.sky_to_cartesian(dist, randoms['RA_REC'], randoms['DEC_REC'])
                dist = cosmo.comoving_radial_distance(randoms['Z'])
                positions['randoms'] = utils.sky_to_cartesian(dist, randoms['RA'], randoms['DEC'])
                
                print('Assigning weights\n')
                weights['data'] = w_fkp(data['NZ_MAIN'])
                weights['randoms'] = w_fkp(randoms['NZ_MAIN'])
                
                
                print('Beginning power spectrum calculation...')
                # Calculate power spectrum multipoles
                poles = CatalogFFTPower(data_positions1=positions['data'], data_weights1=weights['data'],
                                        randoms_positions1=positions['randoms'], randoms_weights1=weights['randoms'],
                                        shifted_positions1=positions['randoms_shift'], shifted_weights1=weights['randoms'],
                                        interlacing=2, resampler='tsc', ells=(0,2,4), nmesh=nmesh_pk, boxpad=1.2,
                                        edges={'step':step}, position_type='pos', mpicomm=mpicomm, dtype='f4').poles
                
                ofile = odir + ('Pk_' + '_'.join(file_name.replace('displaced_', '').split('_')[:-2]) +
                                f'_{convention}_{cosmoname}_ph0{ph}.npy')
                poles.save(ofile)
                if ph == '00':
                    poles.save_txt(ofile.replace('npy', 'txt'), complex=False)
                
                t1 = time()
                print('Elapsed time: ', t1-t0, 's\n')
