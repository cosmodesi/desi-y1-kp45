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

# read function for fits file
def read(fn, columns=('x', 'y', 'z'), ext=1):
    gsize = fitsio.FITS(fn)[ext].get_nrows()
    start, stop = mpicomm.rank * gsize // mpicomm.size, (mpicomm.rank + 1) * gsize // mpicomm.size
    tmp = fitsio.read(fn, ext=ext, columns=columns, rows=range(start, stop))
    return [tmp[col] for col in columns]

# MPI
mpicomm = mpi.COMM_WORLD
mpiroot = None

# settings dictionary
settings = {'LRG': {'zbox': 0.8,
                    'snap': 20,
                    'bias': 1.99,
                    'smoothing': 10},
            'ELG': {'zbox': 1.1,
                    'snap': 16,
                    'bias' 1.2,
                    'smoothing': 10},
            'QSO': {'zbox': 1.4,
                    'snap': 12,
                    'bias': 2.07,
                    'smoothing': 15}}

# input tracer as an argument when running the script
tracer = sys.argv[1].upper()
zbox = settings[tracer]['zbox']
bias = settings[tracer]['bias']
snap = settings[tracer]['snap']
smoothing = settings[tracer]['smoothing']

# Additional settings
boxsize = 2000.
boxcenter = 1000.
nmesh_pk = 512
los = 'z'
nmesh_rec = 512
step = 0.005

# directories 
paren_dir = f'/pscratch/sd/a/alexpzfz/KP4/Testing_fiducial_cosmo/CubicBox/{tracer}/'
phases = [f'{i:0>2}' for i in range(25)]
odir = f'/global/cfs/projectdirs/desi/users/alexpzfz/kp4/Testing_fiducial_cosmo/CubicBox/{tracer}/'

# cosmologies
cosmo_desi = fiducial.DESI()
names = ['000', '001', '002', '003', '004']

for name in names:
    cosmo = fiducial.AbacusSummit(name=name, engine='class') 
    cosmo_name = 'AbacusSummit_c' + name

    # Calculate Alcock Paczynski dilation parameters
    h_fid, h_real = cosmo.efunc(zbox).item(), cosmo_desi.efunc(zbox).item()
    da_fid, da_real = cosmo.angular_diameter_distance(zbox).item(), cosmo_desi.angular_diameter_distance(zbox).item()
    alpha_par = h_fid / h_real
    alpha_perp = da_real / da_fid
    print('alpha_par =', alpha_par)
    print('alpha_perp =', alpha_perp, '\n')
    boxsize_ap = np.array([boxsize]*3, dtype='f4')
    boxsize_ap /= np.array([alpha_perp, alpha_perp, alpha_par], dtype='f4')
    boxcenter_ap = np.array([boxcenter]*3, dtype='f4')
    boxcenter_ap /= np.array([alpha_perp, alpha_perp, alpha_par], dtype='f4')

    # growth rate for file names
    f = cosmo.growth_rate(zbox)

    # loop over Abacus boxes
    for ph in phases:
        base_dir = paren_dir + f'AbacusSummit_base_c000_ph0{ph}/'
        file_name = ( f'{tracer}_snap{snap}_displaced_multigrid_nmesh{nmesh_rec}_'
                      f'sm{int(smoothing):0>2d}_f{f:.3f}_b{bias:.2f}_{cosmo_name}.fits' )
        data_dir = base_dir + file_name
        x, y, z = read(data_dir)
        data = np.array([x, y, z]).T

        #for convention in ['recsym', 'reciso']:
        for convention in ['recsym']:
            randoms_dirs = [base_dir + 'randoms/' + 
                            '_'.join(file_name.replace('displaced', 'shifted').split('_')[:-2]) +
                            f'_{convention}_{cosmo_name}_S{seed}.fits'  for seed in range(100, 2100, 100)]
            xr_list = []
            yr_list = []
            zr_list = []
            for randoms_dir in randoms_dirs:
                xr, yr, zr = read(randoms_dir)
                xr_list.append(xr)
                yr_list.append(yr)
                zr_list.append(zr)
            xr = np.concatenate(xr_list)
            yr = np.concatenate(yr_list)
            zr = np.concatenate(zr_list)
            randoms = np.array([xr, yr, zr]).T
            print(mpicomm.rank, len(randoms))

            # Calculate power spectrum multipoles
            poles = CatalogFFTPower(data_positions1=data, shifted_positions1=randoms,
                                    boxsize=boxsize_ap, boxcenter=boxcenter_ap, nmesh=nmesh_pk, resampler='tsc',
                                    interlacing=2, ells=(0, 2, 4), los=los, edges={'step':step}, position_type='pos', 
                                    mpicomm=mpicomm, mpiroot=mpiroot, dtype='f4').poles

            ofile = odir + ('Pk_' + '_'.join(file_name.replace('displaced_', '').split('_')[:-2]) +
                            f'_{convention}_{cosmo_name}_ph0{ph}.npy') 
            poles.save(ofile)
            if ph == '00':
                poles.save_txt(ofile.replace('npy', 'txt'), complex=False)
