import os, sys
import tempfile
import fitsio
import numpy as np
from cosmoprimo import fiducial
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging, mpi, PowerSpectrumMultipoles
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction
from astropy.table import Table
from time import time
setup_logging()

# read function for fit files
def read(fn, columns=('x', 'y', 'z'), ext=1):
    tmp = fitsio.read(fn, ext=ext, columns=columns)
    return [tmp[col] for col in columns]

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

# additional settings for reconstruction
boxsize = 2000.0
boxcenter = 1000.0
los = 'z'
ReconstructionAlgorithm = MultiGridReconstruction
nmesh = 512

# directories
base_data_dir = f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/{tracer}/z{zbox:.3f}/'
phases = [f'{i:0>2}' for i in range(25)]
base_randoms_dir = f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/RandomBox/{tracer}/'
odir = f'/pscratch/sd/a/alexpzfz/KP4/Testing_fiducial_cosmo/CubicBox/{tracer}/'

if tracer == 'ELG':
    base_file_name = f'{tracer}lowDens_snap{snap}'
else:
    base_file_name = f'{tracer}_snap{snap}'


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
    
    # growth rate
    f = cosmo.growth_rate(redshift)

    # read randoms
    seeds = list(range(100, 2100, 100))
    randoms_list = []
    for seed in range(100, 2100, 100):
        dir_list = [base_randoms_dir + 
                    f'{base_file_name}_SB{i}_S{seed}_ph000.fits' for i in range(64)]
        x_list = []
        y_list = []
        z_list = []
        randoms = {}
        for file in dir_list:
            x, y, z = read(file, columns=('x','y','z'))
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        z = np.concatenate(z_list)
        x = x / alpha_perp
        y = y / alpha_perp
        z = z / alpha_par    
        randoms['Position'] = np.array([x, y, z]).T
        print(f'Succesfully read randoms box S{seed}.')
        print(f'Number of tracers: {len(x)} \n')
        randoms_list.append(randoms)

    # Loop over Abacus boxes
    for ph in phases:
        data_dir = [base_data_dir + 
                    f'AbacusSummit_base_c000_ph0{ph}/{base_file_name}_ph0{ph}.gcat.sub{i}.fits' for i in range(64)]
        output_dir = odir + f'AbacusSummit_base_c000_ph0{ph}/'
        x_list = []
        y_list = []
        z_list = []
        vz_list = []
        for file in data_dir:
            x, y, z, vz = read(file, columns=('x','y','z','vz'))
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            vz_list.append(vz)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        z = np.concatenate(z_list)
        vz = np.concatenate(vz_list)

        print(f'\nSuccesfully read box {ph}')
        print(f'Number of tracers: {len(x)}\n')

        # Add RSD to the z coordinate
        H = 100 * cosmo_desi.efunc(redshift)
        a = 1 / (1 + redshift)
        z = (z + vz/(a*H)) % boxsize

        # Alcock Paczynski
        x = x / alpha_perp
        y = y / alpha_perp
        z = z / alpha_par

        data = {}
        data['Position'] = np.array([x, y, z]).T

        # RECONSTRUCTION
        print('Begining reconstruction')
        recon = ReconstructionAlgorithm(f=f, bias=bias, los=los, nmesh=nmesh, boxsize=boxsize_ap, dtype='f4',
                                    boxcenter=boxcenter_ap, fft_engine='fftw', fft_plan='estimate')

        t0 = time()
        recon.assign_data(data['Position'])
        recon.set_density_contrast(smoothing_radius=smoothing)
        recon.run()
        t1 = time()
        print('Elapsed time: ', t1-t0, 's\n')

        # data positions post-rec
        print('Saving displaced field')
        data['Position_rec'] = data['Position'] - recon.read_shifts(data['Position'], field='disp+rsd')
        data['Position_rec'] = data['Position_rec'] % boxsize_ap
        d = Table(data['Position_rec'], names=('x', 'y', 'z'))
        ofile_data = ( output_dir + 
                      f'{tracer}_snap{snap}_displaced_multigrid_nmesh{nmesh}_sm{int(smoothing):0>2d}_'
                      f'f{f:.3f}_b{bias:.2f}_{cosmo_name}.fits' )
        d.write(ofile_data, format='fits')


        # Randoms
        print('Saving randoms \n')
        for randoms, seed in zip(randoms_list, seeds):
            for convention, field in zip(['recsym','reciso'], ['disp+rsd','disp']):
                randoms['Position_rec'] = randoms['Position'] - recon.read_shifts(randoms['Position'], field=field)
                randoms['Position_rec'] = randoms['Position_rec'] % boxsize_ap
                ofile = ( output_dir + 
                         f'randoms/{tracer}_snap{snap}_shifted_multigrid_nmesh{nmesh}_sm{int(smoothing):0>2d}'
                         f'_f{f:.3f}_b{bias:.2f}_{convention}_{cosmo_name}_S{seed}.fits' )
                randoms_table = Table(randoms['Position_rec'], names=('x', 'y', 'z'))
                randoms_table.write(ofile, format='fits')
