import os
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from mockfactory import EulerianLinearMock, LagrangianLinearMock, utils, setup_logging
from cosmoprimo.fiducial import DESI
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction, IterativeFFTParticleReconstruction
from pypower import CatalogFFTPower
from pyrecon.metrics import MeshFFTCorrelator, MeshFFTPropagator, CatalogMesh
from pathlib import Path
from astropy.io import fits
import math

setup_logging()


redshift = 1.1
bias = 1.2
#nbar = 
boxsize = 2000.
boxcenter = boxsize/2
los = 'z'
offset = boxcenter - boxsize/2
nmeshs = [1024]
recon_algos = ['multigrid']
conventions = ['reciso']
smoothing_radii = [15.0]
node = '000'
phases = []

for i in range(25):
    phases.append(f'{i:03d}')

# define fiducial cosmology
cosmo = DESI()
power = cosmo.get_fourier().pk_interpolator().to_1d(z=redshift)
f = (cosmo.sigma8_z(z=redshift, of='theta_cb')
    / cosmo.sigma8_z(z=redshift, of='delta_cb'))
H_0 = 100.0
az = 1/(1 + redshift)
Omega_m = cosmo._params['Omega_cdm'] + cosmo._params['Omega_b']
Omega_l = 1 - Omega_m
Hz = H_0 * np.sqrt(Omega_m * (1 + redshift)**3 + Omega_l)

#Load Cubic galaxy catalog
for phase in phases:
    data_dir =  f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/ELG/z{redshift:.3f}/AbacusSummit_base_c{node}_ph{phase}'
    
    data_x = np.empty(0)
    data_y = np.empty(0)
    data_z = np.empty(0)
    data_vx = np.empty(0)
    data_vy = np.empty(0)
    data_vz = np.empty(0)
    data = {}
    
    for i in range(0,64):
        data_dir =  f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/ELG/z{redshift:.3f}/AbacusSummit_base_c{node}_ph{phase}/'
        data_fn = data_dir + f'ELGlowDens_snap16_ph{phase}.gcat.sub{i}.fits'
        hdul = fits.open(data_dir + f'ELGlowDens_snap16_ph{phase}.gcat.sub{i}.fits')
        Data = hdul[1].data
        xtemp = Data['x'] 
        ytemp = Data['y'] 
        ztemp = Data['z'] 
        vxtemp  = Data['vx']
        vytemp  = Data['vy']
        vztemp  = Data['vz'] 
        hdul.close()
        data_x = np.hstack((data_x, xtemp ))
        data_y = np.hstack((data_y, ytemp ))
        data_z = np.hstack((data_z, ztemp ))
        data_vx = np.hstack((data_vx, vxtemp))
        data_vy = np.hstack((data_vy, vytemp))
        data_vz = np.hstack((data_vz, vztemp))
        print(" Sub-box number {} loaded".format(i))
    
    data['Position'] = np.array([data_x, data_y, data_z]).T
    data['Velocity'] = np.array([data_vx, data_vy, data_vz]).T
        #apply RSD
    data['Position'][:, 2] += data['Velocity'][:, 2] / (az * Hz)
    data['Position'] = (data['Position'] - offset) % boxsize + offset
        
    print(f"Shape of data pos: {np.shape(data['Position'])}")
        
#reconstruction
    output_dir1 = f'/global/cscratch1/sd/yunanxie/2pt/pyrecon/runs/output/CubicBox/AbacusSummit_base_c{node}_ph{phase}/reconstruction/data/'
    output_dir2 = f'/global/cscratch1/sd/yunanxie/2pt/pyrecon/runs/output/CubicBox/AbacusSummit_base_c{node}_ph{phase}/reconstruction/random/'
    Path(output_dir1).mkdir(parents=True, exist_ok=True)
    Path(output_dir2).mkdir(parents=True, exist_ok=True)
    for nmesh in nmeshs:
        for smooth_radius in smoothing_radii:
            for recon_algo in recon_algos:
                for convention in conventions:
                    if recon_algo == 'multigrid':
                        ReconstructionAlgorithm = MultiGridReconstruction 
                    elif recon_algo == 'ifft':
                        ReconstructionAlgorithm = IterativeFFTReconstruction 
                    elif recon_algo == 'ifftp':
                        ReconstructionAlgorithm = IterativeFFTParticleReconstruction 
    
                    recon = ReconstructionAlgorithm(
                        f=f, bias=bias, los=los, nmesh=nmesh,
                        boxsize=boxsize, boxcenter=boxcenter,
                        wrap=True)
                    recon.assign_data(data['Position'])
                    recon.set_density_contrast(smoothing_radius=smooth_radius)
                    recon.run()
                    
                    data['Position_rec'] = data['Position'] - recon.read_shifts(data['Position'], field='disp+rsd')
                    data['Position_rec'] = (data['Position_rec'] - offset) % boxsize + offset
                    output_fn = Path(output_dir1,
                        f'data_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                    np.save(output_fn, data)
                    
                    nden = len(data['Position']) / boxsize ** 3 
                    nrand = 20 * int(nden * boxsize ** 3)
                    nrand_split = int(nrand / 5)
                    for i in range(1, 6):
                        randoms = {}
                        randoms['Position'] = np.array([np.random.uniform(boxcenter - boxsize/2.,boxcenter + boxsize/2., nrand_split) for j in range(3)]).T
                        print(f"Shape of randoms_{i} pos: {np.shape(randoms['Position'])}")
    
                        if convention ==  'recsym':
                            field = 'disp+rsd'
                        elif convention == 'reciso':
                            field = 'disp'
                        else:
                            raise Exception('Invalid RSD convention.')
                        
                        randoms['Position_rec'] = randoms['Position'] - recon.read_shifts(randoms['Position'], field=field)
                        randoms['Position_rec'] = (randoms['Position_rec'] - offset) % boxsize + offset
                        output_fn = Path(output_dir2,f'randoms{i}_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                        np.save(output_fn, randoms)


