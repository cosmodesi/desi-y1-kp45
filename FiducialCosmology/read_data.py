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


mpicomm = mpi.COMM_WORLD
mpiroot = None

# settings dictionary
settings_cubic = {'LRG': {'zbox': 0.8,
                          'snap': 20,
                          'bias': 1.99,
                          'smoothing': 10},
                  'ELG': {'zbox': 1.1,
                          'snap': 16,
                          'bias': 1.2,
                          'smoothing': 10},
                  'QSO': {'zbox': 1.4,
                          'snap': 12,
                          'bias': 2.07,
                          'smoothing': 15}}

settings_cutsky = {'LRG': {'zbox': 0.8,
                    'snap': 20,
                    'bias': 1.99,
                    'smoothing': 10,
                    'zbins': [{'zmin': 0.4, 'zmin':0.6},
                              {'zmin': 0.6, 'zmax': 0.8},
                              {'zmin': 0.8, 'zmax': 1.1}]},
                    'ELG': {'zbox': 1.1,
                            'snap': 16,
                            'bias': 1.2,
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

# read function for fits file
def read(fn, columns=('x', 'y', 'z'), ext=1):
    gsize = fitsio.FITS(fn)[ext].get_nrows()
    start, stop = mpicomm.rank * gsize // mpicomm.size, (mpicomm.rank + 1) * gsize // mpicomm.size
    tmp = fitsio.read(fn, ext=ext, columns=columns, rows=range(start, stop))
    return [tmp[col] for col in columns]

# print messages only once
def print0(*messages):
    if mpicomm.rank == 0:
        print(*messages)

# to print number of tracers
def tot_len(array):
    n = len(array)
    if mpicomm.rank != 0:
        mpicomm.send(n, dest=0)
    elif mpicomm.rank == 0:
        for i in range(1, mpicomm.size):
            n += mpicomm.recv(source=i)
    return n


# Function for applying binary mask
def mask(main=0, nz=0, Y5=0, sv3=0):
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

def apply_mask(file, zmin, zmax, nz='NZ', extracols=None):
    mask_y5 = mask(main=1, nz=0, Y5=1, sv3=0)
    columns = ['RA', 'DEC', 'Z', 'STATUS', nz]
    if extracols:
        columns += extracols
    data = fitsio.read(file, ext=1, columns=columns)
    status = data['STATUS']
    idx = np.arange(len(status))
    idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
    data = data[idx_Y5]
    mask_z = (data['Z']>=zmin) & (data['Z']<zmax)
    data = data[mask_z]
    
    return data


def read_randoms(mocktype, tracer, rectype=None, nzbin=None, path=None, style='firstgen',
                 ncosmo_true=None, ncosmo_grid=None, ph=None, rec_settings=None):
    seeds = list(range(100, 2100, 100))
    randoms_list = []
    
    if mocktype=='cubicbox':
        settings = settings_cubic
        zbox = settings[tracer]['zbox']
        bias = settings[tracer]['bias']
        snap = settings[tracer]['snap']
        smoothing = settings[tracer]['smoothing']

        if not rectype:
            base_dir = ('/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'
                         f'CubicBox/RandomBox/{tracer}/')
            if tracer == 'ELG':
                base_file_name = f'{tracer}lowDens_snap{snap}'
            else:
                base_file_name = f'{tracer}_snap{snap}'

            for seed in seeds:
                files = [base_dir + 
                        f'{base_file_name}_SB{i}_S{seed}_ph000.fits' for i in range(64)]
                x_list = []
                y_list = []
                z_list = []
                randoms = {}
                for file in files:
                    x, y, z = read(file, columns=('x','y','z'))
                    x_list.append(x)
                    y_list.append(y)
                    z_list.append(z)
                x = np.concatenate(x_list)
                y = np.concatenate(y_list)
                z = np.concatenate(z_list)
                randoms = np.array([x, y, z])
                n = tot_len(randoms[0])
                print0(f'Succesfully read randoms(seed={seed}) with ntracers={n} scattered across {mpicomm.size} ranks.')
                randoms_list.append(randoms)
                
        else:
            cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')
            if style=='firstgen':
                wherefrom = f'AbacusSummit_base_c{ncosmo_true}_FirstGen_ph{ph}/'
            else:
                wherefrom = f'AbacusSummit_base_c{ncosmo_true}_SV3_ph{ph}/'
            
            base_dir = path + f'CubicBox/{tracer}/' + wherefrom + 'randoms/'
            f = cosmo_grid.growth_rate(zbox)
            recalg = rec_settings['recalg']
            nmesh = rec_settings['nmesh']
            for seed in seeds:
                file_name = ( f'{tracer}_snap{snap}_shifted_{recalg}_nmesh{nmesh}_sm{int(smoothing):0>2d}_'
                              f'f{f:.3f}_b{bias:.2f}_{rectype}_Grid{ncosmo_grid}_S{seed}.fits')
                file = base_dir + file_name
                x, y, z = read(file, columns=('x', 'y', 'z'))
                randoms = np.array([x, y, z])
                n = tot_len(randoms[0])
                print0(f'Succesfully read randoms(seed={seed}) with ntracers={n} scattered across {mpicomm.size} ranks.')
                randoms_list.append(randoms)

    elif mocktype=='cutsky':
        settings = settings_cutsky
        zbox = settings[tracer]['zbox']
        bias = settings[tracer]['bias']
        smoothing = settings[tracer]['smoothing']
        zbins = settings[tracer]['zbins']
        zbin = zbins[nzbin]

        zmin = zbin['zmin']
        zmax = zbin['zmax']
        zmid = 0.5 * (zmin + zmax)
            
        if not rec:
            base_dir = ('/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'
                         f'CutSky/{tracer}/z{zbox:.3f}/')
            
            nz = 'NZ_MAIN' if tracer=='LRG' else 'NZ'          
            for seed in seeds:
                file = base_dir + f'cutsky_{tracer}_random_S{seed}_1X.fits'
                randoms = apply_mask(file, zmin, zmax, nz)
                randoms_list.append(randoms)        
    
    return randoms_list

def generate_randoms(seeds, size, boxsize=2000.):
    randoms_list = []
    for seed in seeds:
        np.random.seed(seed)
        x, y, z = np.random.uniform(low=0.0, high=boxsize, size=size)
        x = x.astype('f4')
        y = y.astype('f4')
        z = z.astype('f4')    
        randoms = np.array([x, y, z])
        randoms_list.append(randoms)
    
    return randoms_list
    

def read_data(mocktype, tracer, ncosmo_true, ph, rectype=None, style='firtsgen',
              path=None, ncosmo_grid=None, rec_settings=None):
    cosmo_true = fiducial.AbacusSummit(name=ncosmo_true, engine='class')
    
    if mocktype=='cubicbox':
        settings = settings_cubic
        zbox = settings[tracer]['zbox']
        bias = settings[tracer]['bias']
        snap = settings[tracer]['snap']
        smoothing = settings[tracer]['smoothing']
        
        if not rectype:
            
            if style=='firstgen':
                base_dir = ('/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/'
                            f'AbacusSummit/CubicBox/{tracer}/z{zbox:.3f}/')
                if tracer == 'ELG':
                    base_file_name = f'{tracer}lowDens_snap{snap}'
                else:
                    base_file_name = f'{tracer}_snap{snap}'
                    
                files = [base_dir + f'AbacusSummit_base_c{ncosmo_true}_ph{ph}/'
                         f'{base_file_name}_ph{ph}.gcat.sub{i}.fits' for i in range(64)]
                x_list = []
                y_list = []
                z_list = []
                vz_list = []
                for file in files:
                    x, y, z, vz = read(file, columns=('x','y','z','vz'))
                    x_list.append(x)
                    y_list.append(y)
                    z_list.append(z)
                    vz_list.append(vz)
                x = np.concatenate(x_list)
                y = np.concatenate(y_list)
                z = np.concatenate(z_list)
                vz = np.concatenate(vz_list)
                
                # Add RSD to the z coordinate
                boxsize = 2000.
                H = 100 * cosmo_true.efunc(zbox)
                a = 1 / (1 + zbox)
                z = (z + vz/(a*H)) % boxsize

                data = np.array([x, y, z])

                
            elif style=='antoine':
                base_dir = '/global/cfs/cdirs/desi/users/arocher/mock_challenge_ELG/v2/ELG/z1.1/'
                file = base_dir + f'AbacusSummit_base_c{ncosmo_true}_ph{ph}/catalog_ELG_LNHOD_z1.1.fits'
                x, y, z = read(file, columns=('x','y','z_rsd'))
                
                data = np.array([x, y, z])
                

            
        else:
            cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')
            if style=='firstgen':
                wherefrom = f'AbacusSummit_base_c{ncosmo_true}_FirstGen_ph{ph}/'
            else:
                wherefrom = f'AbacusSummit_base_c{ncosmo_true}_SV3_ph{ph}/'
            
            base_dir = path + f'CubicBox/{tracer}/' + wherefrom
            f = cosmo_grid.growth_rate(zbox)
            recalg = rec_settings['recalg']
            nmesh = rec_settings['nmesh']
            file_name = ( f'{tracer}_snap{snap}_displaced_{recalg}_nmesh{nmesh}_sm{int(smoothing):0>2d}_'
                          f'f{f:.3f}_b{bias:.2f}_Grid{ncosmo_grid}.fits')
            file = base_dir + file_name
            x, y, z = read(file, columns=('x', 'y', 'z'))
            data = np.array([x, y, z])
    
    n = tot_len(data[0])
    print0(f'Succesfully read data with ntracers={n} scattered across {mpicomm.size} ranks.')
            
    if not rectype:
        return data
    else:
        return data, file_name
