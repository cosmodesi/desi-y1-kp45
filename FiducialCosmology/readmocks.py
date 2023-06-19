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

settings_cutsky = {
            'LRG': {'zbox': 0.8,
                    'snap': 20,
                    'bias': 1.99,
                    'smoothing': 10,
                    'zbins': [{'zmin': 0.4, 'zmax':0.6},
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

class CubicBox:
    def __init__(self, 
                 tracer,
                 whichmocks='firstgen',
                 ncosmo_true='000',
                 ph=None,
                 rectype=None,
                 ncosmo_grid='000',
                 rec_settings=None,
                 boxsize=2000.,
                 boxcenter=1000.,
                 path=None
                ):
        
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        tracer = tracer.upper()
        ph = f'{ph:03d}'
        self.tracer = tracer
        self.ph = ph
        self.whichmocks = whichmocks
        self.ncosmo_true = ncosmo_true
        self.ncosmo_grid = ncosmo_grid
        self.rectype = rectype
        for key, value in settings_cubic[tracer].items():
            setattr(self, key, value)
        self.rec_settings = rec_settings if rec_settings else {'recalg': 'multigrid',
                                                               'nmesh': 512}
        self.path = path
        
        zbox = self.zbox
        snap = self.snap
        
        # set cosmologies
        cosmo_true = fiducial.AbacusSummit(name=ncosmo_true, engine='class')
        cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')
        self.f_true = cosmo_true.growth_rate(self.zbox).item()
        self.f_grid = cosmo_grid.growth_rate(self.zbox).item()
        
        self.boxsize_true = self.boxsize
        self.boxcenter_true = self.boxcenter
        
        self.q_par, self.q_perp, self.boxsize, self.boxcenter = self.get_AP_params()
        
        # set style for reading data
        if whichmocks=='firstgen':
            style ='firstgen'
        elif whichmocks=='sv3':
            if tracer=='LRG':
                style = 'sandy'
            elif tracer=='ELG':
                style = 'antoine'
        self.style = style
                
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
                self.filename = files
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
                

            elif style=='antoine':
                base_dir = '/global/cfs/cdirs/desi/users/arocher/mock_challenge_ELG/v2/ELG/z1.1/'
                file = base_dir + f'AbacusSummit_base_c{ncosmo_true}_ph{ph}/catalog_ELG_LNHOD_z1.1.fits'
                self.filename = file
                x, y, z = read(file, columns=('x','y','z_rsd'))

            
            positions = np.array([x, y, z])
            positions /= np.array([self.q_perp, self.q_perp, self.q_par], dtype='f4')[:,None]
            
        else:
            bias = self.bias
            smoothing = self.smoothing
            if not self.path:
                scratch = os.environ.get('SCRATCH')
                path = f'{scratch}/KP4/Testing_fiducial_cosmo/'
            self.path = path
                
            if style=='firstgen':
                wherefrom = f'AbacusSummit_base_c{ncosmo_true}_FirstGen_ph{ph}/'
            else:
                wherefrom = f'AbacusSummit_base_c{ncosmo_true}_SV3_ph{ph}/'
            
            base_dir = path + f'CubicBox/{tracer}/' + wherefrom
            f = cosmo_grid.growth_rate(zbox)
            recalg = self.rec_settings['recalg']
            nmesh = self.rec_settings['nmesh']
            file_name = ( f'{tracer}_snap{snap}_displaced_{recalg}_nmesh{nmesh}_sm{int(smoothing):0>2d}_'
                          f'f{f:.3f}_b{bias:.2f}_Grid{ncosmo_grid}.fits')
            file = base_dir + file_name
            self.filename = file
            x, y, z = read(file, columns=('x', 'y', 'z'))
            positions = np.array([x, y, z])
        
        self.positions = positions

        n = tot_len(self.positions[0])
        print0(f'Succesfully read data with ntracers={n} scattered across {mpicomm.size} ranks.')    
            
    def __call__(self):
        return self.positions
    
    
    def get_AP_params(self):
        cosmo_true = fiducial.AbacusSummit(name=self.ncosmo_true, engine='class')
        cosmo_grid = fiducial.AbacusSummit(name=self.ncosmo_grid, engine='class')
        
        # Calculate Alcock Paczynski dilation parameters
        h_true = cosmo_true.efunc(self.zbox).item()
        h_grid = cosmo_grid.efunc(self.zbox).item()
        da_true = cosmo_true.angular_diameter_distance(self.zbox).item()
        da_grid = cosmo_grid.angular_diameter_distance(self.zbox).item()
        q_par = h_grid / h_true
        q_perp = da_true / da_grid
        print0('\nRescaling parameters:')
        print0('q_par =', q_par)
        print0('q_perp =', q_perp, '\n')
        
        boxsize_ap = np.array([self.boxsize_true]*3, dtype='f4')
        boxsize_ap /= np.array([q_perp, q_perp, q_par], dtype='f4')
        boxcenter_ap = np.array([self.boxcenter_true]*3, dtype='f4')
        boxcenter_ap /= np.array([q_perp, q_perp, q_par], dtype='f4')
        
        return q_par, q_perp, boxsize_ap, boxcenter_ap
    
    
    def read_randoms(self, seeds=None, shifted=False):
        randoms_list = []
        if not seeds:
            seeds = list(range(100, 2100, 100))
        elif type(seeds) == int:
            seeds = [seeds]
            
        if not shifted:
            if self.style == 'firstgen':
                base_dir = ('/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'
                             f'CubicBox/RandomBox/{self.tracer}/')
                if self.tracer == 'ELG':
                    base_file_name = f'{self.tracer}lowDens_snap{self.snap}'
                else:
                    base_file_name = f'{self.tracer}_snap{self.snap}'

                for seed in seeds:
                    files = [base_dir + 
                            f'{base_file_name}_SB{i}_S{seed}_ph000.fits' for i in range(64)]
                    x_list = []
                    y_list = []
                    z_list = []
                    for file in files:
                        x, y, z = read(file, columns=('x','y','z'))
                        x_list.append(x)
                        y_list.append(y)
                        z_list.append(z)
                    x = np.concatenate(x_list)
                    y = np.concatenate(y_list)
                    z = np.concatenate(z_list)
                    randoms = np.array([x, y, z])
                    randoms /= np.array([self.q_perp, self.q_perp, self.q_par], dtype='f4')[:,None]
                    n = tot_len(randoms[0])
                    print0(f'Succesfully read randoms(seed={seed}) with ntracers={n} scattered across {mpicomm.size} ranks.')
                    randoms_list.append(randoms)
        else:
            cosmo_grid = fiducial.AbacusSummit(name=self.ncosmo_grid, engine='class')
            if self.style=='firstgen':
                wherefrom = f'AbacusSummit_base_c{self.ncosmo_true}_FirstGen_ph{self.ph}/'
            else:
                wherefrom = f'AbacusSummit_base_c{self.ncosmo_true}_SV3_ph{self.ph}/'
            
            base_dir = self.path + f'CubicBox/{self.tracer}/' + wherefrom + 'randoms/'
            f = cosmo_grid.growth_rate(self.zbox)
            recalg = self.rec_settings['recalg']
            nmesh = self.rec_settings['nmesh']
            for seed in seeds:
                file_name = ( f'{self.tracer}_snap{self.snap}_shifted_{recalg}_nmesh{nmesh}_sm{int(self.smoothing):0>2d}_'
                              f'f{f:.3f}_b{self.bias:.2f}_{self.rectype}_Grid{self.ncosmo_grid}_S{seed}.fits')
                file = base_dir + file_name
                x, y, z = read(file, columns=('x', 'y', 'z'))
                randoms = np.array([x, y, z])
                n = tot_len(randoms[0])
                print0(f'Succesfully read randoms(seed={seed}) with ntracers={n} scattered across {mpicomm.size} ranks.')
                randoms_list.append(randoms)
        
        return randoms_list if len(randoms_list)>1 else randoms_list[0]
            
    def generate_randoms(self, seeds=None, size=None):
        randoms_list = []
        if not seeds:
            seeds = list(range(100, 2100, 100))
        elif type(seeds) == int:
            seeds = [seeds]    
        if not size:
            size = self.positions.shape
            
        for seed in seeds:
            np.random.seed(seed)
            x, y, z = np.random.uniform(low=0.0, high=self.boxsize_true, size=size)
            x = x.astype('f4') / self.q_perp
            y = y.astype('f4') / self.q_perp
            z = z.astype('f4') / self.q_par  
            randoms = np.array([x, y, z])
            randoms_list.append(randoms)
            
        return randoms_list if len(randoms_list)>1 else randoms_list[0]
        
    def get_randoms(self, seeds=None, shifted=False, concat=False):
        if not shifted:
            if self.style == 'firstgen':
                randoms = self.read_randoms(seeds=seeds, shifted=shifted)
            else:
                randoms = self.generate_randoms(seeds=seeds)
        else:
            randoms = self.read_randoms(seeds=seeds, shifted=shifted)
            
        if concat:
            randoms = np.concatenate(randoms, axis=1)
        
        return randoms
    

    def get_ofilename(self, kind, out_path=None):
        
        if kind == 'recon':
            if not out_path:
                scratch = os.environ.get('SCRATCH')
                out_path = f'{scratch}/KP4/Testing_fiducial_cosmo/'
            
            if self.whichmocks=='firstgen':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_FirstGen_ph{self.ph}/'
            elif self.whichmocks=='sv3':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_SV3_ph{self.ph}/'
            
            output_dir = out_path + f'CubicBox/{self.tracer}/' + whereto
            recalg = self.rec_settings['recalg']
            nmesh = self.rec_settings['nmesh']
            file_name = ( f'{self.tracer}_snap{self.snap}_displaced_{recalg}_nmesh{nmesh}_sm{int(self.smoothing):0>2d}_'
                          f'f{self.f_grid:.3f}_b{self.bias:.2f}_Grid{self.ncosmo_grid}.fits')
            ofile = output_dir + file_name
            
        else:
            
            if self.whichmocks=='firstgen':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_FirstGen/'
            elif self.whichmocks=='sv3':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_SV3/'
                
            if not out_path:
                user = os.environ.get('USER')
                out_path = f'/global/cfs/projectdirs/desi/users/{user}/kp4/Testing_fiducial_cosmo/'
                
            if kind == 'pk':
                output_dir = out_path + f'CubicBox/Pk/{self.tracer}/' + whereto
                if not self.rectype:
                    ofile = output_dir + f'Pk_{self.tracer}_snap{self.snap}_Grid{self.ncosmo_grid}_ph{self.ph}.npy'
                else:
                    filename = os.path.basename(self.filename)
                    ofile = output_dir + ('Pk_' + '_'.join(filename.replace('displaced_', '').split('_')[:-1]) +
                                          f'_{self.rectype}_Grid{self.ncosmo_grid}_ph{self.ph}.npy')
            elif kind == 'xi':
                output_dir = out_path + f'CubicBox/Xi/{self.tracer}/' + whereto
                if not self.rectype:
                    ofile = output_dir + f'Xi_{self.tracer}_snap{self.snap}_Grid{self.ncosmo_grid}_ph{self.ph}.npy'
                else:
                    filename = os.path.basename(self.filename)
                    ofile = output_dir + ('Xi_' + '_'.join(filename.replace('displaced_', '').split('_')[:-1]) +
                                          f'_{self.rectype}_Grid{self.ncosmo_grid}_ph{self.ph}.npy')
            
            
        if not os.path.exists(output_dir):
            if mpicomm.rank==0: # avoid error if multiple tasks try to create the directory
                if kind == 'recon':
                    os.makedirs(output_dir + 'randoms/')
                else:
                    os.makedirs(output_dir)
            
        return ofile
        
    def get_randoms_ofilename(self, seed, convention, kind='recon', out_path=None):
        base_file_name = self.get_ofilename(kind, out_path)
        dirname = os.path.dirname(base_file_name)
        file_name = os.path.basename(base_file_name)
        dirname += '/randoms/'
        file_name = file_name.replace('displaced', 'shifted')
        file_name = file_name.replace(f'Grid{self.ncosmo_grid}', f'{convention}_Grid{self.ncosmo_grid}_S{seed}')
        ofile = dirname + file_name
        
        return ofile
    
    
class CutSky:
    def __init__(self, 
                 tracer,
                 whichmocks='firstgen',
                 ncosmo_true='000',
                 ph=None,
                 rectype=None,
                 ncosmo_grid='000',
                 rec_settings=None,
                 nzbin=None,
                 path=None
                ):
        
        tracer = tracer.upper()
        ph = f'{ph:03d}'
        self.tracer = tracer
        self.ph = ph
        self.whichmocks = whichmocks
        self.ncosmo_true = ncosmo_true
        self.ncosmo_grid = ncosmo_grid
        self.rectype = rectype
        for key, value in settings_cutsky[tracer].items():
            setattr(self, key, value)
            
        self.zbin = self.zbins[nzbin]
        self.zbin['zmid'] = 0.5 * (self.zbin['zmin'] + self.zbin['zmax'])
        self.rec_settings = rec_settings if rec_settings else {'recalg': 'multigrid',
                                                               'cellsize': 2000./512}
        self.path = path
        
        # set cosmologies
        cosmo_true = fiducial.AbacusSummit(name=ncosmo_true, engine='class')
        cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')
        
        f_values = {}
        for cosmo, name in zip([cosmo_true, cosmo_grid], ['f_true', 'f_grid']):
            f = cosmo.growth_rate(self.zbox).item()
            f *= cosmo.efunc(self.zbox) / cosmo.efunc(self.zbin['zmid'])
            f *= (1 + self.zbin['zmid']) / (1 + self.zbox)
            f_values[name] = f
        self.f_true = f_values['f_true']
        self.f_grid = f_values['f_grid']
        
        # set style for reading data
        if whichmocks=='firstgen':
            style ='firstgen'
        elif whichmocks=='sv3':
            if tracer=='LRG':
                style = 'sandy'
            elif tracer=='ELG':
                style = 'antoine'
        self.style = style
        
        if not rectype:
            if style=='firstgen':
                base_dir = ('/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'
                             f'CutSky/{tracer}/z{self.zbox:.3f}/')
                    
                if tracer == 'LRG':
                    mask_y5 = mask(main=1, nz=0, Y5=1, sv3=0)
                    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ_MAIN']
                else:
                    mask_y5 = mask(main=0, nz=1, Y5=1, sv3=0)
                    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ']
                    
                file = base_dir + f'cutsky_{tracer}_z{self.zbox:.3f}_AbacusSummit_base_c000_ph{ph}.fits'
                data = fitsio.read(file, ext=1, columns=columns) #define columns
                status = data['STATUS']
                idx = np.arange(len(status))
                idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
                data = data[idx_Y5]
                mask_z = (data['Z']>=self.zbin['zmin']) & (data['Z']<self.zbin['zmax'])
                data = data[mask_z]
                dist = cosmo_grid.comoving_radial_distance(data['Z'])
                positions =  utils.sky_to_cartesian([dist, data['RA'], data['DEC']])
                self.positions = positions
            else:
                raise NotImplementedError
            
        n = tot_len(self.positions[0])
        print0(f'Succesfully read data with ntracers={n} scattered across {mpicomm.size} ranks.')    
            
    def __call__(self):
        return self.positions
    
    
                
    def read_randoms(self, seeds=None, shifted=False):
        randoms_list = []
        if not seeds:
            seeds = list(range(100, 2100, 100))
        elif type(seeds) == int:
            seeds = [seeds]
            
        if not shifted:
            if self.style == 'firstgen':
                base_dir = ('/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'
                             f'CutSky/{self.tracer}/z{self.zbox:.3f}/')
                
                if self.tracer == 'LRG':
                    mask_y5 = mask(main=1, nz=0, Y5=1, sv3=0)
                    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ_MAIN']
                else:
                    mask_y5 = mask(main=0, nz=1, Y5=1, sv3=0)
                    columns = ['RA', 'DEC', 'Z', 'STATUS', 'NZ']
                    
                    
                for seed in seeds:
                    file = base_dir + f'cutsky_{self.tracer}_random_S{seed}_1X.fits'
                    randoms = fitsio.read(file, ext=1, columns=columns)
                    status = randoms['STATUS']
                    idx = np.arange(len(status))
                    idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
                    randoms = randoms[idx_Y5]
                    mask_z = (randoms['Z']>=self.zbin['zmin']) & (randoms['Z']<self.zbin['zmax'])
                    randoms = randoms[mask_z]
                    
                    cosmo_grid = fiducial.AbacusSummit(name=self.ncosmo_grid, engine='class')
                    dist = cosmo_grid.comoving_radial_distance(randoms['Z'])
                    positions =  utils.sky_to_cartesian([dist, randoms['RA'], randoms['DEC']])
                    n = tot_len(positions[0])
                    print0(f'Succesfully read randoms(seed={seed}) with ntracers={n} scattered across {mpicomm.size} ranks.')
                    randoms_list.append(positions)
                    
        return randoms_list if len(randoms_list)>1 else randoms_list[0]
    
    
    def get_randoms(self, seeds=None, shifted=False, concat=False):
        if not shifted:
            if self.style == 'firstgen':
                randoms = self.read_randoms(seeds=seeds, shifted=shifted)
            else:
                randoms = self.generate_randoms(seeds=seeds)
        else:
            randoms = self.read_randoms(seeds=seeds, shifted=shifted)
            
        if concat:
            randoms = np.concatenate(randoms, axis=1)
        
        return randoms
    
    
    def get_ofilename(self, kind, out_path=None, mkdir=True):
        
        if kind == 'recon':
            if not out_path:
                scratch = os.environ.get('SCRATCH')
                out_path = f'{scratch}/KP4/Testing_fiducial_cosmo/'
            
            if self.whichmocks=='firstgen':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_FirstGen_ph{self.ph}/'
            elif self.whichmocks=='sv3':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_SV3_ph{self.ph}/'
            
            output_dir = out_path + f'CutSky/{self.tracer}/' + whereto
            recalg = self.rec_settings['recalg']
            cellsize = self.rec_settings['cellsize']
            file_name = ( f'cutsky_{self.tracer}_zmin{self.zbin["zmin"]:.1f}_zmax{self.zbin["zmax"]:.1f}_displaced_{recalg}_cellsize{cellsize:.1f}_'
                          f'sm{int(self.smoothing):0>2d}_f{self.f_grid:.3f}_b{self.bias:.2f}_Grid{self.ncosmo_grid}.fits' )
            ofile = output_dir + file_name
            
        else:
            
            if self.whichmocks=='firstgen':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_FirstGen/'
            elif self.whichmocks=='sv3':
                whereto = f'AbacusSummit_base_c{self.ncosmo_true}_SV3/'
                
            if not out_path:
                user = os.environ.get('USER')
                out_path = f'/global/cfs/projectdirs/desi/users/{user}/kp4/Testing_fiducial_cosmo/'
                
            if kind == 'pk':
                output_dir = out_path + f'CubicBox/Pk/{self.tracer}/' + whereto
                if not self.rectype:
                    ofile = output_dir + f'Pk_{self.tracer}_snap{self.snap}_Grid{self.ncosmo_grid}_ph{self.ph}.npy'
                else:
                    filename = os.path.basename(self.filename)
                    ofile = output_dir + ('Pk_' + '_'.join(filename.replace('displaced_', '').split('_')[:-1]) +
                                          f'_{self.rectype}_Grid{self.ncosmo_grid}_ph{self.ph}.npy')
            elif kind == 'xi':
                output_dir = out_path + f'CubicBox/Xi/{self.tracer}/' + whereto
                if not self.rectype:
                    ofile = output_dir + f'Xi_{self.tracer}_snap{self.snap}_Grid{self.ncosmo_grid}_ph{self.ph}.npy'
                else:
                    filename = os.path.basename(self.filename)
                    ofile = output_dir + ('Xi_' + '_'.join(filename.replace('displaced_', '').split('_')[:-1]) +
                                          f'_{self.rectype}_Grid{self.ncosmo_grid}_ph{self.ph}.npy')
            
        if mkdir:    
            if not os.path.exists(output_dir):
                if mpicomm.rank==0: # avoid error if multiple tasks try to create the directory
                    if kind == 'recon':
                        os.makedirs(output_dir + 'randoms/')
                    else:
                        os.makedirs(output_dir)
            
        return ofile
    
    def get_randoms_ofilename(self, seed, convention, kind='recon', out_path=None):
        base_file_name = self.get_ofilename(kind, out_path, mkdir=False)
        dirname = os.path.dirname(base_file_name)
        file_name = os.path.basename(base_file_name)
        dirname += '/randoms/'
        file_name = file_name.replace('displaced', 'shifted')
        file_name = file_name.replace(f'Grid{self.ncosmo_grid}', f'{convention}_Grid{self.ncosmo_grid}_S{seed}')
        ofile = dirname + file_name
        
        return ofile