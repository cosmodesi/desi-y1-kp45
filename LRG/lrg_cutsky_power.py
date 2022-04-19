import os
import logging
import numpy as np
import fitsio
from pyrecon import utils
from pypower import setup_logging, mpi, CatalogMesh, CatalogFFTPower
from cosmoprimo.fiducial import DESI


# --- settings for Abacus cut-sky mocks --- #
randoms_factor = 20 # how many randoms files to concatenate, maximum 50
bias = 2.35 # estimated value that shouldn't be too far off? can check other values too
zranges = np.array([{'min': 0.8, 'max': 1.1}])#, , {'min': 0.6, 'max': 0.8}, {'min': 0.4, 'max': 0.6}, ]) 
cap = 'ngc' # 'ngc', 'sgc' or 'both' (default) 
# ----------------------------------------- #

# --- options for pypower calculation --- #
interlacing = 2
resampling = 'tsc'
pknmesh = 512
ells = (0, 2, 4)
kedges = {'step':0.001}
# --------------------------------------- #

mpicomm = mpi.COMM_WORLD
logger = logging.getLogger('Main')
setup_logging()
cosmo = DESI()
data_dir = '/global/cfs/cdirs/desi/users/nadathur/Y1MockChallenge/LRG/CutSky/recon_catalogues/'
output_dir = '/global/cfs/cdirs/desi/users/nadathur/Y1MockChallenge/LRG/CutSky/Pk/'

def fkp_weights(nz, P0=10000):
    return 1 / (1 + nz * P0)

captxt = f'{cap.upper()}_' if cap in ['ngc', 'sgc'] else ''                            
for ph in range(0, 25):    
    # ----- options for recon runs ------ #
    if ph == 0:
        recon_types = ['MultiGridReconstruction', 'IterativeFFTReconstruction', 'IterativeFFTParticleReconstruction'] 
        recon_types = ['IterativeFFTParticleReconstruction'] 
        smooth_scales = [7.5, 10, 15]
        conventions = ['recsym' , 'reciso', 'rsd']
    else:
        recon_types = ['MultiGridReconstruction'] 
        smooth_scales = [10]
        conventions = ['recsym']
    cellsizes = [7.8] 
    boxpads = [1.5]
    nthreads = int(os.environ.get('OMP_NUM_THREADS'))
    
    for zr in zranges:
        done_original = False
        zmid = 0.5 * (zr['min'] + zr['max']) # approximation to maybe be improved later?        
        ztxt = f'{zr["min"]:0.1f}z{zr["max"]:0.1f}'       
        f = cosmo.growth_rate(zmid)
        betatxt = f'f{f:0.3f}_b{bias:0.2f}'
        for recname in recon_types:
            rectxt = f'{recname.replace("Reconstruction", "")}'
            if 'IterativeFFT' in recname and ph == 0:
                niterations = [3, 5, 7]
            else:
                niterations = [3]  
            for cellsize in cellsizes:
                for smooth in smooth_scales:
                    for boxpad in boxpads:
                        for niter in niterations:
                            itxt = f'{captxt}{ztxt}_shift_{rectxt}_randoms{randoms_factor}X_reso{cellsize}_smooth{smooth}_pad{boxpad}'
                            for convention in conventions:
                                if 'IterativeFFT' in recname:
                                    data_fn = f'cutsky_LRG_z0.800_AbacusSummit_base_c000_ph{ph:03d}_{itxt}_niter{niter}_{convention}_{betatxt}.fits'
                                    random_fn = f'cutsky_LRG_random{randoms_factor}X_ph{ph:03d}_{itxt}_niter{niter}_{convention}_{betatxt}.fits'
                                else:
                                    data_fn = f'cutsky_LRG_z0.800_AbacusSummit_base_c000_ph{ph:03d}_{itxt}_{convention}_{betatxt}.fits'
                                    random_fn = f'cutsky_LRG_random{randoms_factor}X_ph{ph:03d}_{itxt}_{convention}_{betatxt}.fits'
                                data = fitsio.read(os.path.join(data_dir, data_fn))
                                randoms = fitsio.read(os.path.join(data_dir, random_fn))
                                
                                w_fkp_data = fkp_weights(data['NZ_MAIN'])
                                w_fkp_randoms = fkp_weights(randoms['NZ_MAIN'])
                                
                                if not done_original:
                                    # realspace power
                                    distance = cosmo.comoving_radial_distance(data['Z_COSMO'])
                                    positions_data = utils.sky_to_cartesian(distance, data['RA'], data['DEC'])
                                    distance = cosmo.comoving_radial_distance(randoms['Z_COSMO'])
                                    positions_randoms = utils.sky_to_cartesian(distance, randoms['RA'], randoms['DEC'])
                                    result = CatalogFFTPower(data_positions1=positions_data, data_weights1=w_fkp_data,
                                                             randoms_positions1=positions_randoms, randoms_weights1=w_fkp_randoms,
                                                             edges=kedges, ells=ells, interlacing=interlacing, boxpad=1.2, 
                                                             nmesh=pknmesh, los=None, position_type='pos', mpicomm=mpicomm, 
                                                             dtype='f4').poles
                                    output_fn = os.path.join(output_dir, f'cutsky_LRG_{captxt}{ztxt}_ph{ph:03d}.randoms{random_factor:d}X.Pk_realspace_nmesh{pknmesh:d}.npy')
                                    result.save(output_fn)
                                    result.save_txt(output_fn.replace('.npy', '.txt'), complex=False)
                                    
                                    # redshift-space power
                                    distance = cosmo.comoving_radial_distance(data['Z'])
                                    positions_data = utils.sky_to_cartesian(distance, data['RA'], data['DEC'])
                                    distance = cosmo.comoving_radial_distance(randoms['Z'])
                                    positions_randoms = utils.sky_to_cartesian(distance, randoms['RA'], randoms['DEC'])
                                    result = CatalogFFTPower(data_positions1=positions_data, data_weights1=w_fkp_data,
                                                             randoms_positions1=positions_randoms, randoms_weights1=w_fkp_randoms,
                                                             edges=kedges, ells=ells, interlacing=interlacing, boxpad=1.2, 
                                                             nmesh=pknmesh, los=None, position_type='pos', mpicomm=mpicomm, 
                                                             dtype='f4').poles
                                    output_fn = os.path.join(output_dir, f'cutsky_LRG_{captxt}{ztxt}_ph{ph:03d}.randoms{random_factor:d}X.Pk_redshiftspace_nmesh{pknmesh:d}.npy')
                                    result.save(output_fn)
                                    result.save_txt(output_fn.replace('.npy', '.txt'), complex=False)
                                    done_original = True
                                
                                # reconstructed power
                                distance = cosmo.comoving_radial_distance(data['Z_REC'])
                                positions_data = utils.sky_to_cartesian(distance, data['RA_REC'], data['DEC_REC'])
                                distance = cosmo.comoving_radial_distance(randoms['Z'])
                                positions_randoms = utils.sky_to_cartesian(distance, randoms['RA'], randoms['DEC'])
                                if convention=='rsd':    
                                    result = CatalogFFTPower(data_positions1=positions_data, data_weights1=w_fkp_data,
                                                             randoms_positions1=positions_randoms, randoms_weights1=w_fkp_randoms,
                                                             edges=kedges, ells=ells, interlacing=interlacing, boxpad=1.2, 
                                                             nmesh=pknmesh, los=None, position_type='pos', mpicomm=mpicomm, 
                                                             dtype='f4').poles
                                else:
                                    distance = cosmo.comoving_radial_distance(randoms['Z_REC'])
                                    positions_shiftedr = utils.sky_to_cartesian(distance, randoms['RA_REC'], randoms['DEC_REC'])
                                    result = CatalogFFTPower(data_positions1=positions_data, data_weights1=w_fkp_data,
                                                             randoms_positions1=positions_randoms, randoms_weights1=w_fkp_randoms,
                                                             shifted_positions1=positions_shiftedr, shifted_weights1=w_fkp_randoms,
                                                             edges=kedges, ells=ells, interlacing=interlacing, boxpad=1.2, 
                                                             nmesh=pknmesh, los=None, position_type='pos', mpicomm=mpicomm, 
                                                             dtype='f4').poles
                                if 'IterativeFFT' in recname:
                                    output_fn = os.path.join(output_dir, f'cutsky_LRG_{captxt}{ztxt}_ph{ph:03d}.randoms{random_factor:d}X.shift_{rectxt}_randoms{randoms_factor}X_reso{cellsize}_smooth{smooth}_pad{boxpad}_niter{niter}_{convention}_{betatxt}.Pk_nmesh{pknmesh:d}.npy')
                                else:
                                    output_fn = os.path.join(output_dir, f'cutsky_LRG_{captxt}{ztxt}_ph{ph:03d}.randoms{random_factor:d}X.shift_{rectxt}_randoms{randoms_factor}X_reso{cellsize}_smooth{smooth}_pad{boxpad}_{convention}_{betatxt}.Pk_nmesh{pknmesh:d}.npy')
                                result.save(output_fn)
                                result.save_txt(output_fn.replace('.npy', '.txt'), complex=False)
                                    
                                    
                                    
                                    