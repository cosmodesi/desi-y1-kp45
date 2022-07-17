import os
import numpy as np
import logging
import fitsio
import healpy as hp
import h5py
import argparse
import pyrecon
from pyrecon import utils
from cosmoprimo.fiducial import DESI
from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline


def mask(main=0, nz=0, Y5=0, sv3=0):
    """
    Apply desired cuts to mocks
    """
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

def spl_nofz(zarray, fsky, cosmo, zmin, zmax, Nzbins=100):
    
    zbins = np.linspace(zmin, zmax, Nzbins+1)
    Nz, zbins = np.histogram(zarray, zbins)
    
    zmid = zbins[0:-1] + (zmax-zmin)/Nzbins/2.0
    # set z range boundaries to be zmin and zmax and avoid the interpolation error
    zmid[0], zmid[-1] = zbins[0], zbins[-1]   
    
    rmin = cosmo.comoving_radial_distance(zbins[0:-1])
    rmax = cosmo.comoving_radial_distance(zbins[1:])
    
    vol = fsky * 4./3*np.pi * (rmax**3.0 - rmin**3.0)
    nz_array = Nz/vol
    
    spl_nz = InterpolatedUnivariateSpline(zmid, nz_array)
    
    return spl_nz


logger = logging.getLogger('Main')
pyrecon.setup_logging()
cosmo = DESI()
data_dir = '/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CutSky/BGS/z0.200/'
##output_dir = '/global/cfs/cdirs/desi/users/nadathur/Y1MockChallenge/LRG/CutSky/'
output_dir = '/global/cfs/cdirs/desi/users/jerryou/MockChallenge/y1_mockchallenge/reconstruction/AbacusSummit_base_c000_ph000/cutsky/BGS/'

# --- settings for Abacus cut-sky mocks --- #
##randoms_factor = 20 # how many randoms files to concatenate, maximum 50
randoms_factor = 15
bias = 1.63 # estimated value that shouldn't be too far off? can check other values too
columns_input = ['ra', 'dec', 'z_obs', 'z_cos', 'STATUS', 'app_mag']
columns_output = ['RA', 'DEC', 'Z', 'Z_COSMO', 'STATUS', 'app_mag', 'NZ_MAIN']

zranges = np.array([{'min': 0.2, 'max': 0.4}])#, {'min': 0.8, 'max': 1.1}, {'min': 0.4, 'max': 0.6}, ]) 
cap = 'ngc' # 'ngc', 'sgc' or 'both' (default) 
app_mag_cut = 19.5
mask_y5 = mask(main=0, nz=0, Y5=1, sv3=0)      # for BGS, set main=0 based on https://desi.lbl.gov/trac/wiki/CosmoSimsWG/FirstGenerationMocks
Nside = 256
# ----------------------------------------- #

for ph in range(1):
    
    # ----- options for recon runs ------ #
    if ph == 0:
        #recon_types = ['MultiGridReconstruction', 'IterativeFFTReconstruction', 'IterativeFFTParticleReconstruction'] 
        recon_types = ['MultiGridReconstruction'] 
        ##smooth_scales = [7.5, 10, 15]
        smooth_scales = [10]
        ##conventions = ['recsym' , 'reciso', 'rsd']
        conventions = ['recsym']
    else:
        recon_types = ['MultiGridReconstruction'] 
        smooth_scales = [10]
        conventions = ['recsym']
    cellsizes = [7.8] 
    boxpads = [1.5]
    nthreads = int(os.environ.get('OMP_NUM_THREADS'))
    # ----------------------------------- #

    orig_input_files = {'data': f'cutsky_BGS_z0.200_AbacusSummit_base_c000_ph{ph:03d}.hdf5',
                        'randoms': 'random_S{:d}_1X.hdf5'
                       }
    path_cut_catalog = os.path.join(output_dir, 'pre-recon_catalogues')
    if not os.path.exists(path_cut_catalog):
        os.makedirs(path_cut_catalog)
        
    path_recon_catalog = os.path.join(output_dir, 'recon_catalogues')
    if not os.path.exists(path_recon_catalog):
        os.makedirs(path_recon_catalog)
        
    for zr in zranges:
        ztext = f'{zr["min"]:0.1f}z{zr["max"]:0.1f}'
        zmid = 0.5 * (zr['min'] + zr['max']) # approximation to maybe be improved later?        
        if cap=='ngc' or cap=='sgc':
            cut_input_files = {'data': f'cutsky_BGS_{cap.upper()}_z0.200_AbacusSummit_base_c000_ph{ph:03d}_{ztext}.fits',
                               'randoms': f'cutsky_BGS_{cap.upper()}_random_20X_{ztext}.fits'
                              }
            
        # cut_ denotes the catalog with some redshift and magnitude masks
        else:
            cut_input_files = {'data': f'cutsky_BGS_z0.200_AbacusSummit_base_c000_ph{ph:03d}_{ztext}.fits',
                               'randoms': f'cutsky_BGS_random_20X_{ztext}.fits'
                              }
        
        for name in ['data', 'randoms']:
            cut_input_files[name] = os.path.join(path_cut_catalog, cut_input_files[name])
            
        cut_catalog = {'data': None, 'randoms': None}
        catalog = {'data': None, 'randoms': None}
        
        # load the data catalog
        if os.path.isfile(cut_input_files['data']):
            logger.info(f'Loading data file {cut_input_files["data"]}')
            
            tmp = fitsio.read(cut_input_files['data'])
            cut_catalog['data'] = {field:tmp[field] for field in columns_output}
            
        else:
            logger.info(f'Pre-cut file {cut_input_files["data"]} does not exist ...')
            logger.info(f'Loading data file {os.path.join(data_dir, orig_input_files["data"])}')
            
            with h5py.File(os.path.join(data_dir, orig_input_files["data"]), 'r') as f:
                tmp = f['Data']
                
                status = tmp['STATUS'][...]
                idx = np.arange(len(status))
                idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
                catalog['data'] = {}
                for field_in, field_out in zip(columns_input, columns_output):
                    catalog['data'][field_out] = tmp[field_in][...][idx_Y5]
            
        # load the random catalog
        if os.path.isfile(cut_input_files['randoms']):
            logger.info(f'Loading randoms file {cut_input_files["randoms"]}')
            
            tmp = fitsio.read(cut_input_files['randoms'])
            cut_catalog['randoms'] = {field:tmp[field] for field in columns_output}
                
        else:
            logger.info(f'Pre-cut file {cut_input_files["randoms"]} does not exist ...')
            # concatenate multiple randoms files
            for i in range(1, randoms_factor + 1):  
                rfile = os.path.join(data_dir, orig_input_files['randoms'].format(100*i))
                logger.info(f'Loading randoms file {rfile}')
                with h5py.File(os.path.join(data_dir, orig_input_files["randoms"].format(100*i)), 'r') as f:
                    tmp = f['Data']
                    status = tmp['STATUS'][...]
                    idx = np.arange(len(status))
                    idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
                    if i == 1:
                        catalog['randoms'] = {}
                        for field_in, field_out in zip(columns_input, columns_output[:-1]):
                            catalog['randoms'][field_out] = tmp[field_in][...][idx_Y5]
                    
                    else:
                        for field_in, field_out in zip(columns_input, columns_output[:-1]):
                            catalog['randoms'][field_out] = np.append(catalog['randoms'][field_out], tmp[field_in][...][idx_Y5])
        
        positions = {'data': None, 'randoms': None}
        for name in ['data', 'randoms']:
            # if cut_catalog does not exist, we generate it from the raw catalog
            if cut_catalog[name] is None: 
                
                logger.info(f'Cutting {name} catalogues to redshift range {zr["min"]:0.1f}<z<{zr["max"]:0.1f}')
                logger.info(f'Cutting {name} catalogues with the apparent magnitude < {app_mag_cut}')
                # cut the data and random catalogues to the desired selection and redshift range  
                idx = np.arange(len(catalog[name]['Z']))


                if cap=='ngc':
                    idx_cut = idx[(catalog[name]['Z']>zr['min'])&(catalog[name]['Z']<zr['max'])&\
                                  (catalog[name]['RA']>88)&(catalog[name]['RA']<303)&\
                                  (catalog[name]['app_mag']<app_mag_cut)]

                elif cap=='sgc':
                    idx_cut = idx[(catalog[name]['Z']>zr['min'])&(catalog[name]['Z']<zr['max'])&\
                                  (catalog[name]['RA']<88)&(catalog[name]['RA']>303)&\
                                  (catalog[name]['app_mag']<app_mag_cut)]
                else:
                    idx_cut = idx[(catalog[name]['Z']>zr['min'])&(catalog[name]['Z']<zr['max'])&\
                                  (catalog[name]['app_mag']<app_mag_cut)]

                cut_catalog[name] = {field:catalog[name][field][idx_cut] for field in columns_output[:-1]}

                # calculate fsky: the sky coverage percentage
                if name == 'data':   
                    Npix = hp.nside2npix(Nside)
                    phi = np.radians(cut_catalog[name]['RA'])
                    theta = np.radians(90.0 - cut_catalog[name]['DEC'])

                    pixel_indices = hp.ang2pix(Nside, theta, phi)
                    pixel_unique, counts = np.unique(pixel_indices, return_counts=True)
                    fsky = len(pixel_unique)/Npix   # fsky
                    logger.info(f'fsky of {name} catalogues with {cap.upper()} is {fsky}')

                    
                    spl_nz = spl_nofz(cut_catalog[name]['Z'], fsky, cosmo, zr['min'], zr['max'])
                    
                logger.info(f'Add {name} catalogues with the column NZ_MAIN based on the galaxy number density n(z)')
                cut_catalog[name]['NZ_MAIN'] = spl_nz(cut_catalog[name]['Z'])

                # write to file for later use
                logger.info('Saving the pre-recon cut_catalog with masks')

                outfn = os.path.join(cut_input_files[name])

                fitsio.write(outfn, cut_catalog[name], clobber=True)
                
                

            # convert to Cartesian coordinates
            logger.info('Converting to Cartesian positions')
            distance = cosmo.comoving_radial_distance(cut_catalog[name]['Z'])
            positions[name] = utils.sky_to_cartesian(distance, cut_catalog[name]['RA'], cut_catalog[name]['DEC'])
    
            
                    
        # ----- now loop over all reconstruction options ----- #
        for recname in recon_types:
            if 'IterativeFFT' in recname and ph == 0:
                niterations = [3, 5, 7]
            else:
                niterations = [3]  
            for cellsize in cellsizes:
                for smooth in smooth_scales:
                    for boxpad in boxpads:
                        for niter in niterations:
                            # run reconstruction
                            config = {'recon': {'f': cosmo.growth_rate(zmid), 'bias': bias, 'los': 'local'},
                                      'mesh': {'positions': positions['randoms'], 'cellsize': cellsize, 'boxpad': boxpad,
                                               'dtype': 'f4', 'wrap': False},
                                      'algorithm': {'niterations': niter}
                                      }
                            if 'IterativeFFT' in recname:
                                # add kwargs to do with FFTW behaviour – just saves creating and loading the wisdom unnecessarily
                                # when only using MultiGridReconstruction that doesn't require FFTW
                                config['mesh']['fft_engine'] = 'fftw'
                                config['mesh']['save_fft_wisdom'] = True
                            ReconstructionAlgorithm = getattr(pyrecon, recname)
                            recon = ReconstructionAlgorithm(**config['recon'], **config['mesh'], nthreads=nthreads)
                            recon.assign_data(positions['data'])
                            recon.assign_randoms(positions['randoms'])
                            recon.set_density_contrast(smoothing_radius=smooth)
                            if recname == 'MultiGridReconstruction':
                                recon.run()
                            else:
                                recon.run(**config['algorithm'])

                            for convention in conventions:
                                # obtain the shifted positions
                                positions_rec = {'data': None, 'randoms': None}
                                gfield = 'rsd' if convention == 'rsd' else 'disp+rsd'
                                if recname == 'IterativeFFTParticleReconstruction':
                                    positions_rec['data'] = positions['data'] - recon.read_shifts('data', field=gfield)
                                else:
                                    positions_rec['data'] = positions['data'] - recon.read_shifts(positions['data'], field=gfield)
                                if convention == 'recsym':
                                    positions_rec['randoms'] = positions['randoms'] - recon.read_shifts(positions['randoms'], field='disp+rsd')
                                if convention == 'reciso':
                                    positions_rec['randoms'] = positions['randoms'] - recon.read_shifts(positions['randoms'], field='disp')

                                # reconvert shifted positions to rdz format
                                for name in ['data', 'randoms']:
                                    if positions_rec[name] is not None:
                                        distance, ra, dec = utils.cartesian_to_sky(positions_rec[name])
                                        distance_to_redshift = utils.DistanceToRedshift(cosmo.comoving_radial_distance)
                                        z = distance_to_redshift(distance)
                                        for col, value in zip(['RA_REC', 'DEC_REC', 'Z_REC'], [ra, dec, z]):
                                            cut_catalog[name][col] = value

                                # write shifted catalogues to file
                                for name in ['data', 'randoms']:
                                    if 'IterativeFFT' in recname:
                                        txt = f'_{zr["min"]:0.1f}z{zr["max"]:0.1f}_mag{app_mag_cut:.2f}_shift_{recname.replace("Reconstruction", "")}_randoms{randoms_factor:d}X_reso{cellsize}_smooth{smooth}_pad{boxpad}_niter{niter}_{convention}'
                                    else:
                                        txt = f'_{zr["min"]:0.1f}z{zr["max"]:0.1f}_mag{app_mag_cut:.2f}_shift_{recname.replace("Reconstruction", "")}_randoms{randoms_factor:d}X_reso{cellsize}_smooth{smooth}_pad{boxpad}_{convention}'
                                    if cap=='ngc':
                                        txt = f'_NGC{txt}'
                                    elif cap=='sgc':
                                        txt = f'_SGC{txt}'
                                    if positions_rec[name] is not None:
                                        output_fn = os.path.join(path_recon_catalog, orig_input_files[name])
                                        output_fn = output_fn.replace('.hdf5', f'{txt}_f{cosmo.growth_rate(zmid):0.3f}_b{bias:0.2f}.fits')
                                        output_fn = output_fn.replace('_S{:d}_1X', f'{randoms_factor:d}X_ph{ph:03d}')
                                        fitsio.write(output_fn, cut_catalog[name], clobber=True)
