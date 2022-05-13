import os
import numpy as np
import logging
import fitsio
import argparse
import pyrecon
from pyrecon import utils
from cosmoprimo.fiducial import DESI

def mask(main=0, nz=0, Y5=0, sv3=0):
    """
    Apply desired cuts to mocks
    """
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)

logger = logging.getLogger('Main')
pyrecon.setup_logging()
cosmo = DESI()
data_dir = '/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CutSky/LRG/z0.800/'
output_dir = '/global/cfs/cdirs/desi/users/nadathur/Y1MockChallenge/LRG/CutSky/'

# --- settings for Abacus cut-sky mocks --- #
randoms_factor = 20 # how many randoms files to concatenate, maximum 50
bias = 2.35 # estimated value that shouldn't be too far off? can check other values too
columns = ['RA', 'DEC', 'Z', 'Z_COSMO', 'STATUS', 'NZ_MAIN']
zranges = np.array([{'min': 0.6, 'max': 0.8}])#, {'min': 0.8, 'max': 1.1}, {'min': 0.4, 'max': 0.6}, ]) 
cap = 'ngc' # 'ngc', 'sgc' or 'both' (default) 
mask_y5 = mask(main=1, nz=0, Y5=1, sv3=0)
# ----------------------------------------- #

for ph in range(25):
    
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
    # ----------------------------------- #

    orig_input_files = {'data': f'cutsky_LRG_z0.800_AbacusSummit_base_c000_ph{ph:03d}.fits',
                        'randoms': 'cutsky_LRG_random_S{:d}_1X.fits'
                       }
    for zr in zranges:
        ztext = f'{zr["min"]:0.1f}z{zr["max"]:0.1f}'
        zmid = 0.5 * (zr['min'] + zr['max']) # approximation to maybe be improved later?        
        if cap=='ngc' or cap=='sgc':
            cut_input_files = {'data': f'cutsky_LRG_{cap.upper()}_z0.800_AbacusSummit_base_c000_ph{ph:03d}_{ztext}.fits',
                               'randoms': f'cutsky_LRG_{cap.upper()}_random_20X_{ztext}.fits'
                              }
        else:
            cut_input_files = {'data': f'cutsky_LRG_z0.800_AbacusSummit_base_c000_ph{ph:03d}_{ztext}.fits',
                               'randoms': f'cutsky_LRG_random_20X_{ztext}.fits'
                              }
        for name in ['data', 'randoms']:
            cut_input_files[name] = os.path.join(output_dir, 'pre-recon_catalogues', cut_input_files[name])
            
        cut_catalog = {'data': None, 'randoms': None}
        catalog = {'data': None, 'randoms': None}
        if os.path.isfile(cut_input_files['data']):
            logger.info(f'Loading data file {cut_input_files["data"]}')
            tmp = fitsio.read(cut_input_files['data'])
            cut_catalog['data'] = {field:tmp[field] for field in columns}
        else:
            logger.info(f'Pre-cut file {cut_input_files["data"]} does not exist ...')
            logger.info(f'Loading data file {os.path.join(data_dir, orig_input_files["data"])}')
            tmp = fitsio.read(os.path.join(data_dir, orig_input_files["data"]))
            status = tmp['STATUS']
            idx = np.arange(len(status))
            idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
            catalog['data'] = {field:tmp[field][idx_Y5] for field in columns}
            
        if os.path.isfile(cut_input_files['randoms']):
            logger.info(f'Loading randoms file {cut_input_files["randoms"]}')
            tmp = fitsio.read(cut_input_files['randoms'])
            cut_catalog['randoms'] = {field:tmp[field] for field in columns}
        else:
            logger.info(f'Pre-cut file {cut_input_files["randoms"]} does not exist ...')
            # concatenate multiple randoms files
            for i in range(1, randoms_factor + 1): 
                rfile = os.path.join(data_dir, orig_input_files['randoms'].format(100*i))
                logger.info(f'Loading randoms file {rfile}')
                tmp = fitsio.read(rfile)
                status = tmp['STATUS']
                idx = np.arange(len(status))
                idx_Y5 = idx[((status & (mask_y5))==mask_y5)]
                if i == 1:
                    catalog['randoms'] = {field:tmp[field][idx_Y5] for field in columns}
                else:
                    for field in columns:
                        catalog['randoms'][field] = np.append(catalog['randoms'][field], tmp[field][idx_Y5])
        
        positions = {'data': None, 'randoms': None}
        for name in ['data', 'randoms']:
            if cut_catalog[name] is None:
                logger.info(f'Cutting {name} catalogues to redshift range {zr["min"]:0.1f}<z<{zr["max"]:0.1f}')
                # cut the data and random catalogues to the desired selection and redshift range  
                idx = np.arange(len(catalog[name]['Z']))
                if cap=='ngc':
                    idx_cut = idx[(catalog[name]['Z']>zr['min'])&(catalog[name]['Z']<zr['max'])&\
                                  (catalog[name]['RA']>88)&(catalog[name]['RA']<303)]
                elif cap=='sgc':
                    idx_cut = idx[(catalog[name]['Z']>zr['min'])&(catalog[name]['Z']<zr['max'])&\
                                  (catalog[name]['RA']<88)&(catalog[name]['RA']>303)]
                else:
                    idx_cut = idx[(catalog[name]['Z']>zr['min'])&(catalog[name]['Z']<zr['max'])] 
                cut_catalog[name] = {field:catalog[name][field][idx_cut] for field in columns}

                # write to file for later use
                logger.info('Saving cut pre-recon catalogue')
                outfn = os.path.join(output_dir, 'pre-recon_catalogues', orig_input_files[name])
                if name == 'randoms':
                    outfn = outfn.replace('S{:d}_1X.fits', f'{randoms_factor:d}X.fits')
                outfn = outfn.replace('.fits', f'_{zr["min"]:0.1f}z{zr["max"]:0.1f}.fits')
                if cap=='ngc' or cap=='sgc':
                    outfn = outfn.replace('LRG_', f'LRG_{cap.upper()}_')
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
                                        txt = f'_{zr["min"]:0.1f}z{zr["max"]:0.1f}_shift_{recname.replace("Reconstruction", "")}_randoms{randoms_factor:d}X_reso{cellsize}_smooth{smooth}_pad{boxpad}_niter{niter}_{convention}'
                                    else:
                                        txt = f'_{zr["min"]:0.1f}z{zr["max"]:0.1f}_shift_{recname.replace("Reconstruction", "")}_randoms{randoms_factor:d}X_reso{cellsize}_smooth{smooth}_pad{boxpad}_{convention}'
                                    if cap=='ngc':
                                        txt = f'_NGC{txt}'
                                    elif cap=='sgc':
                                        txt = f'_SGC{txt}'
                                    if positions_rec[name] is not None:
                                        output_fn = os.path.join(output_dir, 'recon_catalogues', orig_input_files[name])
                                        output_fn = output_fn.replace('.fits', f'{txt}_f{cosmo.growth_rate(zmid):0.3f}_b{bias:0.2f}.fits')
                                        output_fn = output_fn.replace('_S{:d}_1X', f'{randoms_factor:d}X_ph{ph:03d}')
                                        fitsio.write(output_fn, cut_catalog[name], clobber=True)
