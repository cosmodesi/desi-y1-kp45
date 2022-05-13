import os
import logging
import numpy as np
import fitsio
import pyrecon
from cosmoprimo.fiducial import DESI

# ---- settings for AbacusSummit periodic box LRG mocks ---- #
boxsize = 2000  # size of AbacusSummit boxes in Mpc/h
boxcenter = boxsize / 2 # galaxy positions lie in [0, boxsize) in each direction
offset = boxcenter - boxsize / 2
cols = {'data': ['x', 'y', 'z', 'vx', 'vy', 'vz'],
        'randoms': ['x', 'y', 'z']}
z = 0.8     # redshift of the snapshot from which the data are taken - in this case snapshot 20 at z=0.800
bias = 2.35 # placeholder value that shouldn't be too far off?
data_dir = '/global/cfs/cdirs/desi/users/nadathur/Y1MockChallenge/LRG/CubicBox/AbacusSummit_base_c000/recon_catalogues/'
# ---------------------------------------------------------- #

for ph in range(1):
    
    base_dir = f'/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CubicBox/LRG/z0.800/AbacusSummit_base_c000_ph{ph:03d}/'
    input_files = {'data': os.path.join(base_dir, f'LRG_snap20_ph{ph:03d}.gcat.fits'),
                   'randoms': os.path.join(data_dir.replace('recon_catalogues/', ''), 'LRG_snap20_randoms_20x.fits')
                   }
    output_files = {'data': input_files['data'],
                   'randoms': os.path.join(data_dir, f'LRG_snap20_randoms_20x_ph{ph:03d}.fits')
                   }

    # ----- options for recon runs ------ #
    if ph == 0:
        recon_types = ['MultiGridReconstruction', 'IterativeFFTReconstruction', 'IterativeFFTParticleReconstruction']
        smooth_scales = [10, 15, 20]
        mesh_sizes = [1024, 512, 256]
        conventions = ['recsym', 'reciso', 'rsd']
    else:
        recon_types = ['MultiGridReconstruction']
        smooth_scales = [10]
        mesh_sizes = [512]
        conventions = ['recsym']    
    los = 'z' # axis to be the line-of-sight direction (plane-parallel approx)
    nthreads = int(os.environ.get('OMP_NUM_THREADS'))
    # ----------------------------------- #

    pyrecon.setup_logging()
    cosmo = DESI()

    # read original data and randoms
    print(f'Loading data realisation {ph:03d}')
    catalog = {'data': None, 'randoms': None}
    tmp = fitsio.read(input_files['randoms'])
    catalog['randoms'] = {field:tmp[field] for field in cols['randoms']}
    for sb in range(64):
        tmp = fitsio.read(input_files['data'].replace('.fits', f'.sub{sb}.fits'))
        if sb == 0:
            catalog['data'] = {field:tmp[field] for field in cols['data']}
        else:
            for field in cols['data']:
                catalog['data'][field] = np.append(catalog['data'][field], tmp[field])
    fitsio.write(os.path.join(data_dir, input_files['data']).replace('recon_', 'pre-recon_'), catalog, clobber=True)
                 
    # change the randoms filename
    # apply RSD to galaxy positions
    catalog['data'][los] = catalog['data'][los] + catalog['data'][f'v{los}'] * (1 + z) / (100 * cosmo.efunc(z))
    # rewrap positions back into the box
    catalog['data'][los] = (catalog['data'][los] - offset) % boxsize + offset

    # create positions arrays to pass to recon code
    positions = {'data': None, 'randoms': None}
    for name in ['data', 'randoms']:
        positions[name] = np.array([catalog[name][col] for col in 'xyz']).T

    # ----- loop over all options at runtime ----- #
    for recname in recon_types:
        if 'IterativeFFT' in recname and ph == 0:
            niterations = [3, 5, 7]
        else:
            niterations = [3]  
        for nmesh in mesh_sizes:
            for smooth in smooth_scales:
                for niter in niterations:
                    # run reconstruction
                    config = {'recon': {'f': cosmo.growth_rate(z), 'bias': bias, 'los': los},
                              'mesh': {'nmesh': nmesh, 'boxsize': boxsize, 'boxcenter': boxcenter,
                                       'dtype': 'f4', 'wrap': True},
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

                        # reimpose PBC on the shifted positions
                        for name in ['data', 'randoms']:
                            if positions_rec[name] is not None:
                                positions_rec[name] = (positions_rec[name] - offset) % boxsize + offset

                        # write shifted catalogues to file
                        for name in ['data', 'randoms']:
                            if 'IterativeFFT' in recname:
                                txt = f'_shift_{recname.replace("Reconstruction", "")}_niter{niter}_mesh{nmesh}_smooth{smooth}_{convention}'
                            else:
                                txt = f'_shift_{recname.replace("Reconstruction", "")}_mesh{nmesh}_smooth{smooth}_{convention}'
                            if positions_rec[name] is not None:
                                output = {}
                                for icol, col in enumerate('xyz'):
                                    output[col] = positions_rec[name][:, icol]

                                output_fn = output_files[name].replace(base_dir, data_dir)
                                output_fn = output_fn.replace('.fits', f'{txt}_f{cosmo.growth_rate(z):0.3f}_b{bias:0.2f}.fits')
                                fitsio.write(output_fn, output, clobber=True)
