import numpy as np
from pyrecon import MultiGridReconstruction, IterativeFFTReconstruction, PlaneParallelFFTReconstruction, setup_logging, utils
from astropy.table import Table
from cosmoprimo import fiducial
setup_logging()
from readmocks import *
from setparser import set_parser
parser = set_parser('recon')

args = parser.parse_args()
mocktype, tracer = args.mocktype, args.tracer.upper()
whichmocks = args.whichmocks
ph = args.ph
ncosmo_true, ncosmo_grid = args.cosmo_true, args.cosmo_grid
nzbin = args.zbin
cap = args.cap.upper() if args.cap else None

# settings for reconstruction
ReconstructionAlgorithm = MultiGridReconstruction
rec_algorithm_name= ReconstructionAlgorithm.__name__.lower()
rec_algorithm_name = rec_algorithm_name.strip('reconstruction')

pyrecon_kwargs = {'fft_engine': 'fftw',
                  'fft_plan': 'estimate',
                  'dtype': 'f4',
                 }
if mocktype == 'cubicbox':
    pyrecon_kwargs['nmesh'] = 512
elif mocktype == 'cutsky':
    nmesh = 512
    pyrecon_kwargs['cellsize'] = 2000. / nmesh


print('\nReconstrution computation:')
print(f'{mocktype.upper()} {tracer} {whichmocks.upper()} ph={ph}') 
print(f'True cosmology={ncosmo_true}, Grid cosmology={ncosmo_grid}')

if mocktype=='cubicbox':
    cb = CubicBox(tracer, ph=ph, whichmocks=whichmocks,
                    ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid)
    ofile = cb.get_ofilename('recon')
    data = cb.get_dict()
                
    # RECONSTRUCTION
    print('Begining reconstruction')
    recon = ReconstructionAlgorithm(f=cb.f_grid, bias=cb.bias, los='z',
                                    boxsize=cb.boxsize, boxcenter=cb.boxcenter,
                                    **pyrecon_kwargs)

    recon.assign_data(data['positions'])
    recon.set_density_contrast(smoothing_radius=cb.smoothing)
    recon.run()

    # data positions post-rec
    data['positions_rec'] = data['positions'] - recon.read_shifts(data['positions'], field='disp+rsd')
    data['positions_rec'] = data['positions_rec'] % cb.boxsize
    d = Table(data['positions_rec'], names=('x', 'y', 'z'))
    print('Saving displaced field: \n', ofile, '\n')
    d.write(ofile, format='fits')

    # randoms
    seeds = range(100, 2100, 100)
    for seed in seeds:
        randoms = cb.get_randoms(seed)
        
        for convention, field in zip(['recsym','reciso'], ['disp+rsd','disp']):
            randoms['positions_rec'] = randoms['positions'] - recon.read_shifts(randoms['positions'], field=field)
            randoms['positions_rec'] = randoms['positions_rec'] % cb.boxsize
            randoms_ofile = cb.get_randoms_ofilename(seed, convention)
            randoms_table = Table(randoms['positions_rec'], names=('x', 'y', 'z'))
            print('Saving shifted randoms: \n', randoms_ofile)
            randoms_table.write(randoms_ofile, format='fits')
    print('\n')

    
elif mocktype == 'cutsky':
    print(f'Redshift bin: {nzbin}, cap: {cap} \n')
    
    cs = CutSky(tracer, ph=ph, whichmocks=whichmocks, nzbin=nzbin,
                  ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid,
                  cap=cap)
    ofile = cs.get_ofilename('recon')
    data = cs.get_dict()

    cosmo_grid = fiducial.AbacusSummit(name=ncosmo_grid, engine='class')
    dtoredshift = utils.DistanceToRedshift(cosmo_grid.comoving_radial_distance)
    
    # RECONSTRUCTION
    print('Begining reconstruction')
    seeds = range(100, 2100, 100)
    randoms_x20 = cs.get_randoms(seeds=seeds)
    recon = ReconstructionAlgorithm(f=cs.f_grid, bias=cs.bias, los='local',
                                    positions=np.concatenate([randoms['positions'] for randoms in randoms_x20]),
                                    **pyrecon_kwargs)
    
    recon.assign_randoms(np.concatenate([randoms['positions'] for randoms in randoms_x20]))
    recon.assign_data(data['positions'])
    recon.set_density_contrast(smoothing_radius=cs.smoothing)
    recon.run()
    
    data['positions_rec'] = data['positions'] - recon.read_shifts(data['positions'], field='disp+rsd')
    distance, ra, dec = utils.cartesian_to_sky(data['positions_rec'])
    z = dtoredshift(distance)
    d = Table([ra, dec, z, data['nz']], names=['RA', 'DEC', 'Z', 'NZ'])
    print('Saving displaced field: \n', ofile, '\n')
    d.write(ofile, format='fits')
    
    # randoms
    for seed, randoms in zip(seeds, randoms_x20):
        for convention, field in zip(['recsym','reciso'], ['disp+rsd','disp']):
            randoms['positions_rec'] = randoms['positions'] - recon.read_shifts(randoms['positions'], field=field)
            randoms_ofile = cs.get_randoms_ofilename(seed, convention)
            
            distance, ra, dec = utils.cartesian_to_sky(randoms['positions_rec'])
            z = dtoredshift(distance)
            randoms_table = Table([ra, dec, z, randoms['nz']], names=['RA', 'DEC', 'Z', 'NZ'])
            print('Saving shifted randoms: \n', randoms_ofile)
            randoms_table.write(randoms_ofile, format='fits')
    print('\n')

    
    
