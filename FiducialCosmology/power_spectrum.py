import os, sys
import fitsio
import numpy as np
from cosmoprimo import fiducial
from pypower import CatalogMesh, MeshFFTPower, CatalogFFTPower, PowerSpectrumStatistics, utils, setup_logging, mpi
setup_logging()
from readmocks import *
from setparser import set_parser
parser = set_parser()

args = parser.parse_args()
mocktype, tracer = args.mocktype, args.tracer.upper()
whichmocks = args.whichmocks
ph, rectype = args.ph, args.rectype
ncosmo_true, ncosmo_grid = args.cosmo_true, args.cosmo_grid
nzbin = args.zbin
cap = args.cap

    
pypower_kwargs = {'resampler': 'tsc',
                  'interlacing': 2,
                  'ells': (0, 2, 4),
                  'edges': {'step': 0.001},
                  'mpicomm': mpicomm,
                  'mpiroot': mpiroot,
                  'dtype': 'f4',
                 }
if mocktype == 'cubicbox':
    pypower_kwargs['nmesh'] = 512
elif mocktype == 'cutsky':
    pypower_kwargs['nmesh'] = 1024


print0('\nPower spectrum computation:')
reclabel=rectype.upper() if rectype else 'PRE-REC'
print0(f'{mocktype.upper()} {tracer} {whichmocks.upper()} {ph} {reclabel}') 
print0(f'True cosmology={ncosmo_true}, Grid cosmology={ncosmo_grid}')

if mocktype=='cubicbox':
    cb = CubicBox(tracer, ph=ph, whichmocks=whichmocks, rectype=rectype,
                  ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid)
    ofile = cb.get_ofilename('pk')
    data = cb.get_dict()
    shifted = cb.get_randoms(shifted=True, concat=True) if rectype else None

    poles = CatalogFFTPower(data_positions1=data['positions'], shifted_positions1=shifted['positions'],
                            boxsize=cb.boxsize, boxcenter=cb.boxcenter, los='z', position_type='pos', 
                            **pypower_kwargs).poles
    
elif mocktype=='cutsky':
    cs = CutSky(tracer, ph=ph, whichmocks=whichmocks, rectype=rectype,
                nzbin=nzbin, cap=cap, ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid)
    ofile = cs.get_ofilename('pk')
    data = cs.get_dict()
    randoms = cs.get_randoms(shifted=False, concat=True)
   
    if rectype: 
        shifted = cs.get_randoms(shifted=True, concat=True)
        poles = CatalogFFTPower(data_positions1=data['positions'], data_weights1=w_fkp(data['nz']),
                                randoms_positions1=randoms['positions'], randoms_weights1=w_fkp(randoms['nz']), 
                                shifted_positions1=shifted['positions'], shifted_weights1=w_fkp(shifted['nz']),
                                position_type='pos', **pypower_kwargs).poles
    else:
        poles = CatalogFFTPower(data_positions1=data['positions'], data_weights1=w_fkp(data['nz']),
                                randoms_positions1=randoms['positions'], randoms_weights1=w_fkp(randoms['nz']), 
                                position_type='pos', **pypower_kwargs).poles
    
poles.save(ofile)
