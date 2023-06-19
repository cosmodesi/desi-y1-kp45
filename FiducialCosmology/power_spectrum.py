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
if args.zbin:
    nzbin=args.zbin

    
pypower_kwargs = {'nmesh': 512,
                  'resampler': 'tsc',
                  'interlacing': 2,
                  'ells': (0, 2, 4),
                  'edges': {'step': 0.001},
                  'mpicomm': mpicomm,
                  'mpiroot': mpiroot,
                  'dtype': 'f4',
                 }


print0('\nPower spectrum computation:')
reclabel=rectype.upper() if rectype else 'PRE-REC'
print0(f'{mocktype.upper()} {tracer} {whichmocks.upper()} {ph} {reclabel}') 
print0(f'True cosmology={ncosmo_true}, Grid cosmology={ncosmo_grid}')

if mocktype=='cubicbox':
    data = CubicBox(tracer, ph=ph, whichmocks=whichmocks, rectype=rectype,
                    ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid)
    ofile = data.get_ofilename('pk')
    shifted_positions = data.get_randoms(shifted=True, concat=True) if rectype else None

    poles = CatalogFFTPower(data_positions1=data.positions, shifted_positions1=shifted_positions,
                            boxsize=data.boxsize, boxcenter=data.boxcenter, los='z', position_type='xyz', 
                            **pypower_kwargs).poles
    
elif mocktype=='cutsky':
    raise NotImplementedError
    
poles.save(ofile)