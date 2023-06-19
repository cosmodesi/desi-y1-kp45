import os, sys
import fitsio
import numpy as np
from cosmoprimo import fiducial
from pycorr import TwoPointCorrelationFunction, setup_logging, mpi
from astropy.table import Table
setup_logging()
import argparse
from readmocks import *
from setparser import set_parser
parser = set_parser()

args = parser.parse_args()
mocktype, tracer = args.mocktype, args.tracer.upper()
whichmocks = args.whichmocks
ph = f'{args.ph:03d}'
rectype = args.rectype
ncosmo_true, ncosmo_grid = args.cosmo_true, args.cosmo_grid
if args.zbin:
    nzbin=args.zbin
    
# edges for smu type 2PCF

pycorr_kwargs = {'edges': ((np.linspace(0, 200, 201),
                            np.linspace(-1, 1, 241))),
                 'mpicomm': mpicomm,
                 'mpiroot': mpiroot,
                 'nthreads': threads,
                }

print0('\nTwo point correlation function computation:')
reclabel=rectype.upper() if rectype else 'PRE-REC'
print0(f'{mocktype.upper()} {tracer} {whichmocks.upper()} {ph} {reclabel}') 
print0(f'True cosmology={ncosmo_true}, Grid cosmology={ncosmo_grid}')


if mocktype=='cubicbox':
    data = CubicBox(tracer, ph=ph, whichmocks=whichmocks, rectype=rectype,
                    ncosmo_true=ncosmo_true, ncosmo_grid=ncosmo_grid)
    ofile = data.get_ofilename('xi')
    
    if not rectype:
        # Compute 2PCF with the natural estimator
        result = TwoPointCorrelationFunction(mode='smu', data_positions1=data.positions,
                                             boxsize=data.boxsize, los='z', position_type='xyz',
                                             **pycorr_kwargs)
        
    else:
        # Calculate 2PCF with LS estimator and by summing S1S2 counts for each random catalog
        shifted_list = data.get_randoms(shifted=True)
        D1D2 = None
        result = 0
        for j, shifted_positions in enumerate(shifted_list):
            print0(f'Split {j+1}/{len(shifted_list)}')

            result += TwoPointCorrelationFunction(mode='smu', data_positions1=data.positions,
                                                  shifted_positions1=shifted_positions,
                                                  estimator='landyszalay', boxsize=data.boxsize,
                                                  los='z', position_type='xyz', D1D2=D1D2,
                                                  **pycorr_kwargs)
            D1D2 = result.D1D2
    
elif mocktype=='cutsky':
    raise NotImplementedError


result.save(ofile)
