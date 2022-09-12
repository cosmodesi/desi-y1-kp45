import numpy as np
import json
import sys
import os
from mpi4py import MPI
import time

from compute_fid_dists import compute_fid_dists
from taylor_approximation import compute_derivatives
from make_pkclass import make_pkclass

tic = time.perf_counter()

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
#print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] +'/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])

# Compute fiducial distances
fid_dists = compute_fid_dists(z,Omfid)

#Create template pkclass:
pkclass = make_pkclass(z)

# Set up the output k vector:
from compute_pell_tables import compute_pell_tables, kvec

output_shape = (len(kvec),19) # two multipoles and 19 types of terms

# First construct the grid

order = 4
# these are fsigma8,apar,aperp,m
x0s = [0.46, 1.0,1.0,0.0]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.01, 0.005,0.005,0.01]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P2grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P4grid = np.zeros( (Npoints,)*Nparams+ output_shape)

P0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P4gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

toc1 = time.perf_counter() 
print('time = {}'.format(toc1-tic))

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        p0, p2, p4 = compute_pell_tables(coord,pkclass,z=z,fid_dists=fid_dists)
        
        P0gridii[iis] = p0
        P2gridii[iis] = p2
        P4gridii[iis] = p4
        
toc2 = time.perf_counter() 
print('grid time = {}'.format(toc2-toc1))
        
comm.Allreduce(P0gridii, P0grid, op=MPI.SUM)
comm.Allreduce(P2gridii, P2grid, op=MPI.SUM)
comm.Allreduce(P4gridii, P4grid, op=MPI.SUM)

del(P0gridii, P2gridii, P4gridii)

toc3 = time.perf_counter() 
print('total time = {}'.format(toc3-tic))

# Now compute the derivatives
derivs0 = compute_derivatives(P0grid, dxs, center_ii, 3)
derivs2 = compute_derivatives(P2grid, dxs, center_ii, 3)
derivs4 = compute_derivatives(P4grid, dxs, center_ii, 3)

toc4 = time.perf_counter() 
print('All derivs time = {}'.format(toc4-toc3))
print('Total time = {}'.format(toc4-tic))


if mpi_rank == 0:
    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir+'emu'
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    #
comm.Barrier()

# Now save:
outfile = basedir + 'emu/boss_z_%.2f_pkells.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]
list4 = [ dd.tolist() for dd in derivs4 ]

outdict = {'params': ['f_sig8','apar','aperp','m'],\
           'x0': x0s,\
           'kvec': kvec.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,\
           'derivs4': list4}

if mpi_rank == 0:
    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()
