import os
import logging
import numpy as np
import fitsio
import asdf
from pyrecon.metrics import MeshFFTCorrelator, MeshFFTPropagator, CatalogMesh
from pypower import ArrayMesh, setup_logging, mpi, CatalogFFTPower
from cosmoprimo.fiducial import DESI

# ---- settings for AbacusSummit periodic box LRG mocks ---- #
boxsize = 2000  # size of AbacusSummit boxes in Mpc/h
boxcenter = boxsize / 2
offset = boxcenter - boxsize / 2
z = 0.8     # redshift of the snapshot from which the data are taken - in this case snapshot 20 at z=0.800
bias = 2.35 # placeholder value that shouldn't be too far off?
# ---------------------------------------------------------- #

# --- options for pypower calculation --- #
interlacing = 2
resampling = 'tsc'
pknmesh = 512
# --------------------------------------- #

mpicomm = mpi.COMM_WORLD
mpiroot = 0
setup_logging()
cosmo = DESI()
f = cosmo.growth_rate(z)
kedges = {'step':0.001}
muedges = np.linspace(-1, 1, 20)

# load a galaxy catalogue distributed over all MPI processes
def load_data_mpi(fn, columns=('x', 'y', 'z'), ext=1):
    if not os.path.isfile(fn):
        raise Exception(f'File {fn} does not exist!')
    gsize = fitsio.FITS(fn)[ext].get_nrows()
    start, stop = mpicomm.rank * gsize // mpicomm.size, (mpicomm.rank + 1) * gsize // mpicomm.size
    tmp = fitsio.read(fn, ext=ext, columns=columns, rows=range(start, stop))
    return np.array([tmp[col] for col in columns]).T

for ph in range(1, 25):
    print(f'Realisation {ph:03d}')
    
    # --- paths --- #
    data_dir = f'/global/cfs/cdirs/desi/users/nadathur/Y1MockChallenge/LRG/CubicBox/AbacusSummit_base_c000/'
    recon_dir = os.path.join(data_dir, 'recon_catalogues/')
    ic_file = f'/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/AbacusSummit_base_c000_ph{ph:03d}/ic_dens_N576.asdf'
    input_files = {'data': f'LRG_snap20_ph{ph:03d}.gcat.fits',
                   'randoms': 'LRG_snap20_randoms_20x.fits'
                   }
    # ------------- #
    
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

    # load initial condition data as mpiroot
    if mpicomm.rank == mpiroot:
        with asdf.open(ic_file, lazy_load=False) as af:
            mesh_init = af['data']['density']
    else:
        mesh_init = None
    mesh_init = ArrayMesh(mesh_init, boxsize, mpiroot=mpiroot, mpicomm=mpicomm)

    # separately load the growth table data – I think it doesn't matter if this is spread over MPI processes
    with asdf.open(ic_file, lazy_load=False) as af:
        growth_table = af['header']['GrowthTable']
    redshift, Dz = [], []
    for key in growth_table.keys():
        redshift.append(key)
        Dz.append(growth_table[key])

    # rescale the IC mesh
    mesh_init = 1 + mesh_init * bias * np.interp(z, redshift, Dz)

    # --- load the original pre-recon data for this box --- #
    print(f"Loading data from {os.path.join(data_dir, 'pre-recon_catalogues/', input_files['data'])}")
    positions_real = load_data_mpi(os.path.join(data_dir, 'pre-recon_catalogues', input_files['data']), columns=('x', 'y', 'z', f'v{los}'))
    # apply RSD
    positions_redshift = np.copy(positions_real)
    icol = np.where(los==np.array(['x', 'y', 'z']))[0][0]
    positions_redshift[:, icol] += positions_redshift[:, 3] * (1 + z) / (100 * cosmo.efunc(z))
    positions_redshift[:, icol] = (positions_redshift[:, icol] - offset) % boxsize + offset
    # ----------------------------------------------------- #
    
    # ----- loop over all options at runtime ----- #
    for recname in recon_types:
        if 'IterativeFFT' in recname and ph == 0:
            niterations = [3, 5, 7]
        else:
            niterations = [3]  
        for nmesh in mesh_sizes:
            for smooth in smooth_scales:
                for niter in niterations:
                    for convention in conventions:
                        if 'IterativeFFT' in recname:
                            txt = f'_shift_{recname.replace("Reconstruction","")}_niter{niter}_mesh{nmesh}_smooth{smooth}_{convention}'
                        else:
                            txt = f'_shift_{recname.replace("Reconstruction", "")}_mesh{nmesh}_smooth{smooth}_{convention}'
                        
                        positions_rec = {'data': None, 'randoms': None}    
                        shifted_file = {'data': os.path.join(data_dir, 'recon_catalogues', input_files['data'].replace('.fits', f'{txt}_f{f:0.3f}_b{bias:0.2f}.fits')), 'randoms': None}
                        positions_rec['data'] = load_data_mpi(shifted_file['data'])
                        
                        # ----- compute the propagator ------ #
                        if convention == 'rsd':
                            # for the propagator the "initial" mesh is actually from the real-space field
                            mesh_real = CatalogMesh(positions_real[:, :3], boxsize=boxsize, boxcenter=boxcenter, 
                                                    nmesh=576, resampler=resampling, interlacing=interlacing, position_type='pos', 
                                                    mpicomm=mpicomm, mpiroot=None)
                            # no randoms as they didn't get shifted and selection is uniform
                            mesh_recon = CatalogMesh(positions_rec['data'], boxsize=boxsize, boxcenter=boxcenter, 
                                                     nmesh=576, resampler=resampling, interlacing=interlacing, position_type='pos', 
                                                     mpicomm=mpicomm, mpiroot=None)
                            correlator = MeshFFTCorrelator(mesh_recon, mesh_real, edges=(kedges, muedges), los=los)
                            propagator = correlator.to_propagator(growth=1)
                            
                        elif convention in ['recsym', 'reciso']:
                            shifted_file['randoms'] = os.path.join(recon_dir, input_files['randoms'].replace('.fits', f'_ph{ph:03d}{txt}_f{f:0.3f}_b{bias:0.2f}.fits'))
                            positions_rec['randoms'] = load_data_mpi(shifted_file['randoms'])

                            # selection is uniform but randoms got shifted during recon
                            mesh_recon = CatalogMesh(positions_rec['data'], shifted_positions=positions_rec['randoms'],
                                                     boxsize=boxsize, boxcenter=boxcenter, nmesh=576, resampler=resampling,
                                                     interlacing=interlacing, position_type='pos', mpicomm=mpicomm, mpiroot=None)
                            # for the propagator we use the real initial conditions
                            correlator = MeshFFTCorrelator(mesh_recon, mesh_init, edges=(kedges, muedges), los=los)
                            propagator = correlator.to_propagator(growth=1)

                        if mpicomm.rank == mpiroot:
                            output_fn = os.path.join(data_dir, 'propagators', input_files['data'].replace('.fits', f'{txt}_f{f:0.3f}_b{bias:0.2f}.propagator.npy'))
                            propagator.save(output_fn)
                            #propagator.save_txt(output_fn.replace('.npy', '.txt'))
                        # --------------------------------- #

                        # ----- compute the power spectrum ------ #
                        if convention == 'rsd':
                            result = CatalogFFTPower(data_positions1=positions_rec['data'], nmesh=pknmesh, boxsize=boxsize,
                                                     boxcenter=boxcenter, resampler=resampling, interlacing=interlacing, 
                                                     ells=(0, 2, 4), los=los, edges={'step': 0.001}, 
                                                     position_type='pos', wrap=True, mpicomm=mpicomm, dtype='f4').poles

                        elif convention in ['recsym', 'reciso']:
                            result = CatalogFFTPower(data_positions1=positions_rec['data'], 
                                                     shifted_positions1=positions_rec['randoms'],
                                                     nmesh=pknmesh, boxsize=boxsize, boxcenter=boxcenter, resampler=resampling, 
                                                     interlacing=interlacing, ells=(0, 2, 4), los=los, edges={'step': 0.001}, 
                                                     position_type='pos', mpicomm=mpicomm, dtype='f4').poles

                        output_fn = os.path.join(data_dir, 'Pk', input_files['data'].replace('.fits', f'{txt}_f{f:0.3f}_b{bias:0.2f}.randoms_20X.Pk_nmesh{pknmesh:d}.npy'))
                        result.save(output_fn)
                        result.save_txt(output_fn.replace('.npy', '.txt'), complex=False)
                        # -------------------------------------- #

    # ---- finally, calculate the original pre-recon power spectra in real and redshift space ---- #
    result = CatalogFFTPower(data_positions1=positions_real[:, :3], nmesh=pknmesh, boxsize=boxsize, boxcenter=boxcenter, 
                             resampler=resampling, interlacing=interlacing, ells=(0, 2, 4), los=los, edges={'step': 0.001}, 
                             position_type='pos', mpicomm=mpicomm, dtype='f4').poles
    pk_file = os.path.join(data_dir, 'Pk', input_files['data'].replace('.fits', f'.randoms_20X.realspace_Pk_nmesh{pknmesh:d}.npy'))
    result.save(pk_file)
    result.save_txt(pk_file.replace('.npy', '.txt'), complex=False)

    result = CatalogFFTPower(data_positions1=positions_redshift[:, :3], nmesh=pknmesh, boxsize=boxsize, boxcenter=boxcenter, 
                             resampler=resampling, interlacing=interlacing, ells=(0, 2, 4), los=los, edges={'step': 0.001},  
                             position_type='pos', mpicomm=mpicomm, dtype='f4').poles
    pk_file = os.path.join(data_dir, 'Pk', input_files['data'].replace('.fits', f'.randoms_20X.redshiftspace_Pk_nmesh{pknmesh:d}.npy'))
    result.save(pk_file)
    result.save_txt(pk_file.replace('.npy', '.txt'), complex=False)