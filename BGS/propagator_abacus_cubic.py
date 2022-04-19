import sys
import asdf
import numpy as np
from pathlib import Path
from pyrecon.metrics import MeshFFTCorrelator, MeshFFTPropagator, CatalogMesh
from pypower.mesh import ArrayMesh
from pypower import setup_logging, mpi, MeshFFTPower


setup_logging()
mpicomm = mpi.COMM_WORLD
mpiroot = 0

node, phase = '000', '000'
recon_algos = ['multigrid', 'ifft', 'ifftp']
conventions = ['recsym', 'reciso']
redshift = 0.2
nmesh = 512
box_size = 2000.0
box_center = 0.0
smooth_radii = [10.0]
los = 'z'
bias = 1.63
kedges = np.arange(0.01, 0.5, 0.001)
muedges = np.linspace(-1., 1., 5)


if mpicomm.rank == mpiroot:
    # Read initial condition density field
    data_dir = '/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/ic/AbacusSummit_base_c000_ph000'
    data_fn = Path(data_dir, 'ic_dens_N576.asdf')

    with asdf.open(data_fn, lazy_load=False) as af:
        mesh_ic = af['data']['density']
        growth_table = af['header']['GrowthTable']

    # rescale to z = 0.2
    factor = growth_table[0.2] / growth_table[99.0]
    rescaled_mesh_ic = mesh_ic * factor
    rescaled_mesh_ic += 1.0
else:
    rescaled_mesh_ic = None

rescaled_mesh_ic = ArrayMesh(rescaled_mesh_ic, box_size, mpiroot=mpiroot, mpicomm=mpicomm)

for smooth_radius in smooth_radii:
    for recon_algo in recon_algos:
        for convention in conventions:

            if mpicomm.rank == mpiroot:

                # ----- read reconstructed data
                data_dir = f'local_abacus/cubic/z{redshift}/AbacusSummit_base_c{node}_ph{phase}/reconstruction'
                data_fn = Path(data_dir,
                    f'data_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                data = np.load(data_fn, allow_pickle=True).item()
                print(f'Shape of data: {len(data["Position_rec"])}')

                # ----- read reconstructed randoms
                randoms = {}
                randoms['Position'] = np.empty((0, 3))
                randoms['Position_rec'] = np.empty((0, 3))
                for i in range(1, 6):
                    randoms_dir = f'local_abacus/cubic/z{redshift}/AbacusSummit_base_c{node}_ph{phase}/reconstruction'
                    randoms_fn = Path(randoms_dir,
                        f'randoms{i}_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                    randoms_i = np.load(randoms_fn, allow_pickle=True).item()
                    randoms['Position'] = np.append(randoms['Position'], randoms_i['Position'], axis=0)
                    randoms['Position_rec'] = np.append(randoms['Position_rec'], randoms_i['Position_rec'], axis=0)
                print(f'Shape of concatenated randoms: {len(randoms["Position_rec"])}')

            else:
                data = {'Position': None, 'Position_rec': None}
                randoms = {'Position': None, 'Position_rec': None}

            # paint reconstructed positions to mesh
            mesh_pre = CatalogMesh(
                data['Position'], 
                 boxsize=box_size, boxcenter=box_center, nmesh=576, resampler='tsc',
                 interlacing=2, position_type='pos', mpicomm=mpicomm, mpiroot=mpiroot
            )

            mesh_recon = CatalogMesh(
                data['Position_rec'], shifted_positions=randoms['Position_rec'],
                 boxsize=box_size, boxcenter=box_center, nmesh=576, resampler='tsc',
                 interlacing=2, position_type='pos', mpicomm=mpicomm, mpiroot=mpiroot
            )

            # compute correlator/propagator
            correlator = MeshFFTCorrelator(mesh_recon, rescaled_mesh_ic, edges=(kedges, muedges), los=los)

            propagator = correlator.to_propagator(growth=bias)
            transfer = correlator.to_transfer(growth=bias)

            output_dir = f'local_abacus/cubic/z{redshift}/AbacusSummit_base_c{node}_ph{phase}/reconstruction/propagator/'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_fn = Path(output_dir,
                f'propagator_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
            propagator.save(output_fn)

