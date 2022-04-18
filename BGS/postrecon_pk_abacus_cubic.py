import numpy as np
from pathlib import Path
from pypower import setup_logging, mpi, CatalogFFTPower
from pyrecon.metrics import MeshFFTCorrelator, MeshFFTPropagator, CatalogMesh
from pprint import pprint

# Set up logging
setup_logging()
mpicomm = mpi.COMM_WORLD
mpiroot = 0

redshift = 0.2
box_size = 2000.0
box_center = 0
bias = 1.63
los = 'z'
node, phase = '000', '000'

nmeshs = [512]
smooth_radii = [15.0]
recon_algos = ['multigrid', 'ifft', 'ifftp']
conventions = ['recsym', 'reciso']

data_dir = f'local_abacus/cubic/z{redshift}/\
AbacusSummit_base_c{node}_ph{phase}/reconstruction/'

for nmesh in nmeshs:
    for smooth_radius in smooth_radii:
        for recon_algo in recon_algos:
            for convention in conventions:
        
                if mpicomm.rank == mpiroot:
                    # ----- read data
                    data_fn = Path(data_dir,
                        f'data_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                    data = np.load(data_fn, allow_pickle=True).item()
                    print(f'Shape of data: {len(data["Position_rec"])}')

                    # ----- read randoms
                    randoms = {}
                    randoms['Position_rec'] = np.empty((0, 3))
                    for i in range(1, 6):
                        randoms_fn = Path(data_dir,
                            f'randoms{i}_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                        randoms_i = np.load(randoms_fn, allow_pickle=True).item()
                        randoms['Position_rec'] = np.append(randoms['Position_rec'], randoms_i['Position_rec'], axis=0)

                    print(f'Shape of concatenated randoms: {len(randoms["Position_rec"])}')
                
                else:
                    data = {'Position': None, 'Position_rec': None}
                    randoms = {'Position': None, 'Position_rec': None}

                # ---- Compute power spectra

                kedges = np.arange(0.01, 0.5, 0.001)

                poles = CatalogFFTPower(
                    data_positions1=data['Position'],
                    boxsize=box_size, boxcenter=box_center,
                    nmesh=512, resampler='cic',
                    interlacing=2, ells=(0, 1, 2, 3, 4), los=los,
                    edges=kedges, position_type='pos',
                    wrap=True, mpicomm=mpicomm, mpiroot=mpiroot,
                ).poles

                poles_recon = CatalogFFTPower(
                    data_positions1=data['Position_rec'],
                    boxsize=box_size, boxcenter=box_center,
                    shifted_positions1=randoms['Position_rec'],
                    nmesh=512, resampler='cic', interlacing=2,
                    ells=(0, 1, 2, 3, 4), los=los, edges=kedges, position_type='pos',
                    wrap=True, mpicomm=mpicomm, mpiroot=mpiroot,
                ).poles

                output_dir = f'local_abacus/cubic/z{redshift}/AbacusSummit_base_c{node}_ph{phase}/reconstruction/pk'
                Path(output_dir).mkdir(parents=True, exist_ok=True)

                cout = {
                    'k': poles.k,
                    'Pk_0': poles(ell=0, complex=False),
                    'Pk_1': poles(ell=1, complex=False),
                    'Pk_2': poles(ell=2, complex=False),
                    'Pk_3': poles(ell=3, complex=False),
                    'Pk_4': poles(ell=4, complex=False),
                }
                output_fn = Path(output_dir, f'Pk_z{redshift}.npy')
                np.save(output_fn, cout)

                cout = {
                    'k': poles.k,
                    'Pk_0': poles_recon(ell=0, complex=False),
                    'Pk_1': poles_recon(ell=1, complex=False),
                    'Pk_2': poles_recon(ell=2, complex=False),
                    'Pk_3': poles_recon(ell=3, complex=False),
                    'Pk_4': poles_recon(ell=4, complex=False),
                }

                output_fn = Path(output_dir, f'Pk_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                np.save(output_fn, cout)

     



