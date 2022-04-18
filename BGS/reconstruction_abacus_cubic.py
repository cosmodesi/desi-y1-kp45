import numpy as np
import h5py
from pathlib import Path
from cosmoprimo.fiducial import DESI
from pyrecon import (setup_logging, MultiGridReconstruction,
    IterativeFFTReconstruction, IterativeFFTParticleReconstruction)


# Set up logging
setup_logging()

# define simulation and configuration parameters
node, phase = '000', '000'
redshift = 0.2
bias = 1.63
magnitude_cut = -21
box_size = 2000.
box_center = 0.
nmeshs = [512]
recon_algos = ['multigrid', 'ifft']
conventions = ['recsym', 'reciso']
smoothing_radii = [15.0]
los = 'z'
offset = box_center - box_size / 2

# define fiducial cosmology
cosmo = DESI()
power = cosmo.get_fourier().pk_interpolator().to_1d(z=redshift)
f = (cosmo.sigma8_z(z=redshift, of='theta_cb')
    / cosmo.sigma8_z(z=redshift, of='delta_cb'))
H_0 = 100.0
az = 1/(1 + redshift)
Omega_m = cosmo._params['Omega_cdm'] + cosmo._params['Omega_b']
Omega_l = 1 - Omega_m
Hz = H_0 * np.sqrt(Omega_m * (1 + redshift)**3 + Omega_l)

# ----- read data catalogue
data_dir = f'/global/homes/e/epaillas/data/mock_challenge/bgs/\
abacus_cubic/z{redshift:.3f}/AbacusSummit_base_c{node}_ph{phase}'
data_fn = Path(data_dir, f'BGS_box_ph{phase}.hdf5')
data = {}
fin = h5py.File(data_fn, 'r')
M_r = fin['Data']['abs_mag'][()]
idx1 = M_r < magnitude_cut
data['Position'] = fin['Data']['pos'][()][idx1]
data['Velocity'] = fin['Data']['vel'][()][idx1]
fin.close()

# apply redshift-space distortions
data['Position'][:, 2] += data['Velocity'][:, 2] / (az * Hz)
data['Position'] = (data['Position'] - offset) % box_size + offset

nden = len(data['Position']) / box_size ** 3 

print(f"Shape of data pos: {np.shape(data['Position'])}")

# ----- run reconstruction
output_dir = f'local_abacus/cubic/z{redshift}/\
AbacusSummit_base_c{node}_ph{phase}/reconstruction/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
for nmesh in nmeshs:
    for smooth_radius in smoothing_radii:
        for recon_algo in recon_algos:
            for convention in conventions:

                if recon_algo == 'multigrid':
                    ReconstructionAlgorithm = MultiGridReconstruction 
                elif recon_algo == 'ifft':
                    ReconstructionAlgorithm = IterativeFFTReconstruction 
                elif recon_algo == 'ifftp':
                    ReconstructionAlgorithm = IterativeFFTParticleReconstruction 

                recon = ReconstructionAlgorithm(
                    f=f, bias=bias, los=los, nmesh=nmesh,
                    boxsize=box_size, boxcenter=box_center,
                    wrap=True
                )
                recon.assign_data(data['Position'])
                recon.set_density_contrast(smoothing_radius=smooth_radius)
                recon.run()

                data['Position_rec'] = data['Position'] - recon.read_shifts(data['Position'], field='disp+rsd')
                data['Position_rec'] = (data['Position_rec'] - offset) % box_size + offset
                output_fn = Path(output_dir,
                    f'data_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                np.save(output_fn, data)

                # ----- build random catalogue
                nrand = 20 * int(nden * box_size ** 3)
                nrand_split = int(nrand / 5)
                for i in range(1, 6):
                    randoms = {}
                    randoms['Position'] = np.array([np.random.uniform(box_center - box_size/2.,
                        box_center + box_size/2., nrand_split) for i in range(3)]).T

                    print(f"Shape of randoms_{i} pos: {np.shape(randoms['Position'])}")

                    if convention ==  'recsym':
                        field = 'disp+rsd'
                    elif convention == 'reciso':
                        field = 'disp'
                    else:
                        raise Exception('Invalid RSD convention.')

                    randoms['Position_rec'] = randoms['Position'] - recon.read_shifts(randoms['Position'], field=field)
                    randoms['Position_rec'] = (randoms['Position_rec'] - offset) % box_size + offset

                    output_fn = Path(output_dir,
                        f'randoms{i}_{recon_algo}_{convention}_z{redshift}_nmesh{nmesh}_Rs{smooth_radius}_bias{bias}.npy')
                    np.save(output_fn, randoms)


