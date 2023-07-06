#standard python
"""
Created by Uendert Andrade, 13 Fev 2023.

Example
-------

```python py/bao_fs_fit.py --type LRG --survey main --basedir_out test_w0-0.9040043101843285_wa0.025634205416364297 --verspec mocks/FirstGenMocks/AbacusSummit/Y1/mock1 --version '' --region NScomb --blind_cosmology test_w0-0.9040043101843285_wa0.025634205416364297 --covmat_pk pk_xi_measurements/CovPk --blinded_index 1 --covmat_xi AbacusSummit/CutSky/Y1-blinding/```
"""

import os

import numpy as np

from cosmoprimo.fiducial import DESI
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, DirectPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.samples import Profiles
from desilike.emulators import Emulator, TaylorEmulatorEngine, EmulatedCalculator
from desilike import setup_logging


setup_logging()

if os.environ.get('NERSC_HOST', None) not in ['cori', 'perlmutter']:
    raise ValueError('NERSC_HOST not known (code only works on NERSC), not proceeding')
scratch_dir = os.environ['SCRATCH']


def parse_args():
    """
    This function parses command line arguments for the bao_fit executable and returns them as an
    :class:`argparse.Namespace` object.
    """
    import argparse

    description = 'Script code for fitting BAO in blinded power spectrum measurements.\n'

    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument(
        '--type', type=str,
        help='tracer type to be selected')
    parser.add_argument(
        '--basedir_in', type=str,
        help='base directory for input, default is location for official catalogs',
        default='/global/cfs/cdirs/desi/survey/catalogs/')
    parser.add_argument(
        '--basedir_out', type=str,
        help='base directory for output, default is C(P)SCRATCH',
        default=scratch_dir)
    parser.add_argument(
        '--version', type=str,
        help='catalog version',
        default='EDAbeta')
    parser.add_argument(
        '--survey', type=str,
        help='e.g., main (for all), DA02, any future DA',
        default='DA02')
    parser.add_argument(
        '--verspec', type=str,
        help='version for redshifts',
        default='guadalupe')
    parser.add_argument(
        '--region',
        help='regions', type=str,
        # nargs='*',
        choices=['NGC', 'SGC', 'NS', 'S', 'NGCS', 'SGCS', 'N','NScomb'],
        default='N')
    parser.add_argument(
        '--blind_cosmology', type=str,
        help='directory with blind cosmology and catalogs',
        default='unblinded')
    parser.add_argument(
        '--preli_fit_load', action='store_true',
        help='whether to perform preliminary BAO fit or load it from disk')
    parser.add_argument(
        '--fixed_covariance', type=str,
        help='whether to fix covariance; in this case, path to directory where profiles will be saved')
    parser.add_argument(
        '--covmat_pk', type=str,
        help='wheter to load Pk covariance from DISK; in this case, path to directory where CovPk are saved')
    parser.add_argument(
        '--blinded_index', type=str,
        help='When load Pk covariance from DISK, please also provide blinded_index; zero means unblinded case.', # choides: int [0 1 2 3 4 5 6 7 8] zero is for unblinded
        default='blinded_index_needed')
    parser.add_argument(
        '--covmat_xi', type=str,
        help='wheter to load xi covariance from DISK; in this case, path to directory where RascalC Cov are saved')
    parser.add_argument(
        '--todo', type=str, nargs='*', choices=['bao', 'emulator', 'fs', 'profiling', 'sampling', 'direct'], default=['bao', 'profiling'],
        help='what to do')

    return parser.parse_args()


def cut_matrix(cov, xcov, ellscov, xlim):
    '''
    The function cuts a matrix based on specified indices and returns the resulting submatrix.

    Parameters
    ----------
    cov : 2D array
        A square matrix representing the covariance matrix.
    xcov : 1D array
        x-coordinates in the covariance matrix.
    ellscov : list
        Multipoles in the covariance matrix.
    xlim : tuple
        `xlim` is a dictionary where the keys are `ell` and the values are tuples of two floats
        representing the lower and upper limits of `xcov` for that `ell` value to be returned.

    Returns
    -------
    cov : array
        Subset of the input matrix `cov`, based on `xlim`.
        The subset is determined by selecting rows and columns of `cov` corresponding to the
        values of `ell` and `xcov` that fall within the specified `xlim` range.
    '''
    assert len(cov) == len(xcov) * len(ellscov), 'Input matrix has size {}, different than {} x {}'.format(len(cov), len(xcov), len(ellscov))
    indices = []
    for ell, xlim in xlim.items():
        index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
        index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
        indices.append(index)
    indices = np.concatenate(indices, axis=0)
    return cov[np.ix_(indices, indices)]


def get_footprint(tracer, region, zmin, zmax, completeness=''):
    """
    This function selects a region and concatenates data and random catalogs, renormalizing random
    weights before concatenation.

    Parameters
    ----------
    tracer : str
        The tracer ("LRG", "ELG", or "QSO").
    region : str
        "NGC" for the North Galactic Cap or "SGC" for the South Galactic Cap.
    zmin : float
        Selected galaxies with redshift greater than ``zmin``.
    zmax : float
        Selected galaxies with redshift less than ``zmax``.
    completeness : str, default=''
        'complete_' to select complete catalogs (without fiber assignment).
    cosmo : Cosmoprimo.Cosmology, default=None
        Cosmology for the redshift to distance relation.
        Defaults to :class:`cosmoprimo.fiducial.DESI`.

    Returns
    -------
    footprint : desilike.observables.galaxy_clustering.CutskyFootprint
        A footprint instance, containing everything needed to compute effective volume and redshift for approximate covariance calculation.
    """
    import healpy as hp
    import mpytools as mpy
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint

    cosmo = DESI()

    def select_region(catalog, region):
        '''This function selects a region of the sky and returns the catalog of galaxies in that region.'''
        mask = (catalog['Z'] > zmin) & (catalog['Z'] < zmax)
        if 'NGC' in region:
            mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
        if 'SGC' in region:
            mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
        return catalog[mask]

    def concatenate(list_data, list_randoms, region):
        '''This function concatenates data and random catalogs, renormalizing random weights before concatenation.'''
        list_data = [select_region(catalog, region) for catalog in list_data]
        list_randoms = [select_region(catalog, region) for catalog in list_randoms]
        wsums_data = [data['WEIGHT'].csum() for data in list_data]
        wsums_randoms = [randoms['WEIGHT'].csum() for randoms in list_randoms]
        alpha = sum(wsums_data) / sum(wsums_randoms)
        alphas = [wsum_data / wsum_randoms / alpha for wsum_data, wsum_randoms in zip(wsums_data, wsums_randoms)]
        if list_data[0].mpicomm.rank == 0:
            print('Renormalizing randoms weights by {} before concatenation.'.format(alphas))
        for randoms, alpha in zip(list_randoms, alphas):
            randoms['WEIGHT'] *= alpha
        return Catalog.concatenate(list_data), Catalog.concatenate(list_randoms)

    if 'iron' in args.verspec:
        data_NS_fns = os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, completeness, region))
        randoms_NS_fns = os.path.join(catalog_dir, '{}_{}{}_0_clustering.ran.fits'.format(tracer, completeness, region))
        print('Loading data {}.'.format(data_NS_fns))
        print('Loading randoms {}.'.format(randoms_NS_fns))
        data_NS = [Catalog.read(data_NS_fns)]
        randoms_NS = [Catalog.read(randoms_NS_fns)]
    else:
        data_NS_fns = [os.path.join(catalog_dir, '{}_{}{}_clustering.dat.fits'.format(tracer, completeness, reg)) for reg in ['N', 'S']]
        randoms_NS_fns = [os.path.join(catalog_dir, '{}_{}{}_0_clustering.ran.fits'.format(tracer, completeness, reg)) for reg in ['N', 'S']]
        print('Loading data {}.'.format(data_NS_fns))
        print('Loading randoms {}.'.format(randoms_NS_fns))
        data_NS = [Catalog.read(fn) for fn in data_NS_fns]
        randoms_NS = [Catalog.read(fn) for fn in randoms_NS_fns]

    if region in ['NGC', 'SGC', 'NS']:
        data, randoms = concatenate(data_NS, randoms_NS, region)
    elif region in ['S', 'NGCS', 'SGCS']:
        data, randoms = concatenate(data_NS[1:], randoms_NS[1:], region)
    elif region in ['N']:
        data, randoms = concatenate(data_NS[:1], randoms_NS[:1], region)
    else:
        raise ValueError('Unknown region {}'.format(region))

    mpicomm = data.mpicomm
    nside = 512
    theta, phi = np.radians(90 - randoms['DEC']), np.radians(randoms['RA'])
    hpindex = hp.ang2pix(nside, theta, phi, lonlat=False)
    hpindex = mpy.gather(np.unique(hpindex), mpicomm=mpicomm, mpiroot=0)
    fsky = mpicomm.bcast(np.unique(hpindex).size if mpicomm.rank == 0 else None, root=0) / hp.nside2npix(nside)
    area = fsky * 4. * np.pi * (180. / np.pi)**2
    alpha = data['WEIGHT'].csize / randoms['WEIGHT'].csum()
    density = RedshiftDensityInterpolator(z=randoms['Z'], weights=alpha * randoms['WEIGHT'], bins=30, fsky=fsky, distance=cosmo.comoving_radial_distance, mpicomm=mpicomm)
    return CutskyFootprint(area=area, zrange=density.z, nbar=density.nbar, cosmo=cosmo)


def read_pk(tracer, region, zmin, zmax, plot=False):
    """Read power spectrum from file."""

    from pypower import PowerSpectrumStatistics
    
    if 'iron' in args.verspec:
        poles = PowerSpectrumStatistics.load(os.path.join(data_dir, 'pk','jmena', 'pkpoles_{}_{}_{}_{}_default_FKP_lin.npy'.format(tracer, region, zmin, zmax)))
    else:
        poles = PowerSpectrumStatistics.load(os.path.join(data_dir, 'pk', 'pk', 'pkpoles_{}_{}_{}_{}_default_lin.npy'.format(tracer, region, zmin, zmax)))

    if plot:
        print('Shot noise is {:.4f}.'.format(poles.shotnoise))  # if cross-correlation, shot noise is 0.
        print('Normalization is {:.4f}.'.format(poles.wnorm))
        poles.plot(show=True)

    return poles


def read_xi(tracer, region, zmin, zmax, plot=False):
    """Read correlation function from file."""

    from pycorr import TwoPointCorrelationFunction

    if 'iron' in args.verspec:
        corr = TwoPointCorrelationFunction.load(os.path.join(data_dir, 'xi', 'smu', 'allcounts_{}_{}_{}_{}_default_FKP_lin_njack0_nran1_split20.npy'.format(tracer, region, zmin, zmax)))
    else:
        corr = TwoPointCorrelationFunction.load(os.path.join(data_dir, 'xi', 'smu', 'allcounts_{}_{}_{}_{}_default_lin_njack0_nran1_split20.npy'.format(tracer, region, zmin, zmax)))

    if plot:
        corr.plot(show=True)

    return corr


def get_blind_cosmo(z, tracer, *args, **kwargs):
    """
    This function returns a dictionary of cosmological parameters with blinded parameters if specified.

    Parameters
    ----------
    z : float
        Redshift at which to evaluate `qpar`, `qper`, `df`.
    tracer : str
        The tracer ("LRG", "ELG", or "QSO").

    Returns
    -------
    cosmo : dict
        A dictionary containing the values of `qpar`, `qper`, `df`, and `dm`.
    """
    blinded = 'unblinded' not in in_dir and 'Y1/LSS/iron' not in in_dir

    # Template cosmology for the BAO fits
    cosmo = fiducial = DESI()

    if blinded:
        fn = os.path.join(in_dir, 'LSScats','blinded','blinded_parameters_{}.csv'.format(tracer))
        print('\nLoad blinded_parameters from: \n', fn)
        w0_blind, wa_blind, f_blind = np.loadtxt(fn, delimiter=',', skiprows=1)
        cosmo = fiducial.clone(w0_fld=w0_blind, wa_fld=wa_blind)

    if blinded:
        qpar = cosmo.efunc(z) / fiducial.efunc(z)
        qper = fiducial.comoving_angular_distance(z) / cosmo.comoving_angular_distance(z)
        df = f_blind / cosmo.growth_rate(z)
    else:
        qpar = qper = df = 1.
    print('Expected blinding:\nqpar_th:{} \nqper_th:{} \ndf_th:{}'.format(qpar, qper, df))
    return dict(qpar=qpar, qper=qper, df=df, dm=0.)


def fit_pk(out_dir, tracer, region, covmat_params=None, covmat_pk=None, wmat_pk=None, blinded_index=None, theory_name='bao', save_emulator=False, emulator_fn='power_emulator_{}_{}_{}_{}_{}.npy', template_name='shapefit', todo='profiling', **kwargs):
    """
    This function performs a power spectrum fit for a given tracer and region, using a specified theory
    and covariance matrix, and saves the resulting profiles and / or chains.

    Parameters
    ----------
    out_dir : str
        The directory where the output files will be saved.
    tracer : str
        The tracer ("LRG", "ELG", or "QSO").
    region : str
        "NGC" for the North Galactic Cap or "SGC" for the South Galactic Cap.
    covmat_params : dict, str, default=None
        A dictionary or path to directory containing theory parameters to estimate the covariance matrix used in the final fit.
        If ``None``, and ``covmat_pk`` is not provided, a preliminary fit will be run to compute a more accurate covariance matrix.
    covmat_pk : str, default=None
        Optionally, the path to the directory containing the pk covariance matrix files.
    wmat_pk : str, default=None
        The path to the directory where the window function is saved.
    blinded_index : int, default=None
        The index of the blinded parameter, where 0 indicates an unblinded parameter.
    theory_name : str, default='bao'
        Which theoretical model to use for the power spectrum calculation. It can be either 'bao', 'fs', or 'velocileptors'.
    save_emulator : bool, default=False
        If used, compute and save the emulator used for the power spectrum theory to a file specified by the emulator_fn parameter.
        If ``False``, no emulator will be saved.
    emulator_fn : str, default='power_emulator_{}.npy'
        The filename to save or load the power spectrum emulator. If set to ``None``, no emulator will be used.
    template_name : str, default='shapefit'
        The name of the power spectrum template to use for the fit. It can be either 'bao', 'shapefit', 'direct'.
    todo : str, default='profiling'
        Specifies which tasks to perform. It can contain the values "profiling" and/or "sampling", which indicate whether to run posterior profiling or sampling, respectively.
    """
    fiducial = DESI()
    zmin, zmax, z, b0 = {'LRG': (0.4, 1.1, 0.8, 1.7), 'ELG': (1.1, 1.6, 1.1, 0.84), 'QSO': (0.8, 2.1, 1.4, 1.2)}[tracer]
    b1 = b0 / fiducial.growth_factor(z)
    footprint, expected = None, get_blind_cosmo(z, tracer, region, zmin, zmax, **kwargs)

    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name, tracer, region, zmin, zmax)

    data = read_pk(tracer, region, zmin, zmax, **kwargs)

    if theory_name == 'bao':
        fixed_params = {'sigmaper': 4., 'sigmapar': 8.}
        klim = {0: [0.02, 0.30, 0.005], 2: [0.02, 0.30, 0.005]}
        template = BAOPowerSpectrumTemplate(z=z, fiducial=fiducial)
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template)
        solved_params = ['al*_*']
    elif theory_name in ['fs', 'velocileptors']:
        fixed_params = {}
        if tracer == 'QSO':
            klim = {0: [0.02, 0.30, 0.005], 2: [0.02, 0.30, 0.005], 4: [0.02, 0.30, 0.005]}
        else:
            klim = {0: [0.02, 0.20, 0.005], 2: [0.02, 0.20, 0.005], 4: [0.02, 0.20, 0.005]}
        
        # template_name = 'direct'
        # print("====================================")
        # print('Using {} template'.format(template_name))
        # print("====================================")
        template = (ShapeFitPowerSpectrumTemplate if template_name == 'shapefit' else DirectPowerSpectrumTemplate)(z=z, fiducial=fiducial)
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
        theory.params['b1'].update(ref={'limits': [b1 - 1.2, b1 - 0.8]})
        theory.params['alpha6'].update(fixed=True)
        if 4 not in klim:
            for param in theory.params.select(basename=['alpha4', 'ct4*', 'sn4*']): param.update(fixed=True)
        solved_params = ['alpha*', 'sn*']
    else:
        raise ValueError('Unknown theory {}'.format(theory_name))
    expected = {name: value for name, value in expected.items() if name in template.params}

    run_preliminary = covmat_params is None and covmat_pk is None
    covmat = None
    from desilike.utils import is_path
    if run_preliminary:
        print('\nNo preliminary BAO fit provided. Runing BAO fit preliminary pipeline then.\n')
        covmat_params = {}
    elif is_path(covmat_params):
        fn = os.path.join(covmat_params, 'profile_{}_{}_{}_{}.npy'.format(tracer, region, zmin, zmax))
        profiles = Profiles.load(fn)
        covmat_params = {}
        if 'covmat' in profiles.attrs:
            covmat = profiles.attrs['covmat']
        else:
            covmat_params = profiles.choice(index='argmax', derived=False)
    else:
        covmat_params = {}
        blinded_index = blinded_index  # [0 1 2 3 4 5 6 7 8] zero is unblinded
        if region != 'NScomb':
            raise ValueError('Only CovPT for NScomb so far.')
        covmat_pk_fn = os.path.join(covmat_pk, 'CovGaussian_Pk_0_{}_{}_zmin{}_zmax{}_{}.txt'.format(tracer, region, zmin, zmax, blinded_index))
        k_cov_fn = os.path.join(covmat_pk, '..', 'k.txt')
        print('\nLoading Pk covariance from {}.\n'.format(covmat_pk_fn))
        print('Loading k_cov from {}.\n'.format(k_cov_fn))
        cov = np.loadtxt(covmat_pk_fn)
        k_cov = np.loadtxt(k_cov_fn)
        covmat = cut_matrix(cov, k_cov, (0, 2, 4), klim)  # Now keeping k between 0.02 and 0.3 and removing the third multipole (l=4)
        print("\n covmat:{}\n".format(covmat.shape))

    if wmat_pk is not None:
        # People should save the wide-angle-resumed matrix directly... (in this case, do: wmatrix = BaseMatrix.load(fn))
        #from pypower import PowerSpectrumSmoothWindowMatrix
        #wmatrix = PowerSpectrumSmoothWindowMatrix.load(os.path.join(wmat_pk, 'window_smooth_{}_{}_{}_{}_default_lin_matrix.npy'.format(tracer, region, zmin, zmax)))
        #wmatrix.resum_input_odd_wide_angle()
        from pypower import BaseMatrix, PowerSpectrumOddWideAngleMatrix, PowerSpectrumSmoothWindowMatrix, PowerSpectrumSmoothWindow
        if 'iron' in args.verspec:
            wmat_fn = 'wmatrix_{}_{}_{}_{}_default_FKP_lin.npy'.format(tracer, region, zmin, zmax)
        else:
            wmat_fn = 'wmatrix_{}_{}_{}_{}_default_lin.npy'.format(tracer, region, zmin, zmax)
        if os.path.isfile(wmat_fn):
            wmatrix = BaseMatrix.load(wmat_fn)
        else:
            if 'iron' in args.verspec:
                window_fn = os.path.join(wmat_pk, 'window_smooth_{}_{}_{}_{}_default_FKP_lin.npy'.format(tracer, region, zmin, zmax))
            else:
                window_fn = os.path.join(wmat_pk, 'window_smooth_{}_{}_{}_{}_default_lin.npy'.format(tracer, region, zmin, zmax))
            try:
                window = PowerSpectrumSmoothWindow.load(window_fn)  # for some reason, LRG have PowerSpectrumSmoothWindow but QSO have BaseMatrix
            except AttributeError:
                wmatrix = BaseMatrix.load(window_fn)
                wmatrix.save(wmat_fn)
            else:
                kout = data.k
                ellsout = [0, 2, 4]  # output multipoles
                ellsin = [0, 2, 4]  # input (theory) multipoles
                wa_orders = 1  # wide-angle order
                sep = np.geomspace(1e-4, 1e4, 1024 * 16)  # configuration space separation for FFTlog
                kin_rebin = 4  # rebin input theory to save memory
                kin_lim = (0, 1.)  # pre-cut input (theory) ks to save some memory
                # Input projections for window function matrix:
                # theory multipoles at wa_order = 0, and wide-angle terms at wa_order = 1
                projsin = ellsin + PowerSpectrumOddWideAngleMatrix.propose_out(ellsin, wa_orders=wa_orders)
                # Window matrix
                wmatrix = PowerSpectrumSmoothWindowMatrix(kout, projsin=projsin, projsout=ellsout, window=window, sep=sep, kin_rebin=kin_rebin, kin_lim=kin_lim)
                # We resum over theory odd-wide angle
                wmatrix.resum_input_odd_wide_angle()
                wmatrix.save(wmat_fn)

    observable = TracerPowerSpectrumMultipolesObservable(data=(data or {}),  # data can be a dictionary of parameters
                                                         # fit monopole and quadrupole, between 0.01 and 0.4 h/Mpc, with 0.005 h/Mpc steps
                                                         klim=klim,
                                                         theory=theory,
                                                         wmatrix=wmatrix,
                                                         kinlim=(0.01, 0.4),
                                                         ellsin=list(klim.keys()))
    if covmat is None:
        footprint = get_footprint(tracer, 'NS' if region == 'NScomb' else region, zmin, zmax, **kwargs)
        covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)  # Gaussiancovariance matrix

    covmat_params = {'b1': b1, **expected, **fixed_params, **covmat_params}
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance(**covmat_params) if covmat is None else covmat)
    expected = {param: value for param, value in expected.items() if param in likelihood.all_params}

    if save_emulator:
        likelihood()
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
        theory.init.update(pt=emulator.to_calculator())
    elif emulator_fn is not None:
        theory.init.update(pt=EmulatedCalculator.load(emulator_fn))

    for param in fixed_params:
        likelihood.all_params[param].update(fixed=True, value=fixed_params[param])
    for param in expected:
        likelihood.all_params[param].update(fixed=True, value=expected[param])

    from desilike.profilers import MinuitProfiler

    def save_profiles(profiles, base=''):
        """This function saves profiles and generates plots and summary statistics."""
        likelihood(**profiles.bestfit.choice(input=True))
        if likelihood.mpicomm.rank == 0:
            observable = likelihood.observables[0]
            observable.plot(fn=os.path.join(out_dir, '{}poles_bestfit_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            profiles.attrs['expected'] = expected
            profiles.attrs['covmat'] = observable.covariance
            profiles.save(os.path.join(out_dir, '{}profile_{}_{}_{}_{}.npy'.format(base, tracer, region, zmin, zmax)))
            observable.plot_covariance_matrix(fn=os.path.join(out_dir, '{}covariance_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            profiles.to_stats(fn=os.path.join(out_dir, '{}profile_{}_{}_{}_{}.stats'.format(base, tracer, region, zmin, zmax)))
            print(profiles.to_stats(tablefmt='pretty'))

    def save_chains(chains, base=''):
        """This function saves chains and generates plots and summary statistics."""
        from desilike.samples import Chain, plotting
        if likelihood.mpicomm.rank == 0:
            chain = Chain.concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])
        likelihood(**likelihood.mpicomm.bcast(chain.choice(index='argmax', input=True) if likelihood.mpicomm.rank == 0 else None, root=0))
        if likelihood.mpicomm.rank == 0:
            observable = likelihood.observables[0]
            observable.plot(fn=os.path.join(out_dir, '{}poles_chain_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            observable.plot_covariance_matrix(fn=os.path.join(out_dir, '{}covariance_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            chain.attrs['expected'] = expected
            chain.attrs['covmat'] = observable.covariance
            chain.save(os.path.join(out_dir, '{}chain_{}_{}_{}_{}.npy'.format(base, tracer, region, zmin, zmax)))
            chain.to_stats(fn=os.path.join(out_dir, '{}chain_{}_{}_{}_{}.stats'.format(base, tracer, region, zmin, zmax)))
            plotting.plot_triangle(chain, fn=os.path.join(out_dir, '{}triangle_chain_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            print(chain.to_stats(params=['qpar', 'qper', 'df', 'dm'], tablefmt='pretty'))

    if run_preliminary:
        # Preliminary fit, to recompute a more accuratecovariance matrix
        for param in expected:
            likelihood.all_params[param].update(fixed=True, value=expected[param])
        profiler = MinuitProfiler(likelihood, seed=42)
        profiles = profiler.maximize(niterations=3)
        covmat_params.update(profiles.bestfit.choice(index='argmax', varied=True))
        save_profiles(profiles, base='preliminary-')

    # Final fit with newcovariance matrix
    likelihood.init.update(covariance=covariance(**covmat_params) if covmat is None else covmat)
    for param in fixed_params:
        likelihood.all_params[param].update(fixed=True, value=fixed_params[param])
    for param in expected:
        likelihood.all_params[param].update(fixed=False, value=expected[param])

    if 'profiling' in todo:
        profiler = MinuitProfiler(likelihood, seed=42)
        profiles = profiler.maximize(niterations=3)
        #profiles = profiler.interval([param for param in likelihood.varied_params if param in template.params])
        save_profiles(profiles, base='')

    if 'sampling' in todo:
        from desilike.samplers import EmceeSampler
        for param in likelihood.all_params.select(basename=solved_params): param.update(prior=None, derived='.auto')
        chains = [os.path.join(out_dir, 'chain_{}_{}_{}_{}_{:d}.npy'.format(tracer, region, zmin, zmax, ichain)) for ichain in range(4)]
        sampler = EmceeSampler(likelihood, chains=len(chains), nwalkers=40, seed=42, save_fn=chains)
        chains = sampler.run(check={'max_eigen_gr': 0.03})
        save_chains(chains, base='')


def fit_xi(out_dir, tracer, region, covmat_params=None, covmat_xi=None, theory_name='bao', save_emulator=False, emulator_fn='corr_emulator_{}_{}_{}_{}_{}.npy', pk_emulator_fn='power_emulator_{}_{}_{}_{}_{}.npy', template_name='shapefit', todo='profiling', **kwargs):
    """
    This function performs a correlation function fit for a given tracer and region, using a specified theory
    and covariance matrix, and saves the resulting profiles and / or chains.

    Parameters
    ----------
    out_dir : str
        The directory where the output files will be saved.
    tracer : str
        The tracer ("LRG", "ELG", or "QSO").
    region : str
        "NGC" for the North Galactic Cap or "SGC" for the South Galactic Cap.
    covmat_params : dict, str, default=None
        A dictionary or path to directory containing theory parameters to estimate the covariance matrix used in the final fit.
        If ``None``, and ``covmat_pk`` is not provided, a preliminary fit will be run to compute a more accurate covariance matrix.
    covmat_xi : str, default=None
        Optionally, the path to the directory containing the xi covariance matrix files.
    theory_name : str, default='bao'
        Which theoretical model to use for the power spectrum calculation. It can be either 'bao', 'fs', or 'velocileptors'.
    save_emulator : bool, default=False
        If used, compute and save the emulator used for the power spectrum theory to a file specified by the emulator_fn parameter.
        If ``False``, no emulator will be saved.
    emulator_fn : str, default='power_emulator_{}.npy'
        The filename to save or load the power spectrum emulator. If set to ``None``, no emulator will be used.
    template_name : str, default='shapefit'
        The name of the power spectrum template to use for the fit. It can be either 'bao', 'shapefit', 'direct'.
    todo : str, default='profiling'
        Specifies which tasks to perform. It can contain the values "profiling" and/or "sampling", which indicate whether to run posterior profiling or sampling, respectively.
    """
    fiducial = DESI()
    zmin, zmax, z, b0 = {'LRG': (0.4, 1.1, 0.8, 1.7), 'ELG': (1.1, 1.6, 1.1, 0.84), 'QSO': (0.8, 2.1, 1.4, 1.2)}[tracer]
    b1 = b0 / fiducial.growth_factor(z)
    footprint, expected = None, get_blind_cosmo(z, tracer, region, zmin, zmax, **kwargs)

    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(theory_name, tracer, region, zmin, zmax)

    data = read_xi(tracer, region, zmin, zmax, **kwargs)

    fiducial = DESI()
    if theory_name == 'bao':
        fixed_params = {'sigmaper': 4., 'sigmapar': 8.}
        slim = {0: [50., 150., 4.], 2: [50., 150., 4.]}
        template = BAOPowerSpectrumTemplate(z=z, fiducial=fiducial)
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template)
        solved_params = ['al*_*']
    elif theory_name in ['fs', 'velocileptors']:
        fixed_params = {}
        slim = {0: [30., 150., 4.], 2: [30., 150., 4.], 4: [30., 150., 4.]}
        template = (ShapeFitPowerSpectrumTemplate if template_name == 'shapefit' else DirectPowerSpectrumTemplate)(z=z, fiducial=fiducial)
        theory = LPTVelocileptorsTracerCorrelationFunctionMultipoles(template=template)
        theory.params['b1'].update(ref={'limits': [b1 - 1.2, b1 - 0.8]})
        theory.params['alpha6'].update(fixed=True)
        if 4 not in slim:
            for param in theory.params.select(basename=['alpha4', 'ct4*', 'sn4*']): param.update(fixed=True)
        for param in theory.params.select(basename='sn*'): param.update(fixed=True)
        solved_params = ['alpha*']
    else:
        raise ValueError('Unknown theory {}'.format(theory_name))
    expected = {name: value for name, value in expected.items() if name in template.params}

    run_preliminary = covmat_params is None and covmat_xi is None
    covmat = None
    from desilike.utils import path_types
    if run_preliminary:
        print('\n-->  No preliminary BAO fit provided. Runing BAO fit preliminary pipeline then.\n')
        covmat_params = {}
    elif isinstance(covmat_params, path_types):
        fn = os.path.join(covmat_params, 'profile_{}_{}_{}_{}.npy'.format(tracer, region, zmin, zmax))
        profiles = Profiles.load(fn)
        covmat_params = {}
        if 'covmat' in profiles.attrs:
            covmat = profiles.attrs['covmat']
        else:
            covmat_params = profiles.choice(index='argmax', derived=False)
    else:
        covmat_params = {}
        covmat_xi_fn = os.path.join(covmat_xi, 'xi024_unblinded_{}_{}_{}_{}_default_lin4_s20-200_cov_RascalC_rescaled.txt'.format(tracer, region, zmin, zmax))
        print('\nLoading xi covariance from {}.\n'.format(covmat_xi_fn))
        cov = np.loadtxt(covmat_xi_fn)
        covmat = cut_matrix(cov, np.linspace(22, 202, 45), (0, 2, 4), slim)  # filter only monopole and quadrupole and keeping s between 50.0 and 150.05
        print("\n covmat:{}\n".format(covmat.shape))

    #for name in ['al0_3', 'al2_3', 'al0_4', 'al2_4']:
    #    theory.params[name] = dict(value=0., fixed=False, ref=dict(limits=[-1., 1.]))
    # xi(s)=B*xi_template(s,apar,aper)+A_0+A_1/s+A_2/s^2 for each \ell , where B is an amplitude for each multipole

    observable = TracerCorrelationFunctionMultipolesObservable(data=(data or {}),  # data can be a dictionary of parameters
                                                               # fit monopole and quadrupole, between 40 and 160 Mpc/h, with 2 Mpc/h steps
                                                               slim=slim,
                                                               theory=theory)
    if covmat is None:
        footprint = get_footprint(tracer, 'NS' if region == 'NScomb' else region, zmin, zmax, **kwargs)
        if theory_name == 'bao':
            klim = {0: [0.02, 0.30, 0.005], 2: [0.02, 0.30, 0.005]}
            theory_pk = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=BAOPowerSpectrumTemplate(z=z, fiducial=fiducial))
        elif theory_name in ['fs', 'velocileptors']:
            if tracer == 'QSO':
                klim = {0: [0.02, 0.30, 0.005], 2: [0.02, 0.30, 0.005], 4: [0.02, 0.30, 0.005]}
            else:
                klim = {0: [0.02, 0.20, 0.005], 2: [0.02, 0.20, 0.005], 4: [0.02, 0.20, 0.005]}

            pt = EmulatedCalculator.load(pk_emulator_fn.format(theory_name, tracer, region, zmin, zmax))
            theory_pk = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt=pt, k=pt.k)
            theory_pk.params['b1'].update(ref={'limits': [b1 - 1.2, b1 - 0.8]})
            theory_pk.params['alpha6'].update(fixed=True)
            if 4 not in klim:
                for param in theory_pk.params.select(basename=['alpha4', 'ct4*', 'sn4*']): param.update(fixed=True)
        covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, theories=theory_pk, resolution=5)  # Gaussian covariance matrix

    covmat_params = {'b1': b0 / fiducial.growth_factor(z), **expected, **fixed_params, **covmat_params}
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=covariance(**covmat_params) if covmat is None else covmat)
    expected = {param: value for param, value in expected.items() if param in likelihood.all_params}

    if save_emulator:
        likelihood()
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
        theory.init.update(pt=emulator.to_calculator())
    elif emulator_fn is not None:
        theory.init.update(pt=EmulatedCalculator.load(emulator_fn))

    for param in fixed_params:
        likelihood.all_params[param].update(fixed=True, value=fixed_params[param])
    for param in expected:
        likelihood.all_params[param].update(fixed=True, value=expected[param])

    from desilike.profilers import MinuitProfiler

    def save_profiles(profiles, likelihood=likelihood, base=''):
        likelihood(**profiles.bestfit.choice(input=True))
        if likelihood.mpicomm.rank == 0:
            observable = likelihood.observables[0]
            observable.plot(fn=os.path.join(out_dir, '{}poles_bestfit_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            profiles.attrs['expected'] = expected
            profiles.attrs['covmat'] = observable.covariance
            profiles.save(os.path.join(out_dir, '{}profile_{}_{}_{}_{}.npy'.format(base, tracer, region, zmin, zmax)))
            observable.plot_covariance_matrix(fn=os.path.join(out_dir, '{}covariance_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            profiles.to_stats(fn=os.path.join(out_dir, '{}profile_{}_{}_{}_{}.stats'.format(base, tracer, region, zmin, zmax)))
            print(profiles.to_stats(tablefmt='pretty'))

    def save_chains(chains, base=''):
        from desilike.samples import Chain, plotting
        if likelihood.mpicomm.rank == 0:
            chain = Chain.concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])
        likelihood(**likelihood.mpicomm.bcast(chain.choice(index='argmax', input=True) if likelihood.mpicomm.rank == 0 else None, root=0))
        if likelihood.mpicomm.rank == 0:
            observable = likelihood.observables[0]
            observable.plot(fn=os.path.join(out_dir, '{}poles_chain_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            chain.attrs['expected'] = expected
            chain.attrs['covmat'] = observable.covariance
            chain.save(os.path.join(out_dir, '{}chain_{}_{}_{}_{}.npy'.format(base, tracer, region, zmin, zmax)))
            observable.plot_covariance_matrix(fn=os.path.join(out_dir, '{}covariance_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            chain.to_stats(fn=os.path.join(out_dir, '{}chain_{}_{}_{}_{}.stats'.format(base, tracer, region, zmin, zmax)))
            plotting.plot_triangle(chain, fn=os.path.join(out_dir, '{}triangle_chain_{}_{}_{}_{}.png'.format(base, tracer, region, zmin, zmax)))
            print(chain.to_stats(params=chain.params(basename=['qpar', 'qper', 'df', 'dm']), tablefmt='pretty'))

    if run_preliminary:
        # Preliminary fit, to recompute a more accuratecovariance matrix
        data = read_pk(tracer, region, zmin, zmax, **kwargs)
        observable_pk = TracerPowerSpectrumMultipolesObservable(data=(data or {}),  # data can be a dictionary of parameters
                                                                # fit monopole and quadrupole, between 0.02 and 0.3 h/Mpc, with 0.005 h/Mpc steps
                                                                klim=klim,
                                                                wmatrix='wmatrix_{}_{}_{}_{}_default_lin.npy'.format(tracer, region, zmin, zmax),
                                                                kinlim=(0.01, 0.4),
                                                                theory=theory_pk)
        covariance_pk = ObservablesCovarianceMatrix(observable_pk, footprints=footprint, resolution=5)  # Gaussiancovariance matrix
        likelihood_pk = ObservablesGaussianLikelihood(observables=observable_pk, covariance=covariance_pk(**covmat_params) if covmat is None else covmat)
        #for param in likelihood_pk.all_params.select(basename=solved_params):
        #    param.update(prior=None, derived='.auto')
        for param in fixed_params:
            likelihood_pk.all_params[param].update(fixed=True, value=fixed_params[param])
        for param in expected:
            likelihood_pk.all_params[param].update(fixed=True, value=expected[param])
        profiler = MinuitProfiler(likelihood_pk, seed=42)
        profiles = profiler.maximize(niterations=3)
        covmat_params.update(profiles.bestfit.choice(index='argmax', varied=True))
        save_profiles(profiles, likelihood=likelihood_pk, base='preliminary-')
        for param in expected:
            likelihood_pk.all_params[param].update(fixed=False)

    # Final fit with newcovariance matrix
    likelihood.init.update(covariance=covariance(**covmat_params) if covmat is None else covmat)
    for param in fixed_params:
        likelihood.all_params[param].update(fixed=True, value=fixed_params[param])
    for param in expected:
        likelihood.all_params[param].update(fixed=False, value=expected[param])

    if 'profiling' in todo:
        profiler = MinuitProfiler(likelihood, seed=42)
        profiles = profiler.maximize(niterations=3)
        #profiles = profiler.interval([param for param in likelihood.varied_params if param in template.params])
        save_profiles(profiles, base='')

    if 'sampling' in todo:
        from desilike.samplers import EmceeSampler
        for param in likelihood.all_params.select(basename=solved_params): param.update(prior=None, derived='.auto')
        chains = [os.path.join(out_dir, 'chain_{}_{}_{}_{}_{:d}.npy'.format(tracer, region, zmin, zmax, ichain)) for ichain in range(4)]
        sampler = EmceeSampler(likelihood, chains=len(chains), nwalkers=40, seed=42, save_fn=chains)
        chains = sampler.run(check={'max_eigen_gr': 0.03})
        save_chains(chains, base='')


# Setting up directories and file paths for a BAO and RSD fit
# pipeline for a specific tracer type. It checks if the input version and survey are supported, and
# sets up the necessary directories for catalogs, data, and output. It also sets up file paths for
# covariance matrices and a weight matrix.
if __name__ == '__main__':

    args = parse_args()
    print('\n' + str(args) + '\n')

    print('\n/!\  Running BAO fit pipeline for tracer type {}\n'.format(args.type))

    if 'Y1/mock' in args.verspec:  # e.g., use 'mocks/FirstGenMocks/AbacusSummit/Y1/mock1' to get the 1st mock with fiberassign
        if os.path.normpath(args.basedir_in) == os.path.normpath('/global/cfs/cdirs/desi/survey/catalogs/'):
                in_dir = os.path.join(args.basedir_in, args.survey, args.verspec, 'LSScats', args.version, 'blinded', 'jmena', args.blind_cosmology)
                catalog_dir = os.path.join(in_dir, 'LSScats', 'blinded')
                data_dir = catalog_dir
                if args.blind_cosmology == 'unblinded':
                    in_dir  = os.path.join(args.basedir_in, args.survey, args.verspec, 'LSScats', args.version)
                    catalog_dir = in_dir
                    data_dir = os.path.join(in_dir, 'blinded','jmena', 'unblinded')
        if os.path.normpath(args.basedir_in) == os.path.normpath('/pscratch/sd/u/uendert/blinding_mocks/'):
            in_dir = os.path.join(args.basedir_in, args.blind_cosmology)
            catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/' + args.survey + '/' + args.verspec + '/LSScats/' + args.version + '/blinded' + '/jmena/' + args.blind_cosmology + '/LSScats/blinded'
            data_dir = os.path.join(in_dir, 'LSScats', 'blinded')
            in_dir = os.path.join('/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1/mock1/LSScats/blinded/jmena/', args.blind_cosmology)
            if args.blind_cosmology == 'unblinded':
                in_dir  = os.path.join(args.basedir_in, args.blind_cosmology)
                catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/' + args.survey + '/' + args.verspec + '/LSScats/' + args.version
                data_dir = os.path.join(in_dir)

        if args.covmat_pk:
            covpk_dir = os.path.join(args.basedir_in, args.survey, args.verspec, 'LSScats', args.version, 'blinded', 'jmena', args.covmat_pk)# 'pk_xi_measurements', 'CovPk')
        if args.covmat_xi:
            rascalc_dir = os.path.join('/global/cfs/cdirs/desi/users/mrash/RascalC/', args.covmat_xi)
    
    if 'iron' in args.verspec:
        if os.path.normpath(args.basedir_in) == os.path.normpath('/global/cfs/cdirs/desi/survey/catalogs/'):
            in_dir = os.path.join(args.basedir_in, args.survey, 'LSS', args.verspec, 'LSScats', args.version, 'blinded')
            catalog_dir = os.path.join(in_dir)
            data_dir = catalog_dir
    else:
        raise ValueError('verspec {} not supported'.format(args.verspec))
    out_dir = args.basedir_out

    covmat_params = None
    covmat_pk = None
    covmat_xi = None
    blinded_index = None
    
    if 'iron' in args.verspec:
        wmat_pk = os.path.join(os.path.join(args.basedir_in, args.survey, 'LSS', args.verspec, 'LSScats', args.version, 'blinded', 'pk', 'jmena'))
    else:
        wmat_pk = os.path.join(args.basedir_in, args.survey, args.verspec, 'LSScats', args.version, 'blinded', 'jmena', 'unblinded', 'pk', 'pk')

    # The code is checking for certain input arguments and then running a fit pipeline for Fourier
    # space and configuration space for a given tracer type. It loads covariance matrices and
    # parameters from disk if specified, and saves the results of the fit if specified.
    ############################################## Run functions ##############################################
    if args.fixed_covariance and (args.covmat_pk or args.covmat_xi):
        raise ValueError('\n !!!! You should decide whether to load *Covariance-parameters* or *Covariance*. !!!!\n')
    if args.covmat_pk and args.blinded_index == 'blinded_index_needed':
        raise ValueError('When load Pk covariance from DISK, please also provide blinded_index.')

    if args.preli_fit_load:
        print('\n--> Load profiles from preliminary BAO fit.\n')
        covmat_params = out_dir

    elif args.fixed_covariance:
        print('\n--> Load profiles from preliminary BAO fit to the *UNBLINDED* case.\n')
        covmat_params = args.fixed_covariance

    if args.covmat_pk:
        print('\n--> Load Pk covariance from *DISK*.\n')
        covmat_pk = covpk_dir
        blinded_index = int(args.blinded_index)

    if args.covmat_xi:
        print('\n--> Load xi covariance from *DISK*.\n')
        covmat_xi = rascalc_dir

    ################################ Final FIT ################################
    for theory_name in args.todo:
        if theory_name in ['emulator', 'profiling', 'sampling']: continue
        kwargs = {'theory_name': theory_name}
        if theory_name == 'bao': kwargs['emulator_fn'] = None
        else: kwargs['template_name'] = 'shapefit'
        if 'emulator' in args.todo: kwargs['save_emulator'] = True
        kwargs['todo'] = args.todo
        print('\n-->  Running *Fourier space* fit pipeline for tracer type {}\n'.format(args.type))
        fit_pk(os.path.join(out_dir, 'pk'), args.type, args.region, covmat_params=os.path.join(covmat_params, 'bao', 'pk') if covmat_params is not None else None, covmat_pk=covmat_pk, wmat_pk=wmat_pk, blinded_index=blinded_index, **kwargs)

        print('\n-->  Running *configuration space* fit pipeline for tracer type {}\n'.format(args.type))
        fit_xi(os.path.join(out_dir, 'xi'), args.type, args.region, covmat_params=os.path.join(covmat_params, 'bao', 'xi',) if covmat_params is not None else None, covmat_xi=covmat_xi, **kwargs)
