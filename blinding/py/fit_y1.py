import os
import numpy as np
import argparse

version = 'v0.6'
emulators_dir = os.path.join(os.path.dirname(__file__), '_emulators')

# Checking the existence of directories
if os.path.exists('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.1/blinded/pk/covariances/'): # NOTE: this is the directory for the TheCov covariance matrices still version v0.1
    TheCov_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.1/blinded/pk/covariances/'
else:
    TheCov_dir = None

if os.path.exists('/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/blinded/{}/'.format(version)):
    RascalC_dir = '/global/cfs/cdirs/desi/users/mrash/RascalC/Y1/blinded/{}/'.format(version)
else:
    RascalC_dir = None

# Setting the directories to None to force using the alternate directory
TheCov_dir = None 
# RascalC_dir = None 

# Error handling if neither directory exists
if TheCov_dir is None and RascalC_dir is None:
    raise FileNotFoundError("Both TheCov_dir and RascalC_dir are not found.")


base_path = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}/blinded/'.format(version)

if os.path.exists(base_path):
    data_dir = base_path
    catalog_dir = base_path
else:
    raise FileNotFoundError(f"The specified directory does not exist: {base_path}")
    

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


def get_footprints_from_data(tracer='ELG', region='GCcomb', zlims=()):
    """Return footprints (for all redshift slices), specifying redshift density nbar and area, using Y1 data."""
    from mockfactory import Catalog, RedshiftDensityInterpolator
    from desilike.observables.galaxy_clustering import CutskyFootprint
    from cosmoprimo.fiducial import DESI

    # catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.1/blinded/'

    def select_region(catalog, region):
        mask = catalog.trues()
        if region == 'NGC':
            mask &= (catalog['RA'] > 88) & (catalog['RA'] < 303)
        if region == 'SGC':
            mask &= (catalog['RA'] < 88) | (catalog['RA'] > 303)
        return catalog[mask]

    def concatenate(list_data, list_randoms, region):
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

    data_fns = [os.path.join(catalog_dir, '{}_{}_clustering.dat.fits'.format(tracer, reg)) for reg in ['NGC', 'SGC']]
    randoms_fns = [os.path.join(catalog_dir, '{}_{}_0_clustering.ran.fits'.format(tracer, reg)) for reg in ['NGC', 'SGC']]
    data_NSGC = [Catalog.read(fn) for fn in data_fns]
    randoms_NSGC = [Catalog.read(fn) for fn in randoms_fns]

    if region == 'NGC':
        data, randoms = data_NSGC[0], randoms_NSGC[0]
    elif region == 'SGC':
        data, randoms = data_NSGC[1], randoms_NSGC[1]
    elif region == 'GCcomb':
        data, randoms = concatenate(data_NSGC, randoms_NSGC, region)
    else:
        raise ValueError('Unknown region {}'.format(region))

    mpicomm = data.mpicomm
    import healpy as hp
    import mpytools as mpy
    nside = 512
    theta, phi = np.radians(90 - randoms['DEC']), np.radians(randoms['RA'])
    hpindex = hp.ang2pix(nside, theta, phi, lonlat=False)
    hpindex = mpy.gather(np.unique(hpindex), mpicomm=mpicomm, mpiroot=0)
    fsky = mpicomm.bcast(np.unique(hpindex).size if mpicomm.rank == 0 else None, root=0) / hp.nside2npix(nside)
    area = fsky * 4. * np.pi * (180. / np.pi)**2
    cosmo = DESI()
    footprints = []
    for zlim in zlims:
        # raise NotImplementedError('Still to test if tuple or list is aproprite.')
        num = int(abs(zlim[1] - zlim[0]) / 0.002 + 0.5)
        bins = np.linspace(*zlim, num=num)
        density = RedshiftDensityInterpolator(z=data['Z'], bins=bins, fsky=fsky, distance=cosmo.comoving_radial_distance, mpicomm=mpicomm)
        footprints.append(CutskyFootprint(area=area, zrange=density.z, nbar=density.nbar, cosmo=cosmo))
    return footprints


def get_template(template_name='standard', z=0.8, klim=None):

    """A simple wrapper that returns the template of interest."""

    from desilike.theories.galaxy_clustering import StandardPowerSpectrumTemplate, ShapeFitPowerSpectrumTemplate, WiggleSplitPowerSpectrumTemplate, BandVelocityPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate

    if 'standard' in template_name:
        template = StandardPowerSpectrumTemplate(z=z)
    elif 'shapefit' in template_name:
        template = ShapeFitPowerSpectrumTemplate(z=z, apmode='qisoqap' if 'qisoqap' in template_name else 'qparqper')
        # prior = {'dist': 'norm', 'loc': 1., 'scale': 0.03} if 'prior' in template_name else None
        # if 'qisoqap' in template_name and prior is not None:
        #     1./0.
        #     for param in template.init.params.select(name=['qap']):
        #         param.update(fixed=False, prior=prior)
    elif 'wigglesplit' in template_name:
        template = WiggleSplitPowerSpectrumTemplate(z=z)
    elif 'ptt' in template_name:
        template = BandVelocityPowerSpectrumTemplate(z=z, kp=np.arange(*klim))
    elif 'direct' in template_name:
        template = DirectPowerSpectrumTemplate(z=z)
        template.params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
        template.params['n_s'].update(fixed=True)
    elif 'bao' in template_name:
        template = BAOPowerSpectrumTemplate(z=z, apmode='qisoqap' if 'qisoqap' in template_name else 'qparqper', only_now=True if only_now_name else False)
        for param in template.init.params.select(name=['qpar', 'qper', 'qiso', 'qap']):
            param.update(prior={'limits': [0.9, 1.1]})
    return template


def get_theory(theory_name='velocileptors', observable_name='power', b1E=1.9, template=None, ells=(0, 2, 4)):

    """A simple wrapper that returns the theory of interest."""

    from desilike.theories.galaxy_clustering import (LPTVelocileptorsTracerPowerSpectrumMultipoles, LPTVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles, EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles,
                                                     PyBirdTracerPowerSpectrumMultipoles, PyBirdTracerCorrelationFunctionMultipoles,
                                                     DampedBAOWigglesTracerPowerSpectrumMultipoles, DampedBAOWigglesTracerCorrelationFunctionMultipoles)

    kwargs = {}
    euler = False
    if 'bird' in theory_name:
        euler = True
        kwargs.update(eft_basis='westcoast')
        Theory = PyBirdTracerPowerSpectrumMultipoles if observable_name == 'power' else PyBirdTracerCorrelationFunctionMultipoles
    elif 'velo' in theory_name:
        Theory = LPTVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'lptm' in theory_name:
        Theory = LPTMomentsVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else LPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'eptm' in theory_name:
        euler = True
        Theory = EPTMomentsVelocileptorsTracerPowerSpectrumMultipoles if observable_name == 'power' else EPTMomentsVelocileptorsTracerCorrelationFunctionMultipoles
    elif 'dampedbao' in theory_name:
        euler = True
        Theory = DampedBAOWigglesTracerPowerSpectrumMultipoles if observable_name == 'power' else DampedBAOWigglesTracerCorrelationFunctionMultipoles

    theory = Theory(template=template, **kwargs)
    # Changes to theory.params will remain whatever pipeline is built
    b1 = float(euler) + b1E - 1.
    theory.params['b1'].update(value=b1, ref={'limits': [b1 - 0.1, b1 + 0.1]})
    for param in theory.params.select(basename=['alpha6']): param.update(fixed=True)
    if 4 not in ells:
        for param in theory.params.select(basename=['alpha4', 'sn4*', 'al4_*']): param.update(fixed=True)
    if observable_name != 'power':
        #for param in theory.params.select(basename=['ce1', 'sn0', 'al*_1', 'al*_-3']): param.update(fixed=True)
        for param in theory.params.select(basename=['ce1', 'sn0', 'al*_-3']): param.update(fixed=True)
    return theory


def get_fit_setup(tracer, theory_name='velocileptors'):
    '''The function `get_fit_setup` returns the appropriate redshift limits, bias value, k limits, and s
    limits based on the input tracer and theory name.
    
    Parameters
    ----------
    tracer
        The tracer parameter represents the type of galaxy tracer being used in the analysis. It can take
    the values 'BGS', 'LRG', 'ELG', or '
    theory_name, optional
        The name of the theory being used for the fit. The default value is 'velocileptors'.
    
    Returns
    -------
        a tuple containing the following values:
    - zlim: a list of two values representing the lower and upper redshift limits
    - b0: a float representing the bias parameter
    - klim: a dictionary with keys as integers (0, 2, 4) and values as lists of three values
    representing the lower limit, upper limit, and step size for the w
    
    '''
    ells = (0, 2, 4)
    if 'bao' in theory_name: ells = (0, 2)
    if tracer.startswith('BGS'):
        zbins = [(0.1, 0.4)]
        b0 = 1.34
        smin, kmax = 35., 0.15
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.03, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    if tracer.startswith('LRG'):
        zbins = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)]
        b0 = 1.7
        smin, kmax = 30., 0.17
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.03, kmax, 0.005] for ell in ells}

        slim = {ell: [smin, 150., 4.] for ell in ells}
    if tracer.startswith('ELG'):
        zbins = [(0.8, 1.1), (0.8, 1.6), (1.1, 1.6)]
        b0 = 0.84
        smin, kmax = 25., 0.2
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.05, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    if tracer.startswith('QSO'):
        zbins = [(0.8, 2.1)]
        b0 = 1.2
        smin, kmax = 20., 0.25
        if 'bao' in theory_name: smin, kmax = 40., 0.3
        klim = {ell: [0.03, kmax, 0.005] for ell in ells}
        slim = {ell: [smin, 150., 4.] for ell in ells}
    return zbins, b0, klim, slim



def get_observable_likelihood(theory_name='velocileptors', 
                              template_name='shapefit',
                              observable_name='power',
                              tracer=None, zlim=None,solve=True, save_emulator=False,
                              emulator_fn=os.path.join(emulators_dir, '{}_{}_{}_{}_{}_{}.npy'),
                              footprint_fn=os.path.join(emulators_dir, 'footprint_{}_{}_{}.npy'),
                              rpcut=False, refine_cov=True,
                              covariance_fn=os.path.join(emulators_dir, 'covariance_{}_{}_{}_{}.npy'),
                              cosmo=None, fix_template=False):

    """Return the power spectrum likelihood, optionally computing the emulator (if ``save_emulator``)."""

    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable, TracerCorrelationFunctionMultipolesObservable, ObservablesCovarianceMatrix, CutskyFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    footprint_fn = footprint_fn.format(tracer,  zlim[0], zlim[1])
    if not os.path.isfile(footprint_fn):
        footprint = get_footprints_from_data(tracer=tracer, region='GCcomb', zlims=[zlim])[0] #WARNING: region is hardcoded to GCcomb
        footprint.save(footprint_fn)
    else:
        footprint = CutskyFootprint.load(footprint_fn)
    z = footprint.zavg

    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(tracer,  zlim[0], zlim[1], observable_name, theory_name, template_name)

    from cosmoprimo.fiducial import DESI
    fiducial = DESI()
    b1E = b0 / fiducial.growth_factor(z)

    # Load theory
    theory = get_theory(theory_name=theory_name, observable_name=observable_name, template=None, b1E=b1E, ells=klim.keys())
    if 'bao' in template_name:
        if save_emulator:
            raise ValueError('No need to build an emulator for the BAO model!')
        emulator_fn = None

    template = get_template(template_name=template_name, z=z, klim=(klim[0][0], klim[0][1] + 1e-5, klim[0][2]))
    if save_emulator or emulator_fn is None or not os.path.isfile(emulator_fn):  # No emulator available (yet)
        theory.init.update(template=template)
    else:  # Load emulator
        from desilike.emulators import EmulatedCalculator
        calculator = EmulatedCalculator.load(emulator_fn)
        theory.init.update(pt=calculator)

    for param in theory.init.params:
        if param not in template.params:
            param.update(namespace=tracer)  # set namespace for all bias parameters

    params = {}
    # data_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v0.4/blinded/'

    if observable_name == 'power':
        data = os.path.join(data_dir, 'pk', 'pkpoles_{}_GCcomb_{}_{}_default_FKP_lin'.format(tracer,  zlim[0], zlim[1]))
        wmatrix = os.path.join(data_dir, 'pk', 'wmatrix_smooth_{}_GCcomb_{}_{}_default_FKP_lin'.format(tracer,  zlim[0], zlim[1]))
        if rpcut:
            data += '_rpcut{}'.format(rpcut)
            wmatrix += '_rpcut{}'.format(rpcut)
        observable = TracerPowerSpectrumMultipolesObservable(klim=klim, data=data + '.npy', wmatrix=wmatrix + '.npy', kinlim=(0.005, 0.35), kinrebin=5, theory=theory)   # generates fake data

    if observable_name == 'corr':
        data = os.path.join(data_dir, 'xi','smu/allcounts_{}_GCcomb_{}_{}_default_FKP_lin_njack0_nran4_split20'.format(tracer,  zlim[0], zlim[1]))
        fiber_collisions = None
        if rpcut:
            data += '_rpcut{}'.format(rpcut)
            from desilike.observables.galaxy_clustering import TopHatFiberCollisionsCorrelationFunctionMultipoles
            fiber_collisions = TopHatFiberCollisionsCorrelationFunctionMultipoles(Dfc=2.5, with_uncorrelated=False, mu_range_cut=True)
        observable = TracerCorrelationFunctionMultipolesObservable(slim=slim, data=data + '.npy', wmatrix=None, fiber_collisions=fiber_collisions, theory=theory)

    covariance = ObservablesCovarianceMatrix(observable, footprints=footprint, resolution=5)
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance(**params))
    likelihood.params['{}.loglikelihood'.format(tracer)] = likelihood.params['{}.logprior'.format(tracer)] = {}
    likelihood()  # to set up k-ranges for the emulator
    #for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*', 'al*']):
    #    if param.varied: param.update(derived='.auto')
    if save_emulator:  # Compute and save emulator
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=4))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
        theory.init.update(pt=emulator.to_calculator())
    
    if refine_cov:
        if TheCov_dir is not None:
            region = 'NGCSGCcomb' # hardcoded since we are using the combined NGC and SGC footprint and the nomeclature is different in the TheCov covariance files

            covariance_fn = TheCov_dir + 'cov_gaussian_prerec_{}_{}_{}_{}.txt'.format(tracer, region,  zlim[0], zlim[1])
            if os.path.isfile(covariance_fn):
                cov = np.loadtxt(covariance_fn)
                print('\nLoading Pk covariance from {}.\n'.format(covariance_fn))
                kmin, kmax, dk = 0.0, 0.4, 0.005
                kmid = np.arange(kmin + dk/2, kmax + dk/2, dk)
                cov = cut_matrix(cov, kmid, (0, 2, 4), klim)
                likelihood.init.update(covariance=cov)
            else:
                raise ValueError('Covariance file {} not found!'.format(covariance_fn))
        elif RascalC_dir is not None:
            covariance_fn = RascalC_dir + 'xi024_{}_GCcomb_{}_{}_default_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt'.format(tracer, zlim[0], zlim[1])
            if os.path.isfile(covariance_fn):
                cov = np.loadtxt(covariance_fn)
                print('\nLoading xi024 covariance from {}.\n'.format(covariance_fn))
                smin, smax, ds = 20., 200., 4.
                smid = np.arange(smin + ds/2, smax + ds/2, ds)
                cov = cut_matrix(cov, smid, (0, 2, 4), slim)
                # print("\n covmat:{}\n".format(cov.shape))
                likelihood.init.update(covariance=cov)
            else:
                raise ValueError('Covariance file {} not found'.format(covariance_fn))
        else:
            if covariance_fn is not None:
                covariance_fn = covariance_fn.format(tracer, observable_name, theory_name, template_name)
            if os.path.isfile(covariance_fn):
                likelihood.init.update(covariance=np.load(covariance_fn))
            else:
                template_name_pk, theory_name_pk, observable_name_pk = 'shapefit', 'velocileptors', 'power'
                likelihood_pk = get_observable_likelihood(theory_name=theory_name_pk,
                                                          template_name=template_name_pk,
                                                          observable_name=observable_name_pk,
                                                          tracer=tracer,
                                                          solve=solve, 
                                                          save_emulator=False,
                                                          rpcut=rpcut,
                                                          refine_cov=False,
                                                          fix_template=True)
                from desilike.profilers import MinuitProfiler
                profiler = MinuitProfiler(likelihood_pk, seed=42)
                profiles = profiler.maximize(niterations=10)
                if profiler.mpicomm.rank == 0:
                    print(profiles.to_stats(tablefmt='pretty'))
                cov = ObservablesCovarianceMatrix(observable,
                                                  footprints=footprint,
                                                  theories=[likelihood_pk.observables[0].wmatrix.theory],
                                                  resolution=5)(**profiles.bestfit.choice(input=True))
                if covariance_fn is not None:
                    np.save(covariance_fn, cov)
                likelihood.init.update(covariance=cov)
    
    if 'direct' in template_name and cosmo is not None:  # external cosmo
        template.init.update(cosmo=cosmo)
    
    # likelihood.all_params gives access to the parameters of the likelihood pipeline
    if solve:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*', 'al*']):
            if param.varied: param.update(derived='.auto')
    if fix_template:
        for param in likelihood.varied_params:
            if param in template.params: param.update(fixed=True)
    likelihood()
    if likelihood.mpicomm.rank == 0:
        likelihood.log_info('Use analytic marginalization for {}.'.format(likelihood.all_params.names(solved=True)))

    prior = {'dist': 'norm', 'loc': 1., 'scale': 0.03} if 'prior' in template_name else None
    if 'qisoqap' in template_name and prior is not None:
        likelihood.all_params['qap'].update(prior=dict(dist='norm', loc=1., scale=0.03))
    return likelihood


def get_compressed_likelihood(chains_fn=None,
                              theory_name='velocileptors',
                              template_name='shapefit',
                              observable_name='power',
                              tracer=None,
                              zlim=None,
                              save_emulator=False,
                              emulator_fn=os.path.join(emulators_dir, '{}_{}_{}_compressed_{}_{}_{}.npy'),
                              footprint_fn=os.path.join(emulators_dir, 'footprint_{}_{}_{}.npy'),
                              cosmo=None):

    """Return the likelihood of compressed parameters, optionally computing the emulator (if ``save_emulator``)."""

    from desilike.observables.galaxy_clustering import CutskyFootprint
    from desilike.likelihoods import ObservablesGaussianLikelihood

    footprint_fn = footprint_fn.format(tracer, zlim[0], zlim[1])
    z = CutskyFootprint.load(footprint_fn).zavg

    if emulator_fn is not None:
        emulator_fn = emulator_fn.format(tracer, zlim[0], zlim[1], observable_name, theory_name, template_name)

    if save_emulator or emulator_fn is None:
        from desilike.samples import Chain
        chain = Chain.concatenate([Chain.load(chain_fn).remove_burnin(0.5) for chain_fn in chains_fn])  # MCMC of compressed parameters
        from desilike.observables.galaxy_clustering import ShapeFitCompressionObservable, WiggleSplitCompressionObservable, BAOCompressionObservable
        if 'shapefit' in template_name:
            quantities = ['qpar', 'qper', 'df', 'dm']
            observable = ShapeFitCompressionObservable(data=chain, covariance=chain, z=z, quantities=quantities, dfextractor='fsigmar' if 'sigma8' in template_name else 'shapefit')
        elif 'wigglesplit' in template_name:
            quantities = ['qbao', 'qap', 'df', 'dm']
            observable = WiggleSplitCompressionObservable(data=chain, covariance=chain, z=z, quantities=quantities)
        elif 'bao' in template_name:
            quantities = ['qpar', 'qper']
            observable = BAOCompressionObservable(data=chain, covariance=chain, z=z, quantities=quantities)
    else:
        from desilike.emulators import EmulatedCalculator
        observable = EmulatedCalculator.load(emulator_fn)

    likelihood = ObservablesGaussianLikelihood(observable)
    likelihood.params['{}.loglikelihood'.format(tracer)] = likelihood.params['{}.logprior'.format(tracer)] = {}
    if 'omega_b' in likelihood.all_params:
        likelihood.all_params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037})
    likelihood()

    if save_emulator:
        # A bit of emulation, to speed up inference
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(observable, engine=TaylorEmulatorEngine(order=3))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)

    return likelihood


def samples_fn(outdir,
               base='chain',
               compressed=False,
               theory_name='velocileptors',
               template_name='shapefit',
               observable_name='power',
               tracer=None,
               zlim=None,
               rpcut=False,
               i=None,
               outfile_format=None):
    
    zmin, zmax = zlim
    fn = '_'.join([base, tracer + f'_{zmin}_{zmax}', 'compressed_' + observable_name if compressed else observable_name, theory_name, template_name])
    if rpcut:
        fn += '_rpcut{}'.format(rpcut)
    if i is not None:
        fn += '_{:d}'.format(i)
    fn += '.npy' if outfile_format is None else '.' + outfile_format
    return os.path.join(outdir, fn)


if __name__ == '__main__':
    import time
    time0 = time.time()

    import argparse
    parser = argparse.ArgumentParser(description='Y1 mocks full shape')
    parser.add_argument('--tracer', type=str, nargs='*', required=False, default=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], choices=['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO'], help='Tracer')
    parser.add_argument('--template', type=str, required=False, default='shapefit', choices=['direct', 'shapefit', 'shapefit-qisoqap', 'shapefit-qisoqap-prior' , 'wigglesplit', 'bao', 'bao-qisoqap'], help='Template')
    parser.add_argument('--only_now', action='store_true', required=False, help='no-wiggle only')
    parser.add_argument('--theory', type=str, required=False, default='velocileptors', choices=['velocileptors', 'pybird', 'dampedbao'], help='Theory')
    parser.add_argument('--observable', type=str, required=False, default='power', choices=['power', 'corr'], help='Observable')
    parser.add_argument('--rpcut', type=float, required=False, default=None, help='rp-cut in measurement units')
    parser.add_argument('--todo', type=str, nargs='*', required=False, default=['emulator', 'sampling'], choices=['post', 'emulator', 'profiling', 'sampling', 'bindings', 'inference'], help='To do')
    parser.add_argument('--outdir', type=str, required=False, default=os.path.join(os.getenv('SCRATCH'), 'test_y1_full_shape/{}'.format(version)), help='Where to save results')
    args = parser.parse_args()

    from desilike import setup_logging
    from desilike.samplers import EmceeSampler, ZeusSampler
    from desilike.bindings import CobayaLikelihoodGenerator, CosmoSISLikelihoodGenerator, MontePythonLikelihoodGenerator

    setup_logging()

    print(f'\nargs = {args}\n')
    outdir = args.outdir
    tracers = args.tracer
    template_name = args.template
    only_now_name = args.only_now
    theory_name = args.theory
    observable_name = args.observable
    todo = args.todo
    post = 'post' in args.todo
    rpcut = args.rpcut

    # Explicitly print all the selected settings at the beginning
    print(f"Starting analysis with the following settings:")
    print(f"Tracer: {args.tracer}")
    print(f"Template: {args.template}")
    print(f"Only Now: {'Yes' if args.only_now else 'No'}")
    print(f"Theory: {args.theory}")
    print(f"Observable: {args.observable}")
    print(f"RP Cut: {args.rpcut if args.rpcut is not None else 'No cut specified'}")
    print(f"Tasks to Perform: {', '.join(args.todo)}")
    print(f"Output Directory: {args.outdir}")

    if only_now_name==True and template_name not in ('bao', 'bao-qisoqap'):
        raise ValueError('The --only_now argument can only be used with the bao or bao-qisoqap templates.')

    nchains = 8
    kw_fn = dict(template_name=template_name, theory_name=theory_name, observable_name=observable_name, rpcut=rpcut)
    
    def get_likelihood(compressed=False, tracer='LRG', zlim=None, *args, **kwargs):
        print(f"zlim: {zlim}")
        if compressed:
            chains_fn = [samples_fn(outdir, i=i, base='chain', tracer=tracer, zlim=zlim, **kw_fn) for i in range(nchains)]
            return get_compressed_likelihood(chains_fn, tracer=tracer, zlim=zlim, **kw_fn, **kwargs)
        return get_observable_likelihood(tracer=tracer, zlim=zlim, *args, **kw_fn, **kwargs)
    
    for tracer in tracers:
        zbins = get_fit_setup(tracer, theory_name=theory_name)[0] # this is zlim, b0, klim, slim - This defined fit_setup as a global variable scope.

        for zlim in zbins:
            print(f'\ntracer = {tracer}, zlim = {zlim}\n')
            b0, klim, slim = get_fit_setup(tracer, theory_name=theory_name)[1:4]           

            if 'emulator' in todo:
                likelihood = get_likelihood(compressed=post, tracer=tracer, zlim=zlim, save_emulator=True)
                likelihood.mpicomm.barrier()  # just to wait all processes are done

            if 'profiling' in todo:
                time2 = time.time()
                from desilike.profilers import MinuitProfiler, ScipyProfiler
                likelihood = get_likelihood(compressed=post, tracer=tracer, zlim=zlim, solve=True)
                profiler = MinuitProfiler(likelihood, seed=42, save_fn=samples_fn(outdir, base='profiles'+'_only_now' if only_now_name else 'profiles', compressed=post, tracer=tracer, zlim=zlim, **kw_fn))
                profiles = profiler.maximize(niterations=10)
                if 'qiso' in template_name:
                    profiles = profiler.interval('qiso')
                    profiles = profiler.profile('qiso')

                profiles.bestfit.choice(input=True)
                observable = likelihood.observables[0]
                observable.plot(fn=samples_fn(outdir, base='poles_bestfit'+'_only_now' if only_now_name else 'poles_bestfit', compressed=post, tracer=tracer, zlim=zlim, **kw_fn, outfile_format='png'))
                profiles.to_stats(fn=samples_fn(outdir, base='profiles'+'_only_now' if only_now_name else 'profiles', compressed=post, tracer=tracer, zlim=zlim, **kw_fn, outfile_format='stats'))
                if profiler.mpicomm.rank == 0:
                    print(profiles.to_stats(tablefmt='pretty'))
                # print out time taken
                print(f'## time taken for profiling {(time.time()-time2) / 60:.2f} mins')

            if 'sampling' in todo:
                # start timer for this if statement
                from desilike.samples import Chain, plotting
                time1 = time.time()
                # set up the likelihood
                likelihood = get_likelihood(compressed=post, tracer=tracer, zlim=zlim)
                save_fn = [samples_fn(outdir, i=i, base='chain', compressed=post, tracer=tracer, **kw_fn) for i in range(nchains)]
                chains = nchains
                sampler = EmceeSampler(likelihood, chains=chains, nwalkers=40, seed=42, save_fn=save_fn)
                sampler.run(min_iterations=2000, check={'max_eigen_gr': 0.03})
                chain_ = Chain.concatenate([Chain.load(samples_fn(outdir, i=i, compressed=post, tracer=tracer, zlim=zlim, **kw_fn)).remove_burnin(0.5)[::10] for i in range(nchains)])  # load and concatenate all chains
                likelihood(**chain_.choice(index='argmax', input=True))  # compute the likelihood at the "bestfit of the chain" parameters
                likelihood.observables[0].plot(show=True)
                observable = likelihood.observables[0]
                observable.plot(fn=samples_fn(outdir, base='poles_bestfit', compressed=post, tracer=tracer, zlim=zlim, **kw_fn, outfile_format='png'))
                # print out time taken
                print(f'## time taken for sampling: {(time.time() - time1) / 60:.2f} mins')

            if 'inference' in todo:
                likelihood = sum(get_likelihood(compressed='direct' not in template_name, tracer=tracer, zlim=zlim) for tracer in tracers)
                for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*', 'al*']):
                    if param.varied: param.update(derived='.auto')
                chains = nchains
                save_fn = [os.path.join(outdir, 'chain_{}_{:d}.npy'.format('_'.join(tracers), ichain)) for ichain in range(nchains)]
                # To restart:
                chains = save_fn
                sampler = EmceeSampler(likelihood, chains=chains, nwalkers=40, seed=42, save_fn=save_fn)
                sampler.run(check={'max_eigen_gr': 0.02})

            Likelihoods = [get_observable_likelihood] * len(tracers)
            name_like = ['DESIFullShape{}{}Likelihood'.format(template_name.capitalize(), tracer) for tracer in tracers]
            kw_like = []
            for tracer in tracers:
                kw = {'tracer': tracer, 'theory_name': theory_name, 'template_name': template_name, 'observable_name': observable_name, 'cosmo': 'external'}
                kw_like.append(kw)
            if 'bindings' in todo:
                setup_logging('info')
                overwrite = True
                CobayaLikelihoodGenerator()(Likelihoods, name_like=name_like, kw_like=kw_like, overwrite=overwrite)
                CosmoSISLikelihoodGenerator()(Likelihoods, name_like=name_like, kw_like=kw_like, overwrite=overwrite)
                MontePythonLikelihoodGenerator()(Likelihoods, name_like=name_like, kw_like=kw_like, overwrite=overwrite)

    print(f'## total time taken: {(time.time() - time0) / 60:.2f} mins')
