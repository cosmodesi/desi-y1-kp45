import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate, LPTVelocileptorsTracerPowerSpectrumMultipoles, ShapeFitPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.bao import DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import Fisher
from desilike.profilers import MinuitProfiler
from desilike import setup_logging
from desilike.samples import plotting
from desilike.install import Installer
from desilike.samplers import ZeusSampler, PocoMCSampler
from desilike.emulators import EmulatedCalculator
from desilike.samplers import ZeusSampler, DynamicDynestySampler
from helpers import read_xis_pycorr, samples_to_covariance, mask_fit_range
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD


def read_pks_pypower(list, list_b = None, slice = slice(0,None,1), ells = (0,2,4)):
    import os
    from tqdm import tqdm
    from pypower import CatalogFFTPower, BaseMatrix
    assert len(list) > 0
    if list_b is not None:
        assert len(list) == len(list_b)
    print("WARNING: Assuming both lists have been passed with the right order for combinations.")
    
    xis = []
    thek = None
    for i, f in enumerate(tqdm(list)):
        result = CatalogFFTPower.load(f).poles
        if list_b is not None: result += CatalogFFTPower.load(list_b[i]).poles
        xis.append(result)
    return xis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cov_type", type = str, default = 'mock')
    parser.add_argument("-conv", type = str, default = 'sym')
    parser.add_argument("-cap", type = str, default = 'NGC')
    parser.add_argument("-prior_type", type = str, default = 'fixed')
    parser.add_argument("-chunk", type = int, default = 0)
    parser.add_argument("-nchunks", type = int, default = 1)
    parser.add_argument("-seed", type = int, default = 42)
    parser.add_argument("-rerun", action = 'store_true')
    parser.add_argument("-rerun_emu", action = 'store_true')
    parser.add_argument("-fit_mean", action = 'store_true')
    args = parser.parse_args()
    redshift = 0.8
    cap = args.cap
    tracer = 'LRG'
    zmin, zmax, zeff = 0.8, 1.1, 0.95
    growth_rate = 0.830
    kmin, kmax, dk = 0.01, 0.2, 0.005
    
    if args.cov_type == 'mock':
        covariance = None
    elif args.cov_type == "analytic":
        if args.conv == 'sym':
            covariance_fn = f"/global/cfs/cdirs//desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Post/covariances/cov_gaussian_{args.conv}_{tracer}_{args.cap.upper()}_{zmin:.1f}_{zmax:.1f}.txt"
        elif args.conv == 'pre':
            covariance_fn = f"/global/cfs/cdirs//desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Pre/covariances/cov_gaussian_{args.conv}_{tracer}_{args.cap.upper()}_{zmin:.1f}_{zmax:.1f}.txt"
        elif args.conv == 'iso':
            covariance_fn = f"/global/cfs/cdirs//desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Post/covariances/cov_gaussian_{args.conv}_{tracer}_{args.cap.upper()}_{zmin:.1f}_{zmax:.1f}.txt"
        covariance = np.loadtxt(covariance_fn)
        edges = np.arange(0, 0.4005, 0.005)
        k_centers = 0.5 * (edges[1:] + edges[:-1])
        k_centers = np.concatenate([k_centers]*3, axis=-1)
        k_center_mask = (k_centers < kmax) & (k_centers > kmin)
        k_cov_mask = k_center_mask[:,None] & k_center_mask[None,:]
        covariance = covariance[k_cov_mask].reshape(int(np.sqrt(k_cov_mask.sum())), -1)
        covariance = covariance[:2*covariance.shape[0]//3,:][:,:2*covariance.shape[0]//3]
        print(covariance.shape)
    win_dir = "/global/cfs/projectdirs/desi/users/jerryou/MockChallenge/y1_mockchallenge/reconstruction/AbacusSummit/cutsky/LRG/window_fun/"
    full_window = f"{win_dir}/wideangle_windowmatrix_cutsky_LRG_{cap.upper()}_z0.800_AbacusSummit_base_c000_ph000_{zmin}z{zmax}.npy"
    
    output_dir = f"data/desilike_{args.cov_type}_{args.conv}_shapefit_minuit_pk/"
    os.makedirs(output_dir, exist_ok = True)
    if args.conv != 'pre':    
        ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Pk/Post/forero/fiducial_settings/dk0.005/z0.800/"
        ez_list = [f"{ez_dir}/cutsky_{tracer}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{i}/{cap.upper()}/{zmin:.1f}z{zmax:.1f}f{growth_rate:.3f}/{args.conv}_pk.pkl.npy" for i in range(1, 1001)]
    else: 
        ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Pk/Pre/forero/dk0.005/z0.800/"
        ez_list = [f"{ez_dir}/cutsky_{tracer}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{i}/{cap.upper()}/{zmin:.1f}z{zmax:.1f}f{growth_rate:.3f}/{args.conv}_pk.pkl.npy" for i in range(1, 1001)]
    
        
    setup_logging()
    
        
    
    
    
    id_list = np.array_split(range(0, 1000), args.nchunks)[args.chunk]
    
    
    emulator_fn = f"{output_dir}/emulator.pkl.npy"
    if not os.path.isfile(emulator_fn) or args.rerun_emu:
        print("Did not find", emulator_fn)
        print("Training emulator", flush=True)
        template = ShapeFitPowerSpectrumTemplate(z=redshift, fiducial='DESI', apmode = 'qisoqap')  # effective redshift
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, freedom = "max")
        observable = TracerPowerSpectrumMultipolesObservable(data=ez_list[0], 
                                                             covariance=covariance if args.cov_type == 'analytic' else ez_list, 
                                                             klim={0: [kmin, kmax, dk], 2: [kmin, kmax, dk]}, 
                                                             theory=theory,
                                                             wmatrix = full_window,
                                                            kinlim = (0., 0.5),
                                                            ellsin = (0,2),
                                                             )
        likelihood = ObservablesGaussianLikelihood(observable)

        likelihood()  # to set up k-ranges for the emulator
        from desilike.emulators import Emulator, TaylorEmulatorEngine
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=3))
        emulator.set_samples()
        emulator.fit()
        emulator.save(emulator_fn)
    
    
    #from desilike.install import Installer
    #installer = Installer(user=True)
    #installer(DynamicDynestySampler)
    #installer(ZeusSampler)
    for j, i in enumerate(id_list):
        
        # Or StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate
        
        theory = LPTVelocileptorsTracerPowerSpectrumMultipoles(pt = EmulatedCalculator.load(emulator_fn), freedom = "max")
        
        output_fn = f"{output_dir}/minuit_prof_{i:03d}.txt" if not args.fit_mean else f"{output_dir}/minuit_prof_mean.txt"
        if os.path.isfile(output_fn + ".npy") and not args.rerun and not args.fit_mean: continue
        print(f"==> Starting for id {i}", flush=True)
        observable = TracerPowerSpectrumMultipolesObservable(data=ez_list[i] if not args.fit_mean else np.sum(read_pks_pypower(ez_list, ells = (0,2))),  
                                                            covariance = covariance if args.cov_type == 'analytic' else ez_list,
                                                            klim={0: [kmin, kmax, dk], 2: [kmin, kmax, dk]},  # k-limits, between 0.01 and 0.2 h/Mpc with 0.005 h/Mpc step size for ell = 0, 2
                                                            theory=theory,
                                                            wmatrix = full_window,
                                                            kinlim = (0., 0.5),
                                                            ellsin = (0,2),
                                                            )  # previously defined theory
        
        likelihood = ObservablesGaussianLikelihood(observables=[observable],
                                                   scale_covariance = 1./len(ez_list) if args.fit_mean else 1.,)
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*']):
            param.update(derived='.auto')
        #if j == 1:
        observable.plot_covariance_matrix(show=True)
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_tc_shapefit_cov.png")
             
        
        #installer = Installer(user=True)
        #installer(MinuitProfiler)
        grid = np.linspace(0.8, 1.2, 31)
        if os.path.isfile(output_fn+".npy"): os.remove(output_fn+".npy")
        profiler = MinuitProfiler(likelihood, save_fn = output_fn+".npy", seed = args.seed, mpicomm = comm)
        profiles = profiler.maximize(niterations=3)
        profiles = profiler.interval(params=['qiso', 'qap'])
        # To print relevant information
        print(profiles.to_stats(tablefmt='pretty', fn = output_fn))
        #profiles.save(output_fn)
        #if j == 1:
        likelihood(**profiler.profiles.bestfit.choice(input=True))
        observable.plot()
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_tc_shapefit_bestfit.png")
        
             
        if args.fit_mean: 
            profiles = profiles.choice()
            chain_filename = f"{output_dir}/chain_mean_shapefit_{args.prior_type}_0.npy"
            for param in likelihood.varied_params.select(basename="b*"):
                print("Setting prior for", param.name)
                print(profiles.bestfit[param.name])
                if args.prior_type == 'fixed':
                    param.update(fixed = True, value=profiles.bestfit[param.name], prior=dict(dist='norm', loc=profiles.bestfit[param.name], scale=profiles.error[param.name]))
                elif args.prior_type == 'gauss':
                    param.update(fixed = False, ref=dict(dist='norm', loc=profiles.bestfit[param.name], scale=profiles.error[param.name]))
                elif args.prior_type == 'flat':
                    param.update(fixed = False, ref=dict(limits=(-10,10.), dist='uniform'))
            if not os.path.isfile(chain_filename) or args.rerun:
                sampler = ZeusSampler(likelihood, save_fn=chain_filename, seed=42, nwalkers = "6*ndim")#, chains = f'{output_dir}/chain_mean_shapefit_*.npy')
                #sampler = DynamicDynestySampler(likelihood, save_fn=f'{output_dir}/chain_mean_shapefit_*.npy', seed=42)#, chains = f'{output_dir}/chain_mean_shapefit_*.npy')
                sampler.run(check={'max_eigen_gr': 0.3})
                #sampler.run()
            from desilike.samples import plotting, Chain
            chain = Chain.load(chain_filename).remove_burnin(0.5)
            likelihood(**chain.choice(input=True))
            observable.plot()
            plt.gcf()
            plt.savefig("plots/desilike_ez_v_tc_shapefit_bestfit.png")
            print(chain.to_stats(tablefmt='pretty'))
            plotting.plot_triangle(chain, params = ['qiso', 'qap', 'dm', 'df'], filled = True, title_limit = 1)
            plt.gcf()
            plt.savefig("plots/desilike_ez_v_tc_shapefit_corner.png", dpi=100)
            
            
            break
        
        
        
       
        
        