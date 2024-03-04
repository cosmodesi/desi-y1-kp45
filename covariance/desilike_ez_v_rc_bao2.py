import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.bao import DampedBAOWigglesTracerCorrelationFunctionMultipoles
from desilike.observables.galaxy_clustering import TracerCorrelationFunctionMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike import Fisher
from desilike.profilers import MinuitProfiler
from desilike import setup_logging
from desilike.samples import plotting
from desilike.install import Installer
from desilike.samplers import ZeusSampler, PocoMCSampler
from helpers import read_xis_pycorr, samples_to_covariance, mask_fit_range
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
from desilike.samples import plotting, Chain
comm = MPI.COMM_WORLD

def read_xis_pycorr(list, list_b = None, ells = (0,2,4)):
    import os
    from tqdm import tqdm
    from pycorr import TwoPointCorrelationFunction
    assert len(list) > 0
    if list_b is not None:
        assert len(list) == len(list_b)
        print("WARNING: Assuming both lists have been passed with the right order for combinations.")
    
    xis = []
    for i, f in tqdm(enumerate((list))):
        result = TwoPointCorrelationFunction.load(f)[::4]
        if list_b is not None: result = result.normalize() + (TwoPointCorrelationFunction.load(list_b[i])[::4]).normalize()
        s, xiell = result(ells=ells, return_sep=True)
        xis.append(result)
    return xis



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cov_type", type = str, default = 'mock')
    parser.add_argument("-conv", type = str, default = 'sym')
    parser.add_argument("-chunk", type = int, default = 0)
    parser.add_argument("-nchunks", type = int, default = 1)
    parser.add_argument("-seed", type = int, default = 42)
    parser.add_argument("-rerun", action = 'store_true')
    parser.add_argument("-rescaled", action = 'store_true')
    parser.add_argument("-fit_mean", action = 'store_true')
    parser.add_argument("-prior_type", type = str, default = 'fixed')
    parser.add_argument("-chain_input", type = str)
    args = parser.parse_args()
    redshift = 0.8
    cap = 'NGC'
    tracer = 'LRG'
    zmin, zmax, zeff = 0.8, 1.1, 0.95
    growth_rate = 0.830
    smin, smax, ds = 40, 160, 4
    
    
    

    setup_logging()
    if args.cov_type == 'mock':
        covariance = None
    elif args.cov_type == 'analytic':
        if not args.rescaled:
            cov_fn = f"/global/cfs/cdirs//desi/users/mrash/RascalC/AbacusSummit/CutSky/Y5/xi024_EZmocks_LRG_main_rec{args.conv}_{cap.upper()}_{zmin:.1f}_{zmax:.1f}_FKP_lin4_s20-200_cov_RascalC_Gaussian.txt"
        else:
            cov_fn = f"/global/cfs/cdirs//desi/users/mrash/RascalC/AbacusSummit/CutSky/Y5/xi024_EZmocks_LRG_main_rec{args.conv}_{cap.upper()}_{zmin:.1f}_{zmax:.1f}_FKP_lin4_s20-200_cov_RascalC_rescaled1.06.txt"
        cov_rc = np.loadtxt(cov_fn)
        cov_rc = cov_rc[:2*cov_rc.shape[0]//3,:][:,:2*cov_rc.shape[0]//3]
        sedges = np.arange(20, 204, 4)
        s = np.tile(sedges[:-1] + 0.5 * (np.diff(sedges)), 2)
        _, _, _, cov_rc_obs = mask_fit_range(smin, smax, s, s, cov_rc, cov_rc)
        covariance = cov_rc_obs
    else:
        raise NotImplementedError
    
    if args.chain_input is None:
        output_dir = f"data/desilike_{args.cov_type}_{args.conv}_bao2_minuit_xi/"
    else:
        output_dir = f"data/desilike_{args.cov_type}_{args.conv}_bao2_minuit_xi_gauss/"
    
    
    os.makedirs(output_dir, exist_ok = True)
    
    if args.conv != 'pre':    
        ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Xi/Post/forero/fiducial_settings/z0.800/"
        ez_list = [f"{ez_dir}/cutsky_{tracer}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{i}/{cap.upper()}/{zmin:.1f}z{zmax:.1f}f{growth_rate:.3f}/{args.conv}_tpcf.pkl.npy" for i in range(1, 1001)]
    else: 
        ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Xi/Pre/forero/z0.800/"
        ez_list = [f"{ez_dir}/cutsky_{tracer}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{i}/{cap.upper()}/{zmin:.1f}z{zmax:.1f}f{growth_rate:.3f}/{args.conv}_tpcf.pkl.npy" for i in range(1, 1001)]
        
        
       
    
    id_list = np.array_split(range(0, 1000), args.nchunks)[args.chunk]
    
    for j, i in enumerate(id_list):
        # Or StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate
        template = BAOPowerSpectrumTemplate(z=redshift, fiducial='DESI', apmode = 'qisoqap')  # effective redshift
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, mode = "" if args.conv == 'pre' else f'rec{args.conv}', )
        
        output_fn = f"{output_dir}/minuit_prof_{i:03d}.txt" if not args.fit_mean else f"{output_dir}/minuit_prof_mean.txt"
        if os.path.isfile(output_fn + ".npy") and not args.rerun and not args.fit_mean: continue
        print(f"==> Starting for id {i}", flush=True)
        observable = TracerCorrelationFunctionMultipolesObservable(data = ez_list[i] if not args.fit_mean else np.sum(list(map(lambda x: x.normalize(), read_xis_pycorr(ez_list, ells = (0,2))))),  
                                                                    covariance = covariance if args.cov_type == 'analytic' else ez_list,
                                                                    slim={0: [smin, smax, ds], 2: [smin, smax, ds]},  # k-limits, between 0.01 and 0.2 h/Mpc with 0.005 h/Mpc step size for ell = 0, 2
                                                                    theory=theory)  # previously defined theory
        
        likelihood = ObservablesGaussianLikelihood(observables=[observable],
                                                   scale_covariance = 1./len(ez_list) if args.fit_mean else 1.,)#, mpicomm = comm)
        for param in likelihood.all_params.select(basename='al*_*'): 
            param.update(derived='.auto')
            
        if args.chain_input is None:
            for param in likelihood.all_params.select(basename="sigma*"):
                param.update(fixed = False, ref=dict(limits=(0,20.)))
        else:
            chain = Chain.load(args.chain_input).remove_burnin(0.5)
            print(chain.choice(input=True))
            
            for param in likelihood.all_params.select(basename="sigma*"):
                param.update(fixed = False, ref=dict(dist='norm', loc=chain.choice(input=True)[param.name], scale=2.))
        
        
        #installer = Installer(user=True)
        #installer(MinuitProfiler)
        grid = np.linspace(0.8, 1.2, 31)
        try:
            os.remove(output_fn+".npy")
        except: 
            pass
        profiler = MinuitProfiler(likelihood, save_fn = output_fn+".npy", mpicomm = comm, seed = args.seed)
        profiles = profiler.maximize(niterations=3)
        profiles = profiler.interval(params=['qiso'])
        # To print relevant information
        print(profiles.to_stats(tablefmt='pretty', fn = output_fn))
        #profiles.save(output_fn)
        #if j == 1:
        likelihood(**profiler.profiles.bestfit.choice(input=True))
        observable.plot()
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_rc_bao2_bestfit.png")
        
        
        if j == 0:
            # Estimate Fisher matrix for one realization since derivatives don't depend on data
            from desilike import Fisher
            fisher = Fisher(likelihood)
            #param_fish = fisher(**profiler.profiles.bestfit.choice(params=likelihood.varied_params))
            param_fish = fisher(**profiler.profiles.bestfit.choice(input=True))
            param_fish.save(f"{output_dir}/fisher_bao2.npy")
            param_cov = param_fish.covariance()
            print(param_fish.to_stats(tablefmt = 'pretty'))
        
        
        if args.fit_mean: 
            profiles = profiles.choice()
            chain_filename = f"{output_dir}/chain_mean_bao2_{args.prior_type}_0.npy"
            for param in likelihood.all_params.select(basename="sigma*"):
                print("Setting prior for", param.name)
                print(profiles.bestfit[param.name])
                if args.prior_type == 'fixed':
                    param.update(fixed = True, value=profiles.bestfit[param.name])
                elif args.prior_type == 'gauss':
                    param.update(fixed = False, ref=dict(dist='norm', loc=profiles.bestfit[param.name], scale=profiles.error[param.name]))
                elif args.prior_type == 'flat':
                    param.update(fixed = False, ref=dict(limits=(0,20.), dist='uniform'))
                #param.update(fixed = False, ref=dict(limits=(0,20.), dist='norm', loc=profiles.bestfit[param.name], scale=profiles.error[param.name]))
            
            if not os.path.isfile(chain_filename) or args.rerun:
                sampler = ZeusSampler(likelihood, save_fn=chain_filename, seed=42, mpicomm=comm, nwalkers = '6*ndim')#, chains = f'{output_dir}/chain_mean_bao2_*.npy')
                #sampler = DynamicDynestySampler(likelihood, save_fn=f'{output_dir}/chain_mean_bao2_*.npy', seed=42, mpicomm=comm)#, chains = f'{output_dir}/chain_mean_bao2_*.npy')
                sampler.run(check={'max_eigen_gr': 0.3})
                #sampler.run()
            
            chain = Chain.load(chain_filename).remove_burnin(0.5)
            likelihood(**chain.choice(input=True))
            observable.plot()
            plt.gcf()
            plt.savefig("plots/desilike_ez_v_rc_bao2_bestfit.png")
            print(chain.to_stats(tablefmt='pretty'))
            plotting.plot_triangle(chain, params = ['qiso', 'qap', 'b1'], filled = True, title_limit = 1)
            plt.gcf()
            plt.savefig("plots/desilike_ez_v_rc_bao2_corner.png", dpi=100)
            
            
            break