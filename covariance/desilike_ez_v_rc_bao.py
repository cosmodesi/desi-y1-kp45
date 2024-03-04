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
comm = MPI.COMM_WORLD





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
        cov_rc = cov_rc[:cov_rc.shape[0]//3,:][:,:cov_rc.shape[0]//3]
        sedges = np.arange(20, 204, 4)
        s = sedges[:-1] + 0.5 * (np.diff(sedges))
        _, _, _, cov_rc_obs = mask_fit_range(smin, smax, s, s, cov_rc, cov_rc)
        covariance = cov_rc_obs
    else:
        raise NotImplementedError
    
    
    output_dir = f"data/desilike_{args.cov_type}_{args.conv}_bao_minuit_xi/"
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
        template = BAOPowerSpectrumTemplate(z=redshift, fiducial='DESI', apmode = 'qiso')  # effective redshift
        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(template=template, mode = "" if args.conv == 'pre' else f'rec{args.conv}', )
        
        output_fn = f"{output_dir}/minuit_prof_{i:03d}.txt"
        if os.path.isfile(output_fn + ".npy") and not args.rerun: continue
        print(f"==> Starting for id {i}", flush=True)
        observable = TracerCorrelationFunctionMultipolesObservable(data=ez_list[i],  # path to data, *pypower* power spectrum measurement, array, or dictionary of parameters where to evaluate the theory to take as a mock data vector
                                                                    covariance = covariance if args.cov_type == 'analytic' else ez_list,
                                                                    slim={0: [smin, smax, ds]},  # k-limits, between 0.01 and 0.2 h/Mpc with 0.005 h/Mpc step size for ell = 0, 2
                                                                    theory=theory)  # previously defined theory
        
        likelihood = ObservablesGaussianLikelihood(observables=[observable])#, mpicomm = comm)
        for param in likelihood.all_params.select(basename='al*_*'): 
            param.update(derived='.auto')
        [p.update(derived='{sigmapar}') for p in likelihood.all_params.select(name="sigmaper")]
        for param in likelihood.all_params.select(basename="sigma*"):
            param.update(fixed = False, ref=dict(limits=(0,20.)))
        
        #if j == 1:
        observable.plot_covariance_matrix(show=True)
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_rc_bao_cov.png")
             
        
        #installer = Installer(user=True)
        #installer(MinuitProfiler)
        grid = np.linspace(0.8, 1.2, 31)
        if os.path.isfile(output_fn+".npy"): os.remove(output_fn+".npy")
        profiler = MinuitProfiler(likelihood, save_fn = output_fn+".npy", mpicomm = comm, seed = args.seed)
        profiles = profiler.maximize(niterations=3)
        profiles = profiler.interval(params=['qiso'])
        # To print relevant information
        print(profiles.to_stats(tablefmt='pretty', fn = output_fn))
        #profiles.save(output_fn)
        #if j == 1:
        likelihood(**profiler.profiles.bestfit.choice(params=likelihood.varied_params))
        observable.plot()
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_rc_bao_bestfit.png")
        