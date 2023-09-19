import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')
from desilike.theories.galaxy_clustering import ShapeFitPowerSpectrumTemplate, StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate, BAOPowerSpectrumTemplate
from desilike.theories.galaxy_clustering.bao import DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
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
    parser.add_argument("-cap", type = str, default = 'NGC')
    parser.add_argument("-chunk", type = int, default = 0)
    parser.add_argument("-nchunks", type = int, default = 1)
    parser.add_argument("-seed", type = int, default = 42)
    parser.add_argument("-rerun", action = 'store_true')
    parser.add_argument("-prior_type", type = str, default = 'fixed')
    args = parser.parse_args()
    redshift = 0.8
    cap = args.cap
    tracer = 'LRG'
    zmin, zmax, zeff = 0.8, 1.1, 0.95
    growth_rate = 0.830
    kmin, kmax, dk = 0.01, 0.3, 0.005
    
    if args.cov_type == 'mock':
        covariance = None
    elif args.cov_type == "analytic":
        if args.conv == 'sym':
            covariance_fn = f"/global/cfs/cdirs//desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Post/covariances/cov_gaussian_{args.conv}_{tracer}_{args.cap.upper()}_{zmin:.1f}_{zmax:.1f}.txt"
        elif args.conv == 'pre':
            covariance_fn = f"/global/cfs/cdirs//desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Pre/covariances/cutsky_prerec_{tracer}_{args.cap.upper()}_{zmin:.1f}z{zmax:.1f}_cov_analytic_gaussian.txt"
        elif args.conv == 'iso':
            covariance_fn = f"/global/cfs/cdirs//desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/LRG/Pk/Post/covariances/cov_gaussian_{args.conv}_{tracer}_{args.cap.upper()}_{zmin:.1f}_{zmax:.1f}.txt"
        covariance = np.loadtxt(covariance_fn)
        edges = np.arange(0, 0.4005, 0.005)
        k_centers = 0.5 * (edges[1:] + edges[:-1])
        k_centers = np.concatenate([k_centers]*3, axis=-1)
        k_center_mask = (k_centers < kmax) & (k_centers > kmin)
        k_cov_mask = k_center_mask[:,None] & k_center_mask[None,:]
        covariance = covariance[k_cov_mask].reshape(int(np.sqrt(k_cov_mask.sum())), -1)
        covariance = covariance[:covariance.shape[0]//3,:][:,:covariance.shape[0]//3]
        print(covariance.shape)
    win_dir = "/global/cfs/projectdirs/desi/users/jerryou/MockChallenge/y1_mockchallenge/reconstruction/AbacusSummit/cutsky/LRG/window_fun/"
    full_window = f"{win_dir}/wideangle_windowmatrix_cutsky_LRG_{cap.upper()}_z0.800_AbacusSummit_base_c000_ph000_{zmin}z{zmax}.npy"
    
    output_dir = f"data/desilike_{args.cov_type}_{args.conv}_bao_minuit_pk/"
    os.makedirs(output_dir, exist_ok = True)
    if args.conv != 'pre':    
        ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Pk/Post/forero/fiducial_settings/dk0.005/z0.800/"
        ez_list = [f"{ez_dir}/cutsky_{tracer}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{i}/{cap.upper()}/{zmin:.1f}z{zmax:.1f}f{growth_rate:.3f}/{args.conv}_pk.pkl.npy" for i in range(1, 1001)]
    else: 
        ez_dir = f"/global/cfs/cdirs/desi/cosmosim/KP45/MC/Clustering/EZmock/CutSky_6Gpc/{tracer}/Pk/Pre/forero/dk0.005/z0.800/"
        ez_list = [f"{ez_dir}/cutsky_{tracer}_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed{i}/{cap.upper()}/{zmin:.1f}z{zmax:.1f}f{growth_rate:.3f}/{args.conv}_pk.pkl.npy" for i in range(1, 1001)]
    
        
    #setup_logging()
    
        
    
    
    
    id_list = np.array_split(range(0, 1000), args.nchunks)[args.chunk]
    id_list = np.array_split(id_list, comm.Get_size())[comm.Get_rank()]
    print(comm.Get_rank(), id_list)
    for j, i in enumerate(id_list):
        
        # Or StandardPowerSpectrumTemplate, DirectPowerSpectrumTemplate
        template = BAOPowerSpectrumTemplate(z=redshift, fiducial='DESI', apmode = 'qiso')  # effective redshift
        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template, mode = "" if args.conv == 'pre' else f'rec{args.conv}', )
        output_fn = f"{output_dir}/minuit_prof_{i:03d}.txt"
        if os.path.isfile(output_fn + ".npy") and not args.rerun: continue
        print(f"==> Starting for id {i}", flush=True)
        observable = TracerPowerSpectrumMultipolesObservable(data=ez_list[i],  # path to data, *pypower* power spectrum measurement, array, or dictionary of parameters where to evaluate the theory to take as a mock data vector
                                                            covariance = covariance if args.cov_type == 'analytic' else ez_list,
                                                            #covariance=covariance,  # path to mocks, array (covariance matrix), or None
                                                            klim={0: [kmin, kmax, dk]},  # k-limits, between 0.01 and 0.2 h/Mpc with 0.005 h/Mpc step size for ell = 0, 2
                                                            theory=theory,
                                                            wmatrix = full_window,
                                                            kinlim = (0., 0.5),
                                                            ellsin = (0,),
                                                            )  # previously defined theory
        
        likelihood = ObservablesGaussianLikelihood(observables=[observable])#, mpicomm = comm)
        #[p.update(derived = ".auto", prior = None) for p in likelihood.all_params.select(basename='al*') ]
        
        for param in likelihood.all_params.select(basename='al*_*'): 
            param.update(derived='.auto')
        [p.update(derived='{sigmapar}') for p in likelihood.all_params.select(name="sigmaper")]
        for param in likelihood.all_params.select(basename="sigma*"):
            param.update(fixed = False, ref=dict(limits=(0,20.)))
        #if j == 1:
        observable.plot_covariance_matrix(show=True)
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_tc_bao_cov.png")
             
        
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
        #from desilike.samples.profiles import Profiles
        #profiles = Profiles.load(fname)
        likelihood(**profiler.profiles.bestfit.choice(params=likelihood.varied_params))
        observable.plot()
        plt.gcf()
        plt.savefig("plots/desilike_ez_v_tc_bao_bestfit.png")
        
       
        
        