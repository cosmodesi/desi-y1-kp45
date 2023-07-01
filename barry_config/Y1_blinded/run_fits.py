import os
import sys
import pickle
import numpy as np

sys.path.append("../../../Barry/")     # Change this so that it points to where you have Barry installed

from barry.samplers import DynestySampler
from barry.config import setup
from barry.models import PowerBeutler2017, CorrBeutler2017
from barry.datasets.dataset_power_spectrum import PowerSpectrum_DESI_KP4
from barry.datasets.dataset_correlation_function import CorrelationFunction_DESI_KP4
from barry.fitter import Fitter
from barry.models.model import Correction

if __name__ == "__main__":
    
    # Get the relative file paths and names
    pfn, dir_name, file = setup(__file__)

    # Set up the Fitting class and Dynesty sampler with 500 live points. 
    # Set remove_output=False to make sure that we don't delete/overwrite existing chains in the same directory.
    fitter = Fitter(dir_name, remove_output=False)
    sampler = DynestySampler(temp_dir=dir_name, nlive=500)
    
    # All the tracer types, with zmin and zmax arrays
    tracers = {'BGS_BRIGHT-21.5': [[0.1, 0.4]], 
           'LRG': [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]], 
           'ELG_LOPnotqso': [[0.8, 1.1], [1.1, 1.6]],
           'QSO': [[0.8, 2.1]]}

    # The different sky areas
    caps = ["NGC", "SGC", "GCcomb"]
    
    # The optimal sigma values. For now, I've chosen these based on intuition rather than 
    # mock fits, so should be updated as appropriate. These are a set of dictionaries within dictionaries
    # based first on tracer, then on reconstruction method. At the lowest level, it contains a list of values
    # corresponding to each redshift bin
    sigma_nl_par = {'BGS_BRIGHT-21.5': {None: [11.0], "sym": [6.5], "iso": [6.0]},
                    'LRG': {None: [10.0, 9.5, 9.0], "sym": [6.0, 5.5, 5.0], "iso": [6.0, 5.5, 5.0]}, 
                    'ELG_LOPnotqso': {None: [8.5, 8.0], "sym": [5.5, 5.0], "iso": [5.5, 5.0]},
                    'QSO': {None: [8.0], "sym": [5.0], "iso": [5.0]}}
    
    sigma_nl_perp = {'BGS_BRIGHT-21.5': {None: [6.0], "sym": [2.0], "iso": [2.0]},
                    'LRG': {None: [6.0, 5.5, 5.0], "sym": [2.0, 2.0, 2.0], "iso": [2.0, 2.0, 2.0]}, 
                    'ELG_LOPnotqso': {None: [5.0, 5.0], "sym": [2.0, 1.5], "iso": [2.0, 1.5]},
                    'QSO': {None: [5.0], "sym": [1.5], "iso": [1.5]}}
    
    sigma_s = {'BGS_BRIGHT-21.5': {None: [2.0], "sym": [0.0], "iso": [0.0]},
                    'LRG': {None: [0.0, 0.0, 0.0], "sym": [0.0, 0.0, 0.0], "iso": [0.0, 0.0, 0.0]}, 
                    'ELG_LOPnotqso': {None: [3.0, 3.0], "sym": [0.0, 0.0], "iso": [0.0, 0.0]},
                    'QSO': {None: [3.0], "sym": [0.0], "iso": [0.0]}}
    
    # Loop over all the tracers, redshift bins and caps
    allnames = []
    for t in tracers:
        for i, zs in enumerate(tracers[t]):
            for cap in caps:
        
                # Loop over pre-recon measurements only for now, but different recon types can be added later.
                for recon in [None]:

                    # Load in the appropriate pickle file. Fit only moonopole and quadrupole. Cut to 50 Mpc/h < s < 150 Mpc/h
                    name = f"DESI_Y1_BLIND_{t.lower()}_{cap.lower()}_{zs[0]}_{zs[1]}.pkl"
                    dataset_xi = CorrelationFunction_DESI_KP4(
                        recon=recon,
                        fit_poles=[0, 2],
                        min_dist=50.0,
                        max_dist=150.0,
                        realisation='data',
                        reduce_cov_factor=1,
                        datafile=name,
                        data_location="./",
                    )
            
                    # Set up the model with 4 polynomial terms for xi and using full marginalisation over the broadband shape.
                    model_xi = CorrBeutler2017(
                        recon=dataset_xi.recon,
                        isotropic=dataset_xi.isotropic,
                        marg="full",
                        poly_poles=dataset_xi.fit_poles,
                        n_poly=4,
                    )
                    
                    # Set Gaussian priors for the BAO damping centred on the optimal values listed above
                    print(name, sigma_nl_par[t][recon][i], sigma_nl_perp[t][recon][i], sigma_s[t][recon][i])
                    model_xi.set_default("sigma_nl_par", sigma_nl_par[t][recon][i], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model_xi.set_default("sigma_nl_perp", sigma_nl_perp[t][recon][i], min=0.0, max=20.0, sigma=2.0, prior="gaussian")
                    model_xi.set_default("sigma_s", sigma_s[t][recon][i], min=0.0, max=20.0, sigma=2.0, prior="gaussian")

                    # Load in the proper DESI BAO template rather than Barry computing its own.
                    pktemplate = np.loadtxt("./DESI_Pk_template.dat")
                    model_xi.kvals, model_xi.pksmooth, model_xi.pkratio = pktemplate.T

                    # Give the data+model pair a name and assign it to the list of fits
                    fitter.add_model_and_dataset(model_xi, dataset_xi, name=dataset_xi.name)
                    allnames.append(dataset_xi.name)

    print(allnames)
                
    # Set the sampler (dynesty) and assign 1 walker (processor) to each. If we assign more than one walker, for dynesty
    # this means running independent chains which will then get added together when they are loaded in.
    fitter.set_sampler(sampler)
    fitter.set_num_walkers(1)
    fitter.fit(file)
    
    # If this is being run for the first time (i.e., not via a submission script), dump the entire fitter class to a file
    # so we can use it to read in the chains/models/datasets for plotting in a jupyter notebook
    if len(sys.argv) == 1:
        outfile = fitter.temp_dir+pfn.split("/")[-1]+".fitter.pkl"
        with open(outfile, 'wb') as pickle_file:
            pickle.dump(fitter, pickle_file)