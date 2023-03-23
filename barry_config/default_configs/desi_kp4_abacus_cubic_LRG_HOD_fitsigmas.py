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

    # Loop over the mocktypes
    allnames = []
    mocknames = ['desi_kp4_abacus_cubicbox_pk_lrg', 'desi_kp4_abacus_cubicbox_xi_lrg']
    for i, mockname in enumerate(mocknames):

        for hod in range(9):
        
            # Loop over pre- and post-recon power spectrum measurements
            for recon in [None, "sym"]:

                # Create the data. We'll fit monopole, quadrupole between k=0.02 and 0.3.
                # First load up mock mean and add it to the fitting list.
                if "_pk_" in mockname:
                    dataset = PowerSpectrum_DESI_KP4(
                        recon=recon,
                        fit_poles=[0, 2],
                        min_k=0.02,
                        max_k=0.30,
                        realisation=None,          # realisation=None while load the average of all the realisations
                        num_mocks=1000,            # Used for Hartlap/Sellentin correction if correction=Correction.HARTLAP or Correction.SELLENTIN
                        reduce_cov_factor=25,      # Reduce covariance matrix by 25 for mock average
                        datafile=mockname + f"_hod{hod}.pkl",
                        data_location="../prepare_data/",
                    )

                    # Set up the appropriate model for the power spectrum
                    model = PowerBeutler2017(
                        recon=dataset.recon,                   
                        isotropic=dataset.isotropic,
                        fix_params=["om", "alpha", "epsilon"],    # Fix Omega_m, alpha and epsilon.
                        marg="full",                              # Analytic marginalisation
                        poly_poles=dataset.fit_poles,
                        correction=Correction.NONE,               # No covariance matrix debiasing
                        n_poly=6,                                 # 6 polynomial terms for P(k)
                    )

                else:
                    dataset = CorrelationFunction_DESI_KP4(
                        recon=recon,
                        fit_poles=[0, 2],
                        min_dist=52.0,
                        max_dist=150.0,
                        realisation=None,
                        num_mocks=1000,
                        reduce_cov_factor=25,
                        datafile=mockname + f"_hod{hod}.pkl",
                        data_location="../prepare_data/",
                    )

                    model = CorrBeutler2017(
                        recon=dataset.recon,
                        isotropic=dataset.isotropic,
                        marg="full",
                        fix_params=["om", "alpha", "epsilon"],
                        poly_poles=dataset.fit_poles,
                        correction=Correction.NONE,
                        n_poly=4,                                 # 4 polynomial terms for Xi(s)
                    )

                # Load in the proper DESI BAO template rather than Barry computing its own.
                pktemplate = np.loadtxt("../prepare_data/DESI_Pk_template.dat")
                model.kvals, model.pksmooth, model.pkratio = pktemplate.T

                # Give the data+model pair a name and assign it to the list of fits
                name = dataset.name + " mock mean"
                fitter.add_model_and_dataset(model, dataset, name=name)
                allnames.append(name)

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