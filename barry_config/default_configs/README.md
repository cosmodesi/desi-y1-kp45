Some default/example scripts for Barry. Python .py files should be run from the terminal. They will create some sub-directories (`job_files`, `out_files`, `plot`) and submit a 
series of model+dataset fitting configurations and submit each configuration as a single core job to the queue. 

`job_files` will contain the submission script created by Barry if you want to look at it
`out_files` will containt the log/output files for the job
`plots` will contain separate sub-directories for each script and, within each sub-directory, the chain files and other useful things

Alongside each python XXXX.py file there is a jupyter notebook (plot_XXXX.ipynb) which will read in the chains it finds in the `plots` directory and 
make some summary plots/files.

At the moment there are two pairs of fitting/plotting routines:
* `desi_kp4_abacus_cubic_LRG_fitsigmas`: Fits the Abacus cubic LRG mock mean, pre and post-recon, P(k), Xi(s) and Control Variate P(k) for the BAO damping and FoG damping parameters will fixing the BAO peak position to the expected value ($\alpha=1$, $\epsilon=0$). 6 total chains
* `desi_kp4_abacus_cubic_LRG`: Fits the mean and 25 individual realisations of the Abacus cubic LRG mocks, pre and post-recon, P(k), Xi(s) and Control Variate P(k) for the BAO parameters using Gaussian priors on the BAO damping and FoG damping parameters. 152 total chains

In both cases the number of polynomial terms, prior widths and prior central values have been optimised for the LRG mocks. For other tracers, I recommend following a similar procedure, i.e.,:
1. Running a variant of the first script to find the optimal central values for the damping parameters, 
2. Run a variant of the second script using the same prior widths and number of polynomials, but changing the central values of the priors to whatever you found from step 1.
