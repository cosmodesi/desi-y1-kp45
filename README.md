# bao & full-shape fits: DESI blinded

This is a tentative repo to gather scripts for BAO & full-shape fits of the blinded power spectrum and correlation function data.

## Directories

- `py:` Python scripts for BAO and FS fits.
- `nb:` postprocess and plots.
- `.:` Bash scripts to automate BAO and full-shape fits for all tracer and blinded cosmology catalogs.

## Usage
```bash
# BAO fits
./bao_fit_blinded.sh
## Additonaly, run the following to run BAO fits fixing the covariance to unblinded cosmology.
./bao_fit_fixed_CovMatrix_to_unblinded.sh

# Full-shape fits
./fs_fit_blinded.sh

# Postprocess and plots
nb/plot_bao.ipynb
nb/plot_fs.ipynb
```

NB.: Curretly, there is no `configuration space covariance` for each specific blinded catalogs. The covariance matrix is the same for all catalogs; generaded with unblinded catalog information. Once the configuration space covariance for each blinded cosmology is available, the BAO and full-shape fits can be easly updated to include it.

## Contact
Contacts: Uendert Andrade (uendsa@umich.edu), and Arnaud de Mattia (arnaud.de-mattia@cea.fr).
