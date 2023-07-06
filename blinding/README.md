# bao & full-shape fits: DESI blinded

This is a tentative repo to gather scripts for BAO & full-shape fits of the blinded power spectrum and correlation function data.

## Directories

- `py:` Python scripts for BAO and FS fits.
- `nb:` postprocess and plots.
- `.:` Bash scripts to automate BAO and full-shape fits for all tracer and blinded cosmology catalogs.

## Usage example
```bash
# BAO & Full-shape fits
python py/fit_y1.py --tracer LRG --template bao --theory dampedbao --observable power --todo profiling

# Postprocess and plots
nb/
```

## Contact
Contacts: Uendert Andrade (uendsa@umich.edu), and Arnaud de Mattia (arnaud.de-mattia@cea.fr).
