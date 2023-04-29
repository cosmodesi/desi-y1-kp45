# bao & full-shape fits: DESI blinded

This is a tentative repo to gather scripts for BAO & full-shape fits of the blinded power spectrum and correlation function data.

## Directories

- `py:` Python scripts for BAO and FS fits.
- `blinded/test_w0-0.9040043101843285_wa-0.025634205416364297:` Results from blined catalogue with `w0_blined=0.9040043101843285` and `wa_blinded=0.025634205416364297`.
- `nb:` postprocess and plots.
- `scripts:` Bash scripts to automate BAO and full-shape fits for all tracer and blinded cosmology catalogs.

## Usage (TODO)
```bash
# BAO fits
./scripts/bao.sh

# Full-shape fits
./scripts/fs.sh

# Postprocess and plots
./scripts/postprocess.sh
```

## Contact
Contacts: Uendert Andrade (uendsa@umich.edu), and Arnaud de Mattia (arnaud.de-mattia@cea.fr).
