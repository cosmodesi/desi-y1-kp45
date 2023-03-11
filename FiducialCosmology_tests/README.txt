Scripts for Fiducial Cosmology tests using FirstGen Mocks
--------------------------------------------------------------------------------
Contact: Alejandro Pérez Fernández, alejandroperez@estudiantes.fisica.unam.mx

These scripts require to input the tracer name as an argument. E.g.

python power_spectrum_CutSky.py elg

Reconstruction is run with Pyrecon, while power spectra are computed with Pypower.
Before running the scripts, the cosmodesi environment by Arnaud de Mattia should
be sourced, as follows:

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main


The choices for biases and redshift bins follow the details found in:

https://desi.lbl.gov/trac/wiki/keyprojects/y1/mockchallenge/2pt

The fiducial cosmologies currently tested are the 4 Abacus secondary cosmologies
c{001..004} as described in:

https://abacussummit.readthedocs.io/en/latest/cosmologies.html
