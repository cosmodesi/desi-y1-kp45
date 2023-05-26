# Tests on Fiducial Cosmology

Contact: Alejandro Pérez Fernández, alejandroperez@estudiantes.fisica.unam.mx

 
The fiducial cosmologies currently tested are the 4 AbacusSummit secondary cosmologies
c{001..004} as described in:
https://abacussummit.readthedocs.io/en/latest/cosmologies.html

These scripts require the following entries as input:
- mocktype: cubic, cutsky
- tracer: lrg, elg, qso
- whichmocks: firstgen, sv3
- true cosmology: 000, 003, 004 (cosmology from the simulation)
- grid cosmology: 000, 001, 002, 003, 004 (cosmology for the redshift to distance relation)
- phase: an integer from 0 to 25 for c000 and from 0 to 5 for c003 and c004.

For cutsky mocks you additionally need:
- zbin: 0, 1, 2



For example:
``` terminal
python power_spectrum.py cutsky lrg firstgen 000 003 22 0
```
For FirtGen mocks, only c000 mocks are available.
For c003 and c004 only boxes for LRGs and ELGs are avaliable (no cutsky yet).

Reconstruction is run with **pyrecon**, while power spectra are computed with **pypower**
and 2PCF with **pycorr**.
Before running the scripts, the cosmodesi environment by Arnaud de Mattia should
be sourced, as follows:
``` terminal
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```

The choices for biases and redshift bins follow the details found in:
https://desi.lbl.gov/trac/wiki/keyprojects/y1/mockchallenge/2pt

The Barry scripts are based on the examples provided by Cullan Howlett in:
https://github.com/cosmodesi/desi-y1-kp45/tree/main/barry_config
