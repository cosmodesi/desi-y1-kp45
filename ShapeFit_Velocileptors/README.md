# ShapeFit_Velocileptors

## Emulator
This is a ShapeFit emulator of velocileptors, based on a Taylor series (to 3rd order). It generates Pell(k) multipoles (mono, quad, and hexa) for a given set of shapefit parameters (fsigma8, alpha_par, alpha_perp, m) and bias terms. 

The emulator supports mpi, and can be created using:

srun -n X -c Y python make_emulator_pells.py

The taylor series is then stored in

/emu/shapefit_z_X.XX_Om_Y.YY_pkells.json.

In order to run the emulator one simply enters:

emu = Emulator_Pells('emu/shapefit_z_%.2f_Om_%.2f_pkells.json'%(z,Omfid),order=4) kvec, p0, p2, p4 = emu(cpars, bpars)

cpars are the shapefit parameters and bpars are the bias terms.

To run this you will need to have velocileptors:

https://github.com/sfschen/velocileptors

and FinDiff https://findiff.readthedocs.io/en/latest/.


## Example notebook: test_emulator.ipynb
The test_emulator.ipynb notebook walks you through generating Pell(k) multipoles given a set of shapefit and bias parameters. We show a comparison of emulator multipoles using a given kvector ki and linear power spectrum pi vs the direct calculation using velocileptors with an input pi*fac(ki,m), where fac(ki,m) is the shapefit factor for a specific m.

## Example Cobaya script

Included are sample scripts for running a fit on BOSS data using the shapefit emulator. The path to the emulator .json file must be provided in the theory section of the .yaml file. Taylor_pk_theory_zs(Theory) class in the pk_likelihood_gc_emu_m.py script unpacks the emulator and generates the theory P_ell(k) multipoles for the fit.