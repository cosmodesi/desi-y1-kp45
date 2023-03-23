This directory contains example scripts and codes for running BAO fits using Barry.
* `/prepare_data/` contains notebooks to read in clustering measurements and pickle them up for Barry.
* `/default_configs/` contains python code and notebooks to run fits and plot the results.

To use these with Barry you need to clone this repo into your home directory. Then `git clone https://github.com/Samreay/Barry.git` into your home directory too.

Barry should be set up to on Perlmutter and Cori using the cosmodesi environment by default. The only thing you need to do is add `export HPC=perlmutter` (or `export HPC=cori` if using cori) to your `~/.bashrc.ext` file and `source ~/.bashrc.ext` it. 
