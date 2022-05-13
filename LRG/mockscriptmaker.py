#!/home/users/csaulder/anaconda3/bin/python

#Initialization
import numpy as np 
import scipy
import math
import astropy
import astropy.units as u
from scipy.spatial import cKDTree as kdtree
from astropy.coordinates import SkyCoord, CartesianRepresentation, SphericalRepresentation ,UnitSphericalRepresentation # High-level coordinates
from astropy.coordinates import Angle, Latitude, Longitude, Distance  # Angles
from astropy import constants as const
from astropy.coordinates import ICRS, Galactic, FK4, FK5 
from astropy.io import fits
import astropy.coordinates as coord
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splev, splrep
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import healpy as hp 
from scipy.stats import skewnorm,norm
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from Corrfunc.theory import *
import multiprocessing as mp
from mpi4py import MPI
from Corrfunc.utils import convert_3d_counts_to_cf
import threading
import glob

import mod_corrfunc

import sys, getopt




def scriptmaker(inputfilename,filename,n_rand,zmin,zmax):
    mainscriptfile = open("mockchallenge_scripts/pbs_mockchallenge_"+filename+".sh","w+")
    mainscriptfile.writelines("#!/bin/bash \n")
    mainscriptfile.writelines("#$PBS -S /bin/bash \n\n")
    #mainscriptfile.writelines("cd .. \n\n")
    mainscriptfile.writelines("#PBS -N "+filename+" \n")
    mainscriptfile.writelines("#PBS -r n \n")
    mainscriptfile.writelines("#PBS -e logs/"+filename+".err.$PBS_JOBID \n")
    mainscriptfile.writelines("#PBS -o logs/"+filename+".log.$PBS_JOBID \n")
    mainscriptfile.writelines("#PBS -q long \n")
    mainscriptfile.writelines("#PBS -l nodes=1:ppn=28 \n\n")
    mainscriptfile.writelines("module load python3 \n\n")    
    mainscriptfile.writelines("cd $PBS_O_WORKDIR \n\n")
    mainscriptfile.writelines("source activate desimock \n\n")
    mainscriptfile.writelines("rm -f /dev/shm/* \n\n")
    mainscriptfile.writelines("./mock_challenge.py -i '"+inputfilename+"' -o 'results/results_"+filename+"' -n "+str(n_rand)+" -c 28 -y "+str(zmin)+" -z "+str(zmax)+" > logs/"+filename+".out.$PBS_JOBID \n\n")
    mainscriptfile.close()







redshiftrange=np.zeros(3,dtype=[('z_name', 'S2'),('zmin', '<f8'), ('zmax', '<f8')])
redshiftrange["z_name"]=["46","68","81"]
redshiftrange["zmin"]=[0.4,0.6,0.8]
redshiftrange["zmax"]=[0.6,0.8,1.1]

controllfile = open("mockchallenge_scripts/run_pycorr.sh","w")
controllfile.writelines("#!/bin/bash \n")

for iphase in range(10):
    phase=str(iphase+1).zfill(3)

    inputfilename="cutsky_LRG_z0.800_AbacusSummit_base_c000_ph"+phase+".fits"    
    for i in range(8):
        n_rand=(i+1)*5


        for iredshift in range(3):
            
            filename='realization'+str(phase)+'_rand'+str(n_rand)+'_'+str(redshiftrange["z_name"][iredshift],'utf-8')
            
            scriptmaker(inputfilename,filename,n_rand,redshiftrange["zmin"][iredshift],redshiftrange["zmax"][iredshift])

            controllfile.writelines("sleep 5 \n")
            controllfile.writelines("qsub pbs_mockchallenge_"+filename+".sh \n\n")


controllfile.close()







redshiftrange=np.zeros(3,dtype=[('z_name', 'S2'),('zmin', '<f8'), ('zmax', '<f8')])
redshiftrange["z_name"]=["46","68","81"]
redshiftrange["zmin"]=[0.4,0.6,0.8]
redshiftrange["zmax"]=[0.6,0.8,1.1]

controllfile = open("mockchallenge_scripts/run_pycorr_rest.sh","w")
controllfile.writelines("#!/bin/bash \n")

for iphase in range(25):
    if ((iphase!=0)):
    
        phase=str(iphase).zfill(3)

        inputfilename="cutsky_LRG_z0.800_AbacusSummit_base_c000_ph"+phase+".fits"    
        #n_rand=5


        #for iredshift in range(3):
            
            #filename='realization'+str(phase)+'_rand'+str(n_rand)+'_'+str(redshiftrange["z_name"][iredshift],'utf-8')
            
            #scriptmaker(inputfilename,filename,n_rand,redshiftrange["zmin"][iredshift],redshiftrange["zmax"][iredshift])

            #controllfile.writelines("sleep 5 \n")
            #controllfile.writelines("qsub pbs_mockchallenge_"+filename+".sh \n\n")



        n_rand=20


        for iredshift in range(3):
            
            filename='realization'+str(phase)+'_rand'+str(n_rand)+'_'+str(redshiftrange["z_name"][iredshift],'utf-8')
            
            scriptmaker(inputfilename,filename,n_rand,redshiftrange["zmin"][iredshift],redshiftrange["zmax"][iredshift])

            controllfile.writelines("sleep 5 \n")
            controllfile.writelines("qsub pbs_mockchallenge_"+filename+".sh \n\n")



controllfile.close()








