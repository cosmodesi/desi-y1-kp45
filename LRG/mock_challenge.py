#!/home/users/csaulder/anaconda3/envs/desimock/bin/python
#this top line is just for our local cluster, so that it finds the right python environment

#loading required (and useful) libaries 
import pycorr
import cosmoprimo
import configparser
import argparse
from pycorr import TwoPointCorrelationFunction,TwoPointCounter
from mpi4py import MPI
import scipy as sp
import numpy as np
import glob
import sys, getopt 
import os
from astropy.io import fits



#define a function to read required data from the fits files
def basicreader(data_file):
    data_raw=fits.open(data_file,memmap=True)[1].data
    
    read_datatype=[('ra','f'),('dec','f'),('z','f'),('flag','i')]
    n_datapoint=len(data_raw)
    data_read=np.zeros(n_datapoint,dtype=read_datatype)

    data_read['ra']=data_raw['RA']
    data_read['dec']=data_raw['DEC']
    data_read['z']=data_raw['Z']
    data_read['flag']=data_raw['STATUS']
    
    return data_read

##function to convert the redshifts to distances and transform everything to cartesian coordinates
def distance_conversion(cosmo,dataset):
    dists = cosmo.comoving_radial_distance(dataset['z'])
    positions = pycorr.utils.sky_to_cartesian(np.array([dataset['ra'],dataset['dec'],dists]))
    positions=np.array(positions).T
    return positions


argv=sys.argv[1:]

# take arguement from the script file running this code
try:
    opts, args = getopt.getopt(argv,"hi:o:n:c::y:z:",["help","infile=","outfile=","n_rand=","n_cpu=","zmin=","zmax="])
except getopt.GetoptError:
    print('mock.py -i <infile> -o <outfile> -n <n_rand> -c <n_cpu> -y <zmin> -z <zmax>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('mock.py -i <infile> -o <outfile> -n <n_rand> -c <n_cpu> -y <zmin> -z <zmax>')
        sys.exit()
    if opt in ("-i", "--infile"):
        infile = arg
    if opt in ("-o", "--outfile"):
        outfile = arg
    if opt in ("-n", "--n_rand"):
        n_rand = arg
    if opt in ("-c", "--n_cpu"):
        n_cpu = arg
    if opt in ("-y", "--zmin"):
        zmin = arg
    if opt in ("-z", "--zmax"):
        zmax = arg
         
print(infile,outfile,n_rand,n_cpu,zmin,zmax)
#make sure that the input data has the right format
n_random=int(n_rand)
nthreads=int(n_cpu)
z_min=float(zmin)
z_max=float(zmax)
data_file=str(infile)
savefile=str(outfile)


#set up mpi
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_comms = comm.size
print(size_comms,rank)

#set up cosmology for distance calculations
h=0.6766
sigma8=0.8102
Omega_m=0.3111
cosmo = cosmoprimo.Cosmology(h=h,sigma8=sigma8,Omega0_m=Omega_m)
cosmo.set_engine('class')
print("setup done")


#path to infile (I could have put this in the seperate input variable, but this code has only one application right now)
LRG_path="/caefs/data/desi/mocks/cutsky_mocks/LRG/z0.800/"

#data_file='cutsky_LRG_z0.800_AbacusSummit_base_c000_ph001.fits'
#load data file
data=basicreader(LRG_path+data_file)
print("data file read in")

#we want the path to all random files (and later pick a number of them)
random_files=glob.glob(LRG_path+"cutsky_LRG_random*.fits")

trimmed_random_filed=random_files[0:n_random]

random_collect=[]
for random_file in trimmed_random_filed:
    random_load=basicreader(random_file)
    random_collect.append(random_load)


# now we have a random catalogue n_random times the size of the data catalogue     
random=np.concatenate(random_collect)
print("random assambled")


#cut down the data and random to the selected redshift range and to the DESI Y5 footprint
data_ranged=data[(data['z']>z_min)&(data['z']<z_max)&((data['flag']&2**1)!=0)&((data['flag']&2**3)!=0)]
random_ranged=random[(random['z']>z_min)&(random['z']<z_max)&((random['flag']&2**1)!=0)&((random['flag']&2**3)!=0)]
print("range selected")
#convert redshifts into distances ...note: maybe I should check if the cartesian conversion is truly neccesiary and pycorr can handle spherical coordinates
data_pos=distance_conversion(cosmo,data_ranged)
rand_pos=distance_conversion(cosmo,random_ranged)
print("distance converison done")

#set binning grid according to specifications
edges = (np.linspace(0, 200.00001, 201), np.linspace(-1, 1, 121))


suitable_RR_exists=False
RR_name="samples/mockchallenge/RR_"+str(n_rand)+"_"+str(np.round(z_min,3))+"_"+str(np.round(z_max,3))


try:
    RR_done=TwoPointCounter.load(RR_name+".npy")
    suitable_RR_exists=True
    print("file there")
except:
    suitable_RR_exists=False
    print("file not there")
    

if (suitable_RR_exists==True):
    print("running with given RR")

    result = TwoPointCorrelationFunction('smu', edges, data_positions1=data_pos, 
                                            randoms_positions1=rand_pos, position_type='pos',
                                            engine='corrfunc', R1R2=RR_done, compute_sepsavg=False, nthreads=nthreads,mpicomm = comm,mpiroot=0)   
else:
    #Calculate the the TwoPointCorrelationFunction
    print("running with new RR")
    result = TwoPointCorrelationFunction('smu', edges, data_positions1=data_pos, 
                                            randoms_positions1=rand_pos, position_type='pos',
                                            engine='corrfunc', compute_sepsavg=False, nthreads=nthreads,mpicomm = comm,mpiroot=0)
    result.R1R2.save(RR_name)



#save all the results 
result.save("samples/mockchallenge/"+savefile)
print("completed")


#dir(result)















