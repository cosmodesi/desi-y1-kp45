#!/home/users/csaulder/anaconda3/envs/desimock/bin/python
#this top line is just for our local cluster, so that it finds the right python environment

#loading required (and useful) libaries 
import pycorr
import cosmoprimo
import configparser
import argparse
from pycorr import TwoPointCorrelationFunction,TwoPointCounter,AnalyticTwoPointCounter
from mpi4py import MPI
import scipy as sp
import numpy as np
import glob
import sys, getopt 
import os
from astropy.io import fits



#define a function to read required data from the fits files
def basicreader(data_file,zcosmo,Lbox,HubbleParam):
    segment_files = glob.glob(data_file+'*.fits.gz')
    
    data_collect=[]
    #loop over the 64 files in which the box is split
    for sfiles in segment_files:
        data_raw=fits.open(sfiles,memmap=True)[1].data
        
        read_datatype=[('x','f'),('y','f'),('z','f')]
        n_datapoint=len(data_raw)
        data_read=np.zeros(n_datapoint,dtype=read_datatype)

        data_read['x']=data_raw['x']
        data_read['y']=data_raw['y']
        #add RSD
        z_spec=data_raw['z']+data_raw['vz']*(1.+zcosmo)/HubbleParam
        data_read['z']=z_spec=z_spec%Lbox # periodic boundary conditions
        
        data_collect.append(data_read)
        
    data_complete=np.concatenate(data_collect,axis=0)
    clean_mask=np.invert(np.isnan(data_complete['x'])|np.isnan(data_complete['y'])|np.isnan(data_complete['z']))
    
    data_complete=data_complete[clean_mask]

    return data_complete




argv=sys.argv[1:]

# take arguement from the script file running this code
try:
    opts, args = getopt.getopt(argv,"hn:c:z:",["help","n_rand=","n_cpu=","zbox="])
except getopt.GetoptError:
    print('mock.py -n <n_rand> -c <n_cpu>  -z <zbox>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('mock.py  -n <n_rand> -c <n_cpu> -z <zbox>')
        sys.exit()
    if opt in ("-n", "--n_rand"):
        n_rand = arg
    if opt in ("-c", "--n_cpu"):
        n_cpu = arg
    if opt in ("-z", "--zbox"):
        zbox = arg



#infile='AbacusSummit_base_c000_ph000/LRG_snap20_ph000.gcat.'
#outfile= 'results/results_mockbox_000'
#n_rand=20
#n_cpu=28
#zbox=0.8

#print(infile,outfile,n_rand,n_cpu,zbox)
#make sure that the input data has the right format
n_random=int(n_rand)
nthreads=int(n_cpu)
zcosmo=float(zbox)
#data_file=str(infile)
#savefile=str(outfile)


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

Hz=cosmo.hubble_function(zcosmo)/h
Lbox=2000.0

#path to infile (I could have put this in the seperate input variable, but this code has only one application right now)
LRG_path="/caefs/data/desi/mocks/cubic_EZmocks_LRG/"
#rand_path="/caefs/data/desi/mocks/cubic_mocks/LRG/randombox/"


for i in range(1000):
    i_seed_str=str(i+1)
    data_file='EZmock_B2000G512Z0.8N8015724_b0.385d4r169c0.3_seed'+i_seed_str+"/"+"seed"+i_seed_str
#load data file
    data=basicreader(LRG_path+data_file,zcosmo,Lbox,Hz)
    print("data file read in")
    savepath="samples/mockchallenge/box/EZmocks/"
    savefile="results_seed"+i_seed_str


    n_gal=len(data)
    n_rand=n_gal*n_random
    #n_rand=len(RR_cat)

    #set binning grid according to specifications
    edges = (np.linspace(0, 200.00001, 201), np.linspace(-1, 1, 121))




    RR_done=AnalyticTwoPointCounter('smu', edges,boxsize=Lbox,size1=n_rand)
    RR_done.run()

    #DR_done=AnalyticTwoPointCounter('smu', edges,boxsize=Lbox,size1=n_gal,size2=n_rand)
    #DR_done.run()

    ##convert coordinates
    #rand_pos=np.array([RR_cat['x'],RR_cat['y'],RR_cat['z']]).T
    #rand_pos=rand_pos.astype('float')

    data_pos=np.array([data['x'],data['y'],data['z']]).T
    data_pos=data_pos.astype('float')


    #if (suitable_R_exists==True):
        #print("running with given RR")

    result = TwoPointCorrelationFunction('smu', edges, data_positions1=data_pos,  
                                        position_type='pos',
                                        engine='corrfunc', R1R2=RR_done, compute_sepsavg=False, nthreads=nthreads,mpicomm = comm,mpiroot=0,los='z',boxsize=Lbox)   
    #else:
        #Calculate the the TwoPointCorrelationFunction
    #print("running with new RR")
    #result = TwoPointCorrelationFunction('smu', edges, data_positions1=data_pos, 
                                            #randoms_positions1=rand_pos, position_type='pos',
                                            #engine='corrfunc', compute_sepsavg=False, nthreads=nthreads,mpicomm = comm,mpiroot=0,los='z',boxsize=Lbox)
        #result.R1R2.save(rand_path+"counters_R"+str(n_random))



    #save all the results 
    result.save(savepath+savefile)
    result.save_txt(savepath+savefile+".dat")
    result.save_txt(savepath+savefile+"_poles.dat",  ells=(0, 2, 4))
    print("results saved")

print("completed")
#dir(result)















