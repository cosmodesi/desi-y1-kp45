#!/home/users/csaulder/anaconda3/envs/mockchallenge/bin/python

import pycorr
import cosmoprimo
import configparser
import argparse
from pycorr import TwoPointCorrelationFunction
#from mpi4py import MPI
import scipy as sp
import numpy as np
import glob
import sys, getopt 
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
#from astropy.cosmology import FlatLambdaCDM


def mockchallenge_dataloader(filename,rebinfactor_s,rebinfactor_mu):
    alldata=TwoPointCorrelationFunction.load(filename)
    alldata.save_txt(filename[:-4]+".dat")
    alldata.save_txt(filename[:-4]+"_poles.dat",  ells=(0, 2, 4))
    alldata.rebin((rebinfactor_s,rebinfactor_mu))
    cf=alldata.corr
    sbins=alldata.sep[:,0]
    mubins=alldata.seps[1][0]
    return cf,sbins,mubins



def plot_redshift_evolution(filename46,filename68,filename81,rebinfactor_s,rebinfactor_mu,plotpath,plotname):
    cf46_all,sbins,mubins=mockchallenge_dataloader(filename46,rebinfactor_s,rebinfactor_mu)
    cf68_all,_,_=mockchallenge_dataloader(filename68,rebinfactor_s,rebinfactor_mu)
    cf81_all,_,_=mockchallenge_dataloader(filename81,rebinfactor_s,rebinfactor_mu)
    
    nmu=len(mubins)
    ns=len(sbins)
    
    for imu in range(nmu):
        cf46=cf46_all[:,imu]
        cf68=cf68_all[:,imu]
        cf81=cf81_all[:,imu]
        
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins)*cf46),np.min(np.square(sbins)*cf68),np.min(np.square(sbins)*cf81)])
        maxval=np.max([np.max(np.square(sbins)*cf46),np.max(np.square(sbins)*cf68),np.max(np.square(sbins)*cf81)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'redshift evolution $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+' $\mu$-bins)')        
        plt.axis([np.min(sbins), np.max(sbins),minval, maxval]) 
        plt.scatter(sbins,np.square(sbins)*cf46,c='b',marker='o',label='0.4 < z < 0.6')
        plt.scatter(sbins,np.square(sbins)*cf68,c='g',marker='v',label='0.6 < z < 0.8')
        plt.scatter(sbins,np.square(sbins)*cf81,c='r',marker='+',label='0.8 < z < 1.1')
                        
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"redshiftevolution_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   
        
        

def plot_compare_realizations(filename_phase1,filename_phase2,filename_phase3,filename_phase4,filename_phase5,rebinfactor_s,rebinfactor_mu,plotpath,plotname):
    
    cf1_all,sbins,mubins=mockchallenge_dataloader(filename_phase1,rebinfactor_s,rebinfactor_mu)
    cf2_all,_,_=mockchallenge_dataloader(filename_phase2,rebinfactor_s,rebinfactor_mu)
    cf3_all,_,_=mockchallenge_dataloader(filename_phase3,rebinfactor_s,rebinfactor_mu)
    cf4_all,_,_=mockchallenge_dataloader(filename_phase4,rebinfactor_s,rebinfactor_mu)
    cf5_all,_,_=mockchallenge_dataloader(filename_phase5,rebinfactor_s,rebinfactor_mu)
    
    nmu=len(mubins)
    ns=len(sbins)
    
    for imu in range(nmu):
        cf1=cf1_all[:,imu]
        cf2=cf2_all[:,imu]
        cf3=cf3_all[:,imu]
        cf4=cf4_all[:,imu]
        cf5=cf5_all[:,imu]
        
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins)*cf1),np.min(np.square(sbins)*cf2),np.min(np.square(sbins)*cf3),np.min(np.square(sbins)*cf4),np.min(np.square(sbins)*cf5)])
        maxval=np.max([np.max(np.square(sbins)*cf1),np.max(np.square(sbins)*cf2),np.max(np.square(sbins)*cf3),np.max(np.square(sbins)*cf4),np.max(np.square(sbins)*cf5)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'realization comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')        
        plt.axis([np.min(sbins), np.max(sbins),minval, maxval]) 
        plt.scatter(sbins,np.square(sbins)*cf1,c='b',marker='o',label='realization 001')
        plt.scatter(sbins,np.square(sbins)*cf2,c='g',marker='v',label='realization 002')
        plt.scatter(sbins,np.square(sbins)*cf3,c='r',marker='+',label='realization 003')
        plt.scatter(sbins,np.square(sbins)*cf4,c='m',marker='*',label='realization 004')
        plt.scatter(sbins,np.square(sbins)*cf5,c='c',marker='^',label='realization 005')
                        
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"realizationcomparison_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   
        

def plot_compare_randoms(filenames_rand5,filenames_rand20,rebinfactor_s,rebinfactor_mu,plotpath,plotname):
    
    cf_all_rand5=[]
    for filename_rand5 in filenames_rand5:
        cf_all,sbins,mubins=mockchallenge_dataloader(filename_rand5,rebinfactor_s,rebinfactor_mu)
        cf_all_rand5.append(cf_all)

    cf_all_rand20=[]
    for filename_rand20 in filenames_rand20:
        cf_all,sbins,mubins=mockchallenge_dataloader(filename_rand20,rebinfactor_s,rebinfactor_mu)
        cf_all_rand20.append(cf_all)

    n_real=len(filenames_rand5)
    nmu=len(mubins)
    ns=len(sbins)

    av_cf_rand5=np.mean(cf_all_rand5,axis=0)
    std_cf_rand5=np.std(cf_all_rand5,axis=0)
    
    av_cf_rand20=np.mean(cf_all_rand20,axis=0)
    std_cf_rand20=np.std(cf_all_rand20,axis=0)
    
    for imu in range(nmu):
        
        current_mu=mubins[imu]
        

        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r'n randoms comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')      
        #plt.axis([np.min(sbins), np.max(sbins),minval, maxval])     

        
        sigma_up=np.square(sbins)*(av_cf_rand5[:,imu]+std_cf_rand5[:,imu])
        sigma_down=np.square(sbins)*(av_cf_rand5[:,imu]-std_cf_rand5[:,imu])
        plt.fill_between(sbins, sigma_up, sigma_down , alpha=0.2,color="blue") 
        plt.plot(sbins,np.square(sbins)*av_cf_rand5[:,imu],lw=3,ls=':',color="blue", label="5x random")

        for ireal in range(n_real):
            current_cf=cf_all_rand5[ireal]
            plt.plot(sbins,np.square(sbins)*current_cf[:,imu],lw=1,ls='-',color="cyan", alpha=0.2)


        sigma_up=np.square(sbins)*(av_cf_rand20[:,imu]+std_cf_rand20[:,imu])
        sigma_down=np.square(sbins)*(av_cf_rand20[:,imu]-std_cf_rand20[:,imu])
        plt.fill_between(sbins, sigma_up, sigma_down , alpha=0.2,color="red") 
        plt.plot(sbins,np.square(sbins)*av_cf_rand20[:,imu],lw=3,ls=':',color="red", label="20x random")

        for ireal in range(n_real):
            current_cf=cf_all_rand20[ireal]
            plt.plot(sbins,np.square(sbins)*current_cf[:,imu],lw=1,ls='-',color="orange", alpha=0.2)

        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"randomcomparison_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()  

    
    #FINE LINES FOR EACH REALIZATION, THICKER LINES FOR AVERAGE AND STD
    

        
        
    
    #for i in range(n_real):
        ##cf_now=cf_all_rand5[i]
        
            ##cf1=cf_now[:,imu]
        
        #CONTINUE HERE 
    #cf2_all,_,_=mockchallenge_dataloader(filename_rand2,rebinfactor_s,rebinfactor_mu)
    #cf3_all,_,_=mockchallenge_dataloader(filename_rand3,rebinfactor_s,rebinfactor_mu)
    #cf4_all,_,_=mockchallenge_dataloader(filename_rand4,rebinfactor_s,rebinfactor_mu)
    
    #nmu=len(mubins)
    #ns=len(sbins)
    
    #for imu in range(nmu):
        #cf1=cf1_all[:,imu]
        #cf2=cf2_all[:,imu]
        #cf3=cf3_all[:,imu]
        #cf4=cf4_all[:,imu]
        
        #current_mu=mubins[imu]
    

        #plt.clf()
        #plt.figure(figsize=[6.4, 4.8])
        #plt.title(r'n randoms comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')        
        #plt.axis([np.min(sbins), np.max(sbins),minval, maxval]) 
        #plt.scatter(sbins,np.square(sbins)*cf1,c='b',marker='o',label='5X random')
        #plt.scatter(sbins,np.square(sbins)*cf2,c='g',marker='v',label='10X random')
        #plt.scatter(sbins,np.square(sbins)*cf3,c='r',marker='+',label='15X random')
        #plt.scatter(sbins,np.square(sbins)*cf4,c='m',marker='*',label='20X random')
                        
        #plt.xlabel(r's [$h^{-1}$ Mpc]')   
        #plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        #plt.tight_layout()    
        #plt.legend(loc='best',markerscale=1.,ncol=1)
        #plt.savefig(plotpath+"randomcomparison_"+plotname+"_sbin_"+str(ns)+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        #plt.close()   
            






def scan_redshift_evolution(rebinfactor_s,rebinfactor_mu):
    for i in range(4):
        n_rand=(i+1)*5
        for iphase in range(5):
            phase=str(iphase+1).zfill(3)
            plotname="results_realization"+str(phase)+"_rand"+str(n_rand)+"_"
            filename46=loadpath+plotname+"46.sh.npy"
            filename68=loadpath+plotname+"68.sh.npy"
            filename81=loadpath+plotname+"81.sh.npy"

            plot_redshift_evolution(filename46,filename68,filename81,rebinfactor_s,rebinfactor_mu,plotpath,plotname)


def scan_realizations(rebinfactor_s,rebinfactor_mu):
    for i in range(4):
        n_rand=(i+1)*5
        for iredshift in range(3):
            current_redshift=str(redshiftrange["z_name"][iredshift],'utf-8')
            
            filename_phase1=loadpath+'results_realization001_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase2=loadpath+'results_realization002_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase3=loadpath+'results_realization003_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase4=loadpath+'results_realization004_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            filename_phase5=loadpath+'results_realization005_rand'+str(n_rand)+'_'+current_redshift+'.sh.npy'
            
            plotname="results_rand"+str(n_rand)+"_"+current_redshift+"_"
            
            plot_compare_realizations(filename_phase1,filename_phase2,filename_phase3,filename_phase4,filename_phase5,rebinfactor_s,rebinfactor_mu,plotpath,plotname)
            
def scan_randoms(rebinfactor_s,rebinfactor_mu):
    for iredshift in range(3):
        filenames_rand5=[]
        filenames_rand20=[]
        for iphase in range(25):
            phase=str(iphase).zfill(3)
            
            current_redshift=str(redshiftrange["z_name"][iredshift],'utf-8')
            
            filename_rand5=loadpath+'results_realization'+phase+'_rand5_'+current_redshift+'.npy'
            filename_rand20=loadpath+'results_realization'+phase+'_rand20_'+current_redshift+'.npy'
            
            filenames_rand5.append(filename_rand5)
            filenames_rand20.append(filename_rand20)
            
            #filename_rand3=loadpath+'results_realization'+phase+'_rand15_'+current_redshift+'.sh.npy'
            #filename_rand4=loadpath+'results_realization'+phase+'_rand20_'+current_redshift+'.sh.npy'
    
    
        plotname="compare_randoms_"+current_redshift+"_"
        plot_compare_randoms(filenames_rand5,filenames_rand20,rebinfactor_s,rebinfactor_mu,plotpath,plotname)
        
    
    
            
                   
def sbin_tests(filename,rebinfactor_mu,plotpath,plotname):
    cf1_all,sbins1,mubins=mockchallenge_dataloader(filename_phase1,1,rebinfactor_mu)
    cf2_all,sbins2,_=mockchallenge_dataloader(filename_phase1,2,rebinfactor_mu)   
    cf5_all,sbins5,_=mockchallenge_dataloader(filename_phase1,5,rebinfactor_mu)   
    cf8_all,sbins8,_=mockchallenge_dataloader(filename_phase1,8,rebinfactor_mu)   
    cf10_all,sbins10,_=mockchallenge_dataloader(filename_phase1,10,rebinfactor_mu)   
    cf20_all,sbins20,_=mockchallenge_dataloader(filename_phase1,20,rebinfactor_mu)   

    nmu=len(mubins)
    
    for imu in range(nmu):
        cf1=cf1_all[:,imu]
        cf2=cf2_all[:,imu]
        cf5=cf5_all[:,imu]
        cf8=cf8_all[:,imu]
        cf10=cf10_all[:,imu]
        cf20=cf20_all[:,imu]
                
        current_mu=mubins[imu]
    
        minval=np.min([np.min(np.square(sbins1)*cf1),np.min(np.square(sbins2)*cf2),np.min(np.square(sbins5)*cf5),np.min(np.square(sbins8)*cf8),np.min(np.square(sbins10)*cf10),np.min(np.square(sbins20)*cf20)])
        maxval=np.max([np.max(np.square(sbins1)*cf1),np.max(np.square(sbins2)*cf2),np.max(np.square(sbins5)*cf5),np.max(np.square(sbins8)*cf8),np.max(np.square(sbins10)*cf10),np.max(np.square(sbins20)*cf20)])
    
        plt.clf()
        plt.figure(figsize=[6.4, 4.8])
        plt.title(r's bin comparison $\bar{\mu}=$'+str(np.round(current_mu,3))+' ('+str(nmu)+')$\mu$-bins')        
        plt.axis([np.min(sbins1), np.max(sbins1),minval, maxval]) 
        plt.scatter(sbins1,np.square(sbins1)*cf1,c='b',marker='o',label='1 Mpc/h',s=10)
        plt.scatter(sbins2,np.square(sbins2)*cf2,c='g',marker='v',label='2 Mpc/h',s=10)
        plt.scatter(sbins5,np.square(sbins5)*cf5,c='r',marker='+',label='5 Mpc/h',s=10)
        plt.scatter(sbins8,np.square(sbins8)*cf8,c='m',marker='*',label='8 Mpc/h',s=10)
        plt.scatter(sbins10,np.square(sbins10)*cf10,c='c',marker='^',label='10 Mpc/h',s=10)
        plt.scatter(sbins20,np.square(sbins20)*cf20,c='orange',marker='s',label='20 Mpc/h',s=10)
                                       
        plt.xlabel(r's [$h^{-1}$ Mpc]')   
        plt.ylabel(r'$s^{2} \xi$ [$h^{-1}$ Mpc]') 
        plt.tight_layout()    
        plt.legend(loc='best',markerscale=1.,ncol=1)
        plt.savefig(plotpath+"sbin_comparison_"+plotname+"_mu_"+str(nmu)+"_"+str(imu)+".png",dpi=300, facecolor='w',edgecolor='w')
        plt.close()   



#def mockchallenge_dataloader_only(filename,rebinfactor_s,rebinfactor_mu):
    #alldata=TwoPointCorrelationFunction.load(filename)
    #alldata.rebin((rebinfactor_s,rebinfactor_mu))
    #cf=alldata.corr
    #sbins=alldata.sep[:,0]
    #mubins=alldata.seps[1][0]
    #return cf,sbins,mubins
##something like "s_d, xiell_d = project_to_multipoles(result_data, ells=ells)" where result_data is the computation of your 2pcf 

#redshiftrange=np.zeros(3,dtype=[('z_name', 'S2'),('zmin', '<f8'), ('zmax', '<f8')])
#redshiftrange["z_name"]=["46","68","81"]
#redshiftrange["zmin"]=[0.4,0.6,0.8]
#redshiftrange["zmax"]=[0.6,0.8,1.1]

#EZmock_path="samples/mockchallenge/EZmocks/"
##result.save("samples/mockchallenge/"+savefile)
#filename="EZmock_results_"+filesuffix+"_1.npy"
#_,sbins,mubin=mockchallenge_dataloader_only(EZmock_path+filename,2,6)


#for iredshift in range(3):
    #filesuffix=str(redshiftrange["z_name"][iredshift],'utf-8')
    #for imu in range(len(mubin)):
        #cf_coll=[]
        #for i in range(1000):
            #currentEZ=str(i+1)
            #filename="EZmock_results_"+filesuffix+"_"+currentEZ+".npy"
            #load_cf,sbins,mubin=mockchallenge_dataloader_only(EZmock_path+filename,2,6)
            #current_cf=load_cf[:,imu]
            #cf_coll.append(current_cf)
        #av_xi=np.mean(cf_coll,axis=0)
        #std_xi=np.std(cf_coll,axis=0)
        


        #cf_all_rand20=[]
        #for filename_rand20 in filenames_rand20:
            #cf_all,sbins,mubins=mockchallenge_dataloader(filename_rand20,2,6)
            #cf_all_rand20.append(cf_all)




plotpath="plots/mockchallenge/new_"
loadpath="samples/mockchallenge/results/"


#rebinfactor_s=1
#rebinfactor_mu=1

#scan_redshift_evolution(rebinfactor_s,rebinfactor_mu)
#scan_realizations(rebinfactor_s,rebinfactor_mu)
#scan_randoms(rebinfactor_s,rebinfactor_mu)



#rebinfactor_s=1
#rebinfactor_mu=6

#scan_redshift_evolution(rebinfactor_s,rebinfactor_mu)
#scan_realizations(rebinfactor_s,rebinfactor_mu)
#scan_randoms(rebinfactor_s,rebinfactor_mu)



rebinfactor_s=5
rebinfactor_mu=6

scan_randoms(rebinfactor_s,rebinfactor_mu)




rebinfactor_s=1
rebinfactor_mu=6

scan_randoms(rebinfactor_s,rebinfactor_mu)



rebinfactor_s=1
rebinfactor_mu=1

scan_randoms(rebinfactor_s,rebinfactor_mu)









#scan_redshift_evolution(rebinfactor_s,rebinfactor_mu)

#scan_realizations(rebinfactor_s,rebinfactor_mu)

        

#filename=loadpath+'results_realization001_rand20_46.sh.npy'
#plotname="results_realization001_rand20_46_"
#sbin_tests(filename,rebinfactor_mu,plotpath,plotname)

#filename=loadpath+'results_realization001_rand20_68.sh.npy'
#plotname="results_realization001_rand20_68_"
#sbin_tests(filename,rebinfactor_mu,plotpath,plotname)

#filename=loadpath+'results_realization001_rand20_81.sh.npy'
#plotname="results_realization001_rand20_81_"
#sbin_tests(filename,rebinfactor_mu,plotpath,plotname)








