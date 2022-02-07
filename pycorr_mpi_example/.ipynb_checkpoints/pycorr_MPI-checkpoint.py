import pycorr
import cosmoprimo


import configparser
import argparse

from pycorr import TwoPointCorrelationFunction
from mpi4py import MPI

import numpy as np



#Functions for moving from sky to cartesian coordinates


#Functions for reading FITS and HDF datasets

def ReadFITS(filename,label_ra,label_dec,label_z,label_w,usew):
    from astropy.io import fits
    hdul = fits.open(filename)
    ra_s = hdul[label_ra].data
    dec_s = hdul[label_dec].data
    z_s = hdul[label_z].data
    if(usew=='True'):
        w_s = hdul[label_w].data
    else:
        w_s = np.ones(len(ra_s))

    return ra_s,dec_s,z_s,w_s

def ReadHDF(filename,label_ra,label_dec,label_z,label_w,usew):
    import h5py
    f = h5py.File(filename, "r")
    ra_s=f[label_ra][...]
    dec_s=f[label_dec][...]
    z_s=f[label_z][...]
    if(usew=='True'):
        w_s=f[label_w][...]
    else:
        w_s = np.ones(len(ra_s))
    f.close()
    return ra_s,dec_s,z_s,w_s


if __name__ == "__main__":


    #Reading configuration


    desc = "CF computation"
    parser = argparse.ArgumentParser(description=desc)


    h = 'conf file'
    parser.add_argument('k', type=str, help=h)

    ns=parser.parse_args()
    config = ns.k

    conf = configparser.ConfigParser()
    conf.read(config)



    data_name = conf.get('Data','filename')

    data_type = (conf.get('Data','data_type'))


    label_ra=(conf.get('Data','label_ra'))
    label_dec=(conf.get('Data','label_dec'))
    label_z=(conf.get('Data','label_z'))

    usew = conf.get('Data','use_weights')

    if(usew=='True'):
        print('Loading weights for Data')
        label_w=(conf.get('Data','label_w'))







    data_name_r = conf.get('Randoms','filename')

    data_type_r = (conf.get('Randoms','data_type'))


    label_ra_r=(conf.get('Randoms','label_ra'))
    label_dec_r=(conf.get('Randoms','label_dec'))
    label_z_r=(conf.get('Randoms','label_z'))


    usew_r = conf.get('Randoms','use_weights')

    if(usew_r=='True'):
        print('Loading weights for Randoms')
        label_w_r=(conf.get('Randoms','label_w'))






    h=float(conf.get('Cosmology','h'))
    Omega_m=float(conf.get('Cosmology','omega_m'))
    sigma8=float(conf.get('Cosmology','sigma8'))
    engine=(conf.get('Cosmology','engine'))


    z_high = float(conf.get('Z_range','z_high'))
    z_low = float(conf.get('Z_range','z_low'))




    use_mpi = conf.get('Parallelisation','use_mpi')
    threads_per_proc = int(conf.get('Parallelisation','threads_per_proc'))




    output = conf.get('Output','filename')






    #MPI initialization

    if(use_mpi == 'True'):
        comm = MPI.COMM_WORLD
    else:
        comm=MPI.COMM_SELF
    rank = comm.Get_rank()
    size_comms = comm.size
    print('Succesfully launched the process of rank №', rank,' of total size of comms being  ',size_comms)




    #Reading datasets
    if(data_type=='FITS'):
        ra,dec,z,weights = ReadFITS(data_name,label_ra,label_dec,label_z,label_w,usew)
    elif(data_type=='HDF'):
        ra,dec,z,weights = ReadHDF(data_name,label_ra,label_dec,label_z,label_w,usew)
    else:
        print('Incorrect data_type in Data. Use either HDF or FITS')


    if(data_type_r=='FITS'):
        ra_r,dec_r,z_r,weights_r = ReadFITS(data_name_r,label_ra_r,label_dec_r,label_z_r,label_w_r,usew_r)
    elif(data_type=='HDF'):
         ra_r,dec_r,z_r,weights_r = ReadHDF(data_name_r,label_ra_r,label_dec_r,label_z_r,label_w_r,usew_r)
    else:
        print('Incorrect data_type in Randoms. Use either HDF or FITS')



    #Defining cosmology
    cosmo = cosmoprimo.Cosmology(h=h,sigma8=sigma8,Omega0_m=Omega_m)
    cosmo.set_engine(engine)


    #Converting coordinates
    dists = cosmo.comoving_radial_distance(z)
    dists_r = cosmo.comoving_radial_distance(z_r)
    pos = pycorr.utils.sky_to_cartesian(np.array([ra,dec,dists]))
    pos_r =pycorr.utils.sky_to_cartesian(np.array([ra_r,dec_r,dists_r]))




    #Cutting the z's
    pos = (np.array(pos).T[(z>z_low)&(z<z_high)])

    pos_r = (np.array(pos_r).T[(z_r>z_low)&(z_r<z_high)])

    weights = weights[(z>z_low)&(z<z_high)]

    weights_r = weights_r[(z_r>z_low)&(z_r<z_high)]



    #Defining s and mu bins


    edges = (np.linspace(0.001, 200.001, 201), np.linspace(0, 1., 121))

    #Making the computation

    result = TwoPointCorrelationFunction('smu', edges, data_positions1=pos, data_weights1=weights,
                                         randoms_positions1=pos_r, randoms_weights1=weights_r, position_type='pos',
                                         engine='corrfunc', compute_sepsavg=False, nthreads=threads_per_proc,mpicomm = comm,mpiroot=0)

    #Saving

    result.save(output)