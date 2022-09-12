import numpy as np
import json
from taylor_approximation import taylor_approximate

class Emulator_Pells(object):
    
    def __init__(self,json_filename,order):
        """Sets up the class by loading the derivative matrices."""
        
        print("Loading Taylor series.")
        
        self.cpars = np.zeros(4)
        self.order = order
        
        # Load sigma8
        # self.compute_sigma8 = Compute_Sigma8(self.basedir + self.s8_filename)
        
        # Load clustering
        self.taylors_pk = {}
        # self.taylors_xi = {}
        
        # for zfid, pk_filename in zip(self.zfids, self.pk_filenames):
        # zstr = "%.2f"%(zfid)
        taylors_pk = {}

        # Load the power spectrum derivatives
        json_file = open(json_filename, 'r')
        emu = json.load( json_file )
        json_file.close()

        x0s = emu['x0']
        kvec = emu['kvec']
        derivs_p0 = [np.array(ll) for ll in emu['derivs0']]
        derivs_p2 = [np.array(ll) for ll in emu['derivs2']]
        derivs_p4 = [np.array(ll) for ll in emu['derivs4']]

        taylors_pk['x0'] = np.array(x0s)
        taylors_pk['kvec'] = np.array(kvec)
        taylors_pk['derivs_p0'] = derivs_p0
        taylors_pk['derivs_p2'] = derivs_p2
        taylors_pk['derivs_p4'] = derivs_p4


        self.taylors_pk = taylors_pk

        del emu
            
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
    
    def update_cosmo(self, cpars):
        '''If the cosmology is not the same as the old one, update the ptables.'''
        if not np.allclose(cpars, self.cpars):
            self.cpars = cpars
            
            self.p0tab = taylor_approximate(cpars,\
                                           self.taylors_pk['x0'],\
                                           self.taylors_pk['derivs_p0'], order=self.order)
            self.p2tab = taylor_approximate(cpars,\
                                           self.taylors_pk['x0'],\
                                           self.taylors_pk['derivs_p2'], order=self.order)
            self.p4tab = taylor_approximate(cpars,\
                                           self.taylors_pk['x0'],\
                                           self.taylors_pk['derivs_p4'], order=self.order)
            
    def __call__(self, cpars, bpars):
        '''Evaluate the Taylor series for the spectrum given by 'spectra'
           at the point given by 'params'.'''
        self.update_cosmo(cpars)
        
        
        pvec =  np.concatenate( ([1], bpars) )

        kvec = self.taylors_pk['kvec']
        p0, p2, p4 = self.combine_bias_terms_pkell(bpars, self.p0tab, self.p2tab, self.p4tab)

        return kvec, p0, p2, p4