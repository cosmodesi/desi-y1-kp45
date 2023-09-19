import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import jax.numpy as jnp
import jax

def plot_contours(fisher, pos,  nstd=1., ax=None, **kwargs):
    """
    Plot 2D parameter contours given a Hessian matrix of the likelihood
    """
    
    def eigsorted(cov):
      vals, vecs = np.linalg.eigh(cov)
      order = vals.argsort()[::-1]
      return vals[order], vecs[:, order]

    mat = fisher
    cov = np.linalg.inv(mat)
    
    sigma_marg = lambda i: np.sqrt(cov[i, i])

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width,
                    height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    sz = max(width, height)
    s1 = 1.5*nstd*sigma_marg(0)
    s2 = 1.5*nstd*sigma_marg(1)
    
    ax.set_xlim(pos[0] - s1, pos[0] + s1)
    ax.set_ylim(pos[1] - s2, pos[1] + s2)
    plt.draw()
    return ellip

def samples_to_covariance(xis, rescale = False):
    if xis.ndim == 3:
      xis = np.concatenate(tuple(map(np.squeeze, np.array_split(xis, xis.shape[1], axis=1))), axis=-1)
      print(xis.shape)
    N_mocks, N_bins = xis.shape 
    print("Nmocks", N_mocks)
    print("Nbins", N_bins)
    mean = xis.mean(axis=0)
    error = xis - mean[None, :] 
    #sample_cov = error.T.dot(error)
    sample_cov = np.cov(xis, rowvar=False)
    hartlap =  (N_mocks - 1) / (N_mocks - N_bins - 2) 
    print(hartlap**-1)
    hartlap = 1
    cov_unbiased = sample_cov * hartlap
    if rescale: cov_unbiased /= N_mocks
    std = np.sqrt(np.diag(cov_unbiased))
    corr = cov_unbiased / (std[:,None] * std[None,:])
    return cov_unbiased, mean, corr, xis

def get_interp_pk(filename):
    k, pk = np.loadtxt(filename, usecols=(0,1), unpack=True)
    pk_interp = lambda kv: jnp.exp(jnp.interp(jnp.log(kv), jnp.log(k), jnp.log(pk)))
    return pk_interp

def mask_fit_range(smin, smax, s_obs, xi_obs, inv_cov, cov):
    fit_mask = (s_obs > smin) & (s_obs < smax)
    s_obs = s_obs[fit_mask]
    xi_obs = xi_obs[fit_mask]
    inv_cov = inv_cov[fit_mask, :][:,fit_mask]
    cov = cov[fit_mask, :][:,fit_mask]

    return s_obs, xi_obs, inv_cov, cov


def read_xis(list, output, usecols = (0, 3, 4, 5)):
  import os
  from tqdm import tqdm

  if not os.path.isfile(output) or 1:
    xis = []
    for f in (list):
      xis.append(np.loadtxt(f))
    
    xis = np.array(xis)
    s = xis[0,:,0]
    xis = xis[:,:,3:]
    dset = dict(s=s, xi = xis)
    np.savez(output, **dset)   
    print(xis.shape) 
  else:
    dset = np.load(output)
  return dset

def read_xis_pycorr(list, list_b = None, slice = slice(0,None,4), ells = (0,2,4), smin = 0, smax = 200):
  import os
  from tqdm import tqdm
  from pycorr import TwoPointCorrelationFunction
  assert len(list) > 0
  if list_b is not None:
    assert len(list) == len(list_b)
    print("WARNING: Assuming both lists have been passed with the right order for combinations.")

  xis = []
  for i, f in tqdm(enumerate((list))):
    result = TwoPointCorrelationFunction.load(f)[slice].select((smin, smax))
    if list_b is not None: result = result.normalize() + (TwoPointCorrelationFunction.load(list_b[i])[slice].select((smin, smax))).normalize()
    s, xiell = result(ells=ells, return_sep=True)
    xis.append(np.vstack(xiell))
    
    
  xis = np.array(xis)
  dset = dict(s=s, xi = xis)
  print(xis.shape) 
  
  return dset

      


def read_pks_pypower(list, list_b = None, slice = slice(0,None,1), ells = (0,2,4), kmin = 0, kmax = 0.3):
  import os
  from tqdm import tqdm
  from pypower import CatalogFFTPower
  assert len(list) > 0
  if list_b is not None:
    assert len(list) == len(list_b)
    print("WARNING: Assuming both listsxi have been passed with the right order for combinations.")

  xis = []
  for i, f in enumerate(tqdm(list)):
    result = CatalogFFTPower.load(f).poles[slice].select((kmin, kmax))
    if list_b is not None: result += CatalogFFTPower.load(list_b[i]).poles[slice].select((kmin, kmax))
    s, xiell = result(ell=ells, return_k=True)
    xis.append(np.vstack(xiell))
  xis = np.array(xis).real
  print(xis.shape)
  dset = dict(k=s, pk = xis)
  
  print(xis.shape) 
  
  return dset




def combine_matrices(mat1, mat2, buffer = 1):
    print(f"Combining matrices. upper triangle is mat1, and lower is mat2")
    mat1_size = mat1.shape[0]
    mat2_size = mat2.shape[0]
    max_size_id = np.argmax([mat1_size, mat2_size])
    mat = np.zeros([x+buffer for x in mat1.shape])
    mat[:] = np.nan
    upper_indices = np.triu_indices(mat1.shape[0], 0)
    lower_indices = np.tril_indices(mat2.shape[0], 0)
    size_diff = mat1_size - mat2_size
    #mat = mat1.copy()
    mat[np.triu_indices(mat.shape[0], buffer)] = mat1[upper_indices]
    mat[np.tril_indices(mat.shape[0], -buffer)] = mat2[lower_indices]
    
    return mat
  
def kl_div(cov1, cov2):

    
    id = np.trace(cov2.dot(jnp.linalg.inv(cov1)))
    k = cov1.shape[0]
    #print(id - k)
    # We assume the means are the same so the second term here is 0 https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    #print(np.linalg.det(np.array(cov2).astype(np.float64)))
    #print(np.linalg.det(np.array(cov1).astype(np.float64)))
    L1 = jnp.linalg.cholesky(cov1)
    L2 = jnp.linalg.cholesky(cov2)
    #print(2 * np.sum(np.diag(L1)))
    #print(2 * np.sum(np.diag(L2)))
    dkl = 0.5 * (id - k + 2 * np.sum(np.diag(L2)) - 2 * np.sum(np.diag(L1)))
    
    return dkl / jnp.log(2)