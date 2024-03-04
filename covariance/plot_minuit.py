import sys, os
import jax
jax.config.update('jax_platform_name', 'cpu')
import numpy as np
import pandas as pd
import proplot as pplt
import scipy.stats as stats
from tqdm import tqdm
from desilike.profilers import MinuitProfiler
from desilike.samples.profiles import Profiles
from uncertainties import unumpy as unp, ufloat

def read_table_minuit(filename):
    df = pd.read_csv(filename, skiprows=3, sep = '|', names = ['bestfit', 'error'], skipfooter = 1, usecols = (3,4))
    return df#[['bestfit', 'error']]
def gather_results(dir, fit):
    res = []
    ids = range(0, 1000)
    for i in tqdm(ids):
        fname = f"{dir}/minuit_prof_{i:03d}.txt.npy"
        
        prof = Profiles.load(fname)
        prof = prof.choice()
        chisq = prof.bestfit.chi2min
        
        if fit == 'bao':
            res.append(np.array([ufloat(prof.bestfit['qiso'], prof.error['qiso']),
                                 ufloat(prof.bestfit['b1'], prof.error['b1']),
                                 ufloat(prof.bestfit['sigmas'], prof.error['sigmas']),
                                 ufloat(chisq, 0.),]))
        elif fit == 'bao2':
            res.append(np.array([ufloat(prof.bestfit['qiso'], prof.error['qiso']), 
                                 ufloat(prof.bestfit['qap'], prof.error['qap']),
                                 ufloat(prof.bestfit['sigmapar'], prof.error['sigmapar']),
                                 ufloat(prof.bestfit['sigmaper'], prof.error['sigmaper']),
                                 ufloat(prof.bestfit['sigmas'], prof.error['sigmas']),
                                 ufloat(prof.bestfit['b1'], prof.error['b1']),
                                 ufloat(chisq, 0.)]))
        elif fit == 'shapefit':
            res.append(np.array([ufloat(prof.bestfit['qiso'], prof.error['qiso']), 
                                 ufloat(prof.bestfit['qap'], prof.error['qap']),
                                 ufloat(prof.bestfit['dm'], prof.error['dm']),
                                 ufloat(prof.bestfit['df'], prof.error['df']),
                                 ufloat(chisq, 0.),
                                 ]))
    return np.array(res)
def single_hist(results_a, results_b, ax, pax_x, pax_y, label_a, label_b, title):
    bins = np.linspace(min(results_a.min(), results_b.min()), max(results_a.max(), results_b.max()), 25+1)
    _, bins, _ = pax_x.hist(results_a, histtype='step', bins=bins, density=True)
    pax_y.histh(results_b, histtype='step', density=True, bins = bins)
    ax.hist2d(
            results_a, results_b, bins, vmin=None, vmax=None, levels=50,
            cmap='reds', colorbar='b', colorbar_kw={'label': 'count'}
        )
    ax.scatter(results_a, results_b, markersize = 0.1)
    ax.format(xlabel = label_a, ylabel = label_b)
    if title is not None:
        ax.set_title(title)
    mean_a = results_a.mean()
    mean_b = results_b.mean()
    ax.axline((mean_a, mean_a), slope=1, ls = ':', c = 'k', lw=1)
    
    pct_diff = 100 * (results_a / results_b - 1.).mean()
    ax.text(0.05, 0.9, rf'$\langle\frac{{x}}{{y}}-1\rangle$={pct_diff:.3f}%', transform='axes', fontsize=15)
    
def plot_results(results_a, results_b, label_a, label_b, fig = None, ax = None, titles = None):

    if fig is None or ax is None:
        fig, ax = pplt.subplots(nrows = 3, ncols = 4, share = 0)
        
    pax_x = ax.panel('top', space=0)
    pax_y = ax.panel('right', space=0)
    results_a = np.atleast_2d(results_a)
    results_b = np.atleast_2d(results_b)
    for i in range(results_a.shape[1]):
        this_results_a_nominal = unp.nominal_values(results_a[:,i])
        this_results_b_nominal = unp.nominal_values(results_b[:,i])
        single_hist(this_results_a_nominal, this_results_b_nominal, ax[i,0], pax_x[i,0], pax_y[i,0], label_a, label_b, titles[i])
        
        
        this_results_a_sigma = unp.std_devs(results_a[:,i])
        this_results_b_sigma = unp.std_devs(results_b[:,i])
        single_hist(this_results_a_sigma, this_results_b_sigma, ax[i,1], pax_x[i,1], pax_y[i,1], label_a, label_b, "sigma_"+titles[i])
        add_gaussian(this_results_a_nominal, this_results_a_sigma, fig, pax_x[i,0])
        add_gaussian(this_results_b_nominal, this_results_b_sigma, fig, pax_y[i,0], vert=True)
            
            
    
    
    return fig, ax, pax_x, pax_y

def add_gaussian(par, sigma_par, fig, ax, vert = False):
    import scipy.stats as stats
    mean_par = par.mean()
    mean_sig = sigma_par.mean()
    
    x = np.linspace(mean_par - 3 * mean_sig, mean_par + 3 * mean_sig, 100)
    if not vert:
        ax.plot(x, stats.norm(loc = mean_par, scale = mean_sig).pdf(x))
    else:
        ax.plot(stats.norm(loc = mean_par, scale = mean_sig).pdf(x), x)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-fit", type = str, default = 'bao')
    parser.add_argument("-stat", type = str, default = 'pk')
    parser.add_argument("-conv", type = str, default = 'sym')
    parser.add_argument("-cubic", action = "store_true")
    parser.add_argument("-gauss_all", action = "store_true")
    
    args = parser.parse_args()
    if args.fit == 'bao':
        titles = ['qiso', 'b1', 'sigmas', 'chisq']
    elif args.fit == 'bao2':
        titles = ['qiso', 'qap', 'sigmapar', 'sigmaper', 'sigmas', 'b1', 'chisq']
    elif args.fit == 'shapefit':
        titles = ['qiso', 'qap', 'dm', 'df', 'chisq']
    cubic_suffix = "" if not args.cubic else "_cubic"
    gauss_suffix = "" if not args.gauss_all else "_gauss"
    mock_res = gather_results(f"data/desilike_mock_{args.conv}_{args.fit}_minuit_{args.stat}{cubic_suffix}{gauss_suffix}/", args.fit)
    analytic_res = gather_results(f"data/desilike_analytic_{args.conv}_{args.fit}_minuit_{args.stat}{gauss_suffix}/", args.fit)
    print(mock_res.shape)
    print(analytic_res.shape)
    
    fig, ax = pplt.subplots(nrows = mock_res.shape[1], ncols=2, share = 0)
    fig, ax, pax_x, pax_y = plot_results(mock_res, analytic_res, 'mock', 'analytic', fig = fig, ax = ax, titles = titles)
    fig.savefig(f"plots/plot_minuit_{args.stat}_{args.fit}{cubic_suffix}{gauss_suffix}.png", dpi=300)
    