import numpy as np
from desilike.samples import Chain, Profiles
from getdist import plots as gdplt
from pathlib import Path
from tabulate import tabulate


def read_desilike_chain(filename, apmode='qiso'):
    if isinstance(filename, list):
        chains = []
        for fn in filename:
            chains.append(Chain.load(fn))
        chain = chains[0].concatenate([chain.remove_burnin(0.5)[::10] for chain in chains])
    else:
        chain = Chain.load(filename)
        chain = chain.remove_burnin(0.5)[::10]
    print(len(chain))
    if apmode == 'qparqper':
        qiso = (chain['qpar']**(1./3.) * chain['qper']**(2./3.)).clone(param=dict(basename='qiso', derived=True, latex=r'q_{\rm iso}'))
        qap = (chain['qpar'] / chain['qper']).clone(param=dict(basename='qap', derived=True, latex=r'q_{\rm AP}'))
        chain.set(qiso)
        chain.set(qap)
    if apmode == 'qisoqap':
        qpar = (chain['qiso'] * chain['qap']**(2/3)).clone(param=dict(basename='qpar', derived=True, latex=r'q_{\parallel}'))
        qper = (chain['qiso'] * chain['qap']**(-1/3)).clone(param=dict(basename='qper', derived=True, latex=r'q_{\perp}'))
        chain.set(qpar)
        chain.set(qper)
    return chain


zranges = {
    'BGS_BRIGHT-21.5': [[0.1, 0.4]],
    'LRG': [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]],
    r'ELG_LOP': [[0.8, 1.1], [1.1, 1.6]],
    'QSO': [[0.8, 2.1]],
}

smoothing_scales = {
    'BGS_BRIGHT-21.5': 15,
    'LRG': 15,
    r'ELG_LOP': 15,
    'QSO': 30,
}

sigmapar = {'BGS_BRIGHT-21.5':{'pre': 10.0, 'post': 8.0}, 'LRG':{'pre': 9.0, 'post': 6.0}, 'ELG_LOP':{'pre': 8.5, 'post': 6.0}, 'QSO': {'pre': 9.0, 'post': 6.0}}
sigmaper = {'BGS_BRIGHT-21.5':{'pre':6.5, 'post':3.0}, 'LRG':{'pre':4.5, 'post':3.0}, 'ELG_LOP':{'pre': 4.5, 'post': 3.0}, 'QSO': {'pre': 3.5, 'post': 3.0}}
sigmas = {'BGS_BRIGHT-21.5':{'pre': 2.0, 'post': 2.0}, 'LRG':{'pre': 2.0, 'post': 2.0}, 'ELG_LOP':{'pre': 2.0, 'post': 2.0}, 'QSO': {'pre': 2.0, 'post': 2.0}}

apmodes = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 'qiso',
          ('LRG', (0.4, 0.6)): 'qisoqap', ('LRG', (0.6, 0.8)): 'qisoqap', ('LRG', (0.8, 1.1)): 'qisoqap', 
          ('ELG_LOP', (0.8, 1.1)): 'qiso', ('ELG_LOP', (1.1, 1.6)): 'qisoqap',
          ('QSO', (0.8, 2.1)): 'qiso'}

headers = [
    'Tracer',
    'Redshift',
    'Recon',
    r'$1 - \alpha_{\rm iso}$',
    r'$\sigma_{\alpha_{\rm iso}}$',
    r'$1 - \alpha_{\rm AP}$',
    r'$\sigma_{\alpha_{\rm AP}}$',
]

bestfit_tracer = []
for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOP', 'QSO']:
    sm = smoothing_scales[tracer]
    bestfit_redshift = []
    for zrange in zranges[tracer]:
        zmin, zmax = zrange
        zmid = float(f'{(zmin + zmax)/2 : .2f}')
        apmode = apmodes[(tracer,  (zmin, zmax))]
        sm = smoothing_scales[tracer]
        for rec in ['pre', 'post']:
            if rec == 'pre':
                chain_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/Y1/mocks/EZmocks/v1/fits_bao/mean_precscale1/fits_correlation_{apmode}_pcs2/'
            else:
                chain_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/Y1/mocks/EZmocks/v1/fits_bao/mean_precscale1/fits_correlation_{apmode}_pcs2/recon_IFT_recsym_sm{sm}/'
            chain_fn = [Path(chain_dir) / f'chain_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer][rec]}_sigmapar{sigmapar[tracer][rec]}_sigmaper{sigmaper[tracer][rec]}_{i}.npy' for i in range(8)]
            chain = read_desilike_chain(chain_fn, apmode=apmode)
            if rec == 'post':
                print(tracer, zmin, zmax, (chain.std('qiso') - error_qiso)/error_qiso)
                if apmode == 'qisoqap':
                    print(tracer, zmin, zmax, (chain.std('qap') - error_qap)/error_qap)
            mean_qiso = chain.mean('qiso')
            error_qiso = chain.std('qiso')
            if apmode == 'qisoqap':
                mean_qap = chain.mean('qap')
                error_qap = float(chain.std('qap'))
                values = [1 - mean_qiso, error_qiso, 1 - mean_qap, error_qap]
            else:
                values = [1 - mean_qiso, error_qiso, None, None]
            bestfit_redshift.append([rf'{tracer}'.replace('_LOP', r'').replace('_BRIGHT-21.5', '')] + [f'{zmin}-{zmax}'] + [rec.title()] + values)
    bestfit_tracer.append(bestfit_redshift)
table = np.concatenate(bestfit_tracer)
print(tabulate(table, tablefmt='latex_raw', headers=headers, floatfmt=".4f"))
