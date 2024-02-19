import numpy as np
from desilike.samples import Profiles
from pathlib import Path
from tabulate import tabulate


base_dir = '/global/homes/e/epaillas/desi/users/epaillas/Y1/mocks/SecondGenMocks/AbacusSummit'
param_names = ['qiso', 'qap']
param_labels = [r'$\alpha_{\rm iso}$', r'$\alpha_{\rm AP}$', r'$\alpha_\parallel$', r'$\alpha_\perp$']

zranges = {'BGS_BRIGHT-21.5': [[0.1, 0.4]], 'LRG': [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]], 'ELG_LOPnotqso': [[0.8, 1.1], [1.1, 1.6]], 'QSO': [[0.8, 2.1]]}
smoothing_scales = {'BGS_BRIGHT-21.5': 15, 'LRG': 15, 'ELG_LOPnotqso': 15, 'QSO': 30,}

sigmapar = {'BGS_BRIGHT-21.5':{'pre': 10.0, 'post': 8.0}, 'LRG':{'pre': 9.0, 'post': 6.0}, 'ELG_LOPnotqso':{'pre': 8.5, 'post': 6.0}, 'QSO': {'pre': 9.0, 'post': 6.0}}
sigmaper = {'BGS_BRIGHT-21.5':{'pre':6.5, 'post':3.0}, 'LRG':{'pre':4.5, 'post':3.0}, 'ELG_LOPnotqso':{'pre': 4.5, 'post': 3.0}, 'QSO': {'pre': 3.5, 'post': 3.0}}
sigmas = {'BGS_BRIGHT-21.5':{'pre': 2.0, 'post': 2.0}, 'LRG':{'pre': 2.0, 'post': 2.0}, 'ELG_LOPnotqso':{'pre': 2.0, 'post': 2.0}, 'QSO': {'pre': 2.0, 'post': 2.0}}

apmodes = {('BGS_BRIGHT-21.5', (0.1, 0.4)): 'qiso',
          ('LRG', (0.4, 0.6)): 'qisoqap', ('LRG', (0.6, 0.8)): 'qisoqap', ('LRG', (0.8, 1.1)): 'qisoqap', 
          ('ELG_LOPnotqso', (0.8, 1.1)): 'qiso', ('ELG_LOPnotqso', (1.1, 1.6)): 'qisoqap',
          ('QSO', (0.8, 2.1)): 'qiso'}

headers = [
    'Tracer',
    'Redshift',
    'Recon',
    r'$\langle \alpha_{\rm iso} \rangle$',
    r'$\langle \sigma_{\alpha_{\rm iso}} \rangle$',
    r'$\langle \alpha_{\rm AP} \rangle$',
    r'$\langle \sigma_{\alpha_{\rm AP}} \rangle$',
]

bestfit_tracer = []
for tracer in ['BGS_BRIGHT-21.5', 'LRG', 'ELG_LOPnotqso', 'QSO']:
    version = 'v1/altmtl' if tracer.startswith('BGS') else 'v3_1/altmtl'
    sm = smoothing_scales[tracer]
    bestfit_redshift = []
    for zrange in zranges[tracer]:
        zmin, zmax = zrange
        apmode = apmodes[(tracer,  (zmin, zmax))]
        sm = smoothing_scales[tracer]
        for rec in ['pre', 'post']:
            bestfit_phases = []
            for phase in range(0, 25):
                if rec == 'post':
                    profiles_dir = f'{base_dir}/{version}/fits_bao/cov_test/RascalC_blinded/{phase}/fits_correlation_{apmode}_pcs2/recon_IFFT_recsym_sm{sm}/'
                elif rec == 'pre':
                    profiles_dir = f'{base_dir}/{version}/fits_bao/cov_test/RascalC_blinded/{phase}/fits_correlation_{apmode}_pcs2/'
                fn = profiles_dir + f"profiles_{tracer}_GCcomb_{zmin}_{zmax}_sigmas{sigmas[tracer][rec]}_sigmapar{sigmapar[tracer][rec]}_sigmaper{sigmaper[tracer][rec]}.npy"
                profiles = Profiles.load(fn)
                bestfit_qiso = profiles.bestfit['qiso'][0]
                error_qiso = profiles.error['qiso'][0]
                if apmode == 'qisoqap':
                    bestfit_qap = profiles.bestfit['qap'][0]
                    error_qap = profiles.error['qap'][0]
                    values = [1 - bestfit_qiso, error_qiso, 1 - bestfit_qap, error_qap]
                else:
                    values = [1 - bestfit_qiso, error_qiso, np.nan, np.nan]
                bestfit_phases.append(values)
            bestfit_phases = np.nanmean(bestfit_phases, axis=0)
            bestfit_redshift.append([rf'{tracer}'.replace('_LOPnotqso', r'').replace('_BRIGHT-21.5', '')] + [f'{zmin}-{zmax}'] + [rec.title()] + list(bestfit_phases))
    bestfit_tracer.append(bestfit_redshift)
table = np.concatenate(bestfit_tracer)
print(tabulate(table, tablefmt='latex_raw', headers=headers, floatfmt=".4f"))
