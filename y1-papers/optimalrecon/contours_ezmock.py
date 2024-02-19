from desilike.samples import Chain
from pathlib import Path
from getdist import plots as gdplt
from desilike.samples import plotting
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})


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
    alpha_iso = chain['qiso'].clone(param=dict(basename='alpha_iso', derived=True, latex=r'\alpha_{\rm iso}'))
    alpha_ap = chain['qap'].clone(param=dict(basename='alpha_ap', derived=True, latex=r'\alpha_{\rm AP}'))
    alpha_par = chain['qpar'].clone(param=dict(basename='alpha_par', derived=True, latex=r'\alpha_{\parallel}'))
    alpha_per = chain['qper'].clone(param=dict(basename='alpha_per', derived=True, latex=r'\alpha_{\perp}'))
    chain.set(alpha_iso)
    chain.set(alpha_ap)
    chain.set(alpha_par)
    chain.set(alpha_per)
    return chain

colors = {'BGS_BRIGHT-21.5': 'black', ('BGS_BRIGHT-21.5', (0.1, 0.4)): 'black',
          'LRG': 'red', ('LRG', (0.4, 0.6)): 'orange', ('LRG', (0.6, 0.8)): 'orangered', ('LRG', (0.8, 1.1)): 'firebrick', 
          ('ELG_LOPnotqso', (0.8, 1.1)): 'lightskyblue', ('ELG_LOPnotqso', (1.1, 1.6)): 'steelblue',
          ('QSO', (0.8, 2.1)): 'seagreen', ('Lya', (0.8, 3.5)): 'purple'}


zmin, zmax = 0.8, 1.1
tracer = 'LRG'
apmode = 'qisoqap'

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/Y1/EZmocks/v1/fits_bao/mean_precscale1/fits_correlation_qisoqap_pcs2/recon_IFT_recsym_sm15/'
data_fn = [Path(data_dir) / f'chain_LRG_GCcomb_0.4_0.6_qisoqap_pcs2_sigmas2.0_sigmapar6.0_sigmaper3.0_IFTrecsym_sm15_{i}.npy' for i in range(8)]
chain1 = read_desilike_chain(data_fn, apmode=apmode)

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/Y1/EZmocks/v1/fits_bao/mean_precscale1/fits_correlation_qisoqap_pcs2/recon_IFT_recsym_sm15/'
data_fn = [Path(data_dir) / f'chain_LRG_GCcomb_0.6_0.8_qisoqap_pcs2_sigmas2.0_sigmapar6.0_sigmaper3.0_IFTrecsym_sm15_{i}.npy' for i in range(8)]
chain2 = read_desilike_chain(data_fn, apmode=apmode)

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/Y1/EZmocks/v1/fits_bao/mean_precscale1/fits_correlation_qisoqap_pcs2/recon_IFT_recsym_sm15/'
data_fn = [Path(data_dir) / f'chain_LRG_GCcomb_0.8_1.1_qisoqap_pcs2_sigmas2.0_sigmapar6.0_sigmaper3.0_IFTrecsym_sm15_{i}.npy' for i in range(8)]
chain3 = read_desilike_chain(data_fn, apmode=apmode)

data_dir = f'/global/cfs/cdirs/desicollab/users/epaillas/Y1/EZmocks/v1/fits_bao/mean_precscale1/fits_correlation_qisoqap_pcs2/recon_IFT_recsym_sm15/'
data_fn = [Path(data_dir) / f'chain_ELG_LOP_GCcomb_1.1_1.6_qisoqap_pcs2_sigmas2.0_sigmapar6.0_sigmaper3.0_IFTrecsym_sm15_{i}.npy' for i in range(8)]
chain4 = read_desilike_chain(data_fn, apmode=apmode)

chains = [chain4, chain1, chain2, chain3,]

g = gdplt.get_subplot_plotter(width_inch=6)
g.settings.axis_marker_lw = 1.0
g.settings.axis_marker_ls = ':'
g.settings.axis_marker_color = 'k'
g.settings.legend_colored_text = True
g.settings.figure_legend_frame = False
g.settings.linewidth = 2.0
g.settings.linewidth_contour = 3.0
g.settings.legend_fontsize = 20
g.settings.axes_fontsize = 15
g.settings.axes_labelsize = 18
g.settings.solid_colors = ['firebrick', 'orangered', 'orange', 'steelblue']

plotting.plot_triangle(
    chains,
    g=g,
    filled=True,
    legend_labels=['ELG 1.1-1.6', 'LRG 0.4-0.6', 'LRG 0.6-0.8', 'LRG 0.8-1.1'],
    legend_loc='upper right',
    params=['alpha_iso', 'alpha_ap', 'alpha_par', 'alpha_per'],
    markers={'alpha_iso': 1., 'alpha_ap': 1, 'alpha_par': 1, 'alpha_per': 1},
    title_limit=0,
)

plt.savefig(f'fig/fits_bao_ezmock.png', bbox_inches='tight', dpi=300)
plt.savefig(f'fig/fits_bao_ezmock.pdf', bbox_inches='tight', format='pdf')
plt.show()
