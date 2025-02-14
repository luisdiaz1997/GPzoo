import torch
import matplotlib.pyplot as plt
from torch import optim, distributions, nn
from tqdm.notebook import tqdm
import gpzoo
from gpzoo.gp import GaussianPrior
from gpzoo.likelihoods import PNMF
from gpzoo.utilities import train_hybrid, train_hybrid_batched, anndata_to_train_val
import gpzoo.pri_experiment_utilities as putil
import squidpy as sq
import numpy as np
from sklearn.decomposition import NMF
import scanpy as sc
import random
from os import path
import time

from gpzoo.utilities import regularized_nmf, dims_autocorr

root_path = '/engelhardt/home/pshrestha/vnngp/'
ROOT_PATH = '/engelhardt/home/pshrestha/vnngp/'
RPATH = path.join(ROOT_PATH, "results/")
SPATH = path.join(RPATH, "slideseq/")
FPATH = path.join(SPATH, "figs/")
torch.manual_seed(256)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device

adata = sq.datasets.slideseqv2()
adata = adata.raw.to_adata()
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata.obs.pct_counts_mt.hist(bins=100)
adata = adata[adata.obs.pct_counts_mt < 20]
sc.pp.filter_cells(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=1)
idx = list(range(adata.shape[0]))
random.shuffle(idx)
adata = adata[idx]
Dtr, Dval = anndata_to_train_val(adata, sz="scanpy")
Y = Dtr['Y'].T
X = Dtr['X']*50.0

X = X.astype('float64')
Y = Y.astype('float64')

save_path = path.join(root_path, "results/slideseq/pnmf")
L = [5, 10, 15]
for l in L:
    dicts = {'L': l, 
             'lr': 1e-2,
             'rs': 256}

    nmf_save_path = path.join(SPATH, 'nmf')
    factors_path = path.join(nmf_save_path, f"nmf_factors_iter=1000_rs=256_L={dicts['L']}.npy")
    loadings_path = path.join(nmf_save_path, f"nmf_loadings_iter=1000_rs=256_L={dicts['L']}.npy")
    factors = np.load(factors_path)
    loadings = np.load(loadings_path)
    moran_idx, moranI = putil.dims_autocorr(factors, np.array(X))
    factors = factors[:, moran_idx]
    loadings = loadings[:, moran_idx]
    
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    kwargs = dicts
    file_path = 'PNMF'
    file_path += f"_L={kwargs['L']}_lr={kwargs['lr']}_iter={1000}"
    prior = GaussianPrior(Y, L=kwargs['L'])
    model = PNMF(prior, Y, L=kwargs['L'])
    PNMF.mean = factors
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])

    X_train = X.to(device)
    Y_train = Y.to(device)

    model.to(device)
    start = time.time()
    losses, means, scales = putil.train(model, optimizer, X_train, Y_train, device, steps=1000, E=1, kwargs=kwargs)
    end = time.time()
    final = end - start
    
    torch.save(model.state_dict(), f'{save_path}/{file_path}_state_dict.pth')
    torch.save({'losses': losses,
                'means': means,
                'scales': scales,
                'time': final},
               f'{save_path}/{file_path}_model.pt')
    
    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.suptitle(f"PNMF Loss")
    fig.savefig(f'{save_path}/{file_path}_loss.png')
    #fig.close()
    
    size=2
    rows = kwargs["L"]//5
    
    model.cpu()
    qF, pF = model.prior()
    mean = torch.exp(qF.mean).detach().numpy()
    if kwargs["L"] == 5:
        fig2, ax = plot_factors_five(mean, X.cpu().detach().numpy(), moran_idx=None, size=2, s=0.7, alpha=0.9)
    else:
        fig2, ax = plot_factors(mean, X.cpu().detach().numpy(), moran_idx=None, size=2, s=0.7, alpha=0.9)
    #fig.suptitle(f'Factors | sigma: {kwargs["sigma"]}, lengthscale: {kwargs["lengthscale"]}')
    fig2.savefig(f'{save_path}/{file_path}_plot.png')
    #fig.close()



