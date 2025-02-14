import torch
import matplotlib.pyplot as plt
from torch import optim, distributions, nn
from tqdm.notebook import tqdm
import gpzoo
from gpzoo.gp import GaussianPrior
from gpzoo.likelihoods import PNMF
from gpzoo.utilities import train_hybrid, train_hybrid_batched, anndata_to_train_val, deviancePoisson
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
SPATH = path.join(RPATH, "merfish/")
FPATH = path.join(SPATH, "figs/")
torch.manual_seed(256)
dpth = path.join(root_path, "data/")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device

merpath = path.join(dpth, "merfish.h5ad")
ad = sc.read_h5ad(merpath)
ad = ad[(ad.obs['percent.mt'] < 5), :]
min_genes_per_cell = 500
ad = ad[ad.obs['nFeature_RNA'] >= min_genes_per_cell, :]
min_value = np.min(ad.X)
if min_value < 0:
    ad.X += np.abs(min_value)
ad.var_names_make_unique()
labels = ad.obs.Age.unique()
ad.obs['Age_cat'] = ad.obs['Age'].astype('category')
ad = ad[~ad.obs['Age'].isin([20.8, 22.6, 3.33])]
Y_sums = np.array(np.sum(ad.raw.X > 0, axis=0))[0]
Y = np.array(ad.X, dtype=int).T
X = ad.obsm['X_harmony']
X = X.astype('float32')
Y = Y.astype('float32')
#X = rescale_spatial_coords(X)
ad.uns['counts'] = ad.X.copy()
    
ad.var['deviance_poisson'] = deviancePoisson(ad.uns["counts"])
o = np.argsort(-ad.var['deviance_poisson'])
idx = list(range(ad.shape[0]))
random.shuffle(idx)
ad = ad[idx,o]
    
X = np.array(ad.obs['Age'])[:, None]
X = torch.tensor(X, dtype=torch.float)
    
Y = np.log(np.exp(Y) + 1e-2)  # correction
Y = Y.T

print("X shape: ", X.shape)

save_path = path.join(root_path, "results/merfish/pnmf")
L = [6, 12, 18]
for l in L:
    dicts = {'L': l, 
             'lr': 1e-2,
             'rs': 256}

    nmf_save_path = path.join(SPATH, 'nmf')
    factors_path = path.join(nmf_save_path, f"3000/nmf_factors_iter=1000_rs=256_L={dicts['L']}_rm(20.8, 22.6, 3.33)_new.npy")
    loadings_path = path.join(nmf_save_path, f"3000/nmf_loadings_iter=1000_rs=256_L={dicts['L']}_rm(20.8, 22.6, 3.33)_new.npy")
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
    
    #size=2
    #rows = kwargs["L"]//5
    
    #model.cpu()
    #qF, pF = model.prior()
    #mean = torch.exp(qF.mean).detach().numpy()
    #if kwargs["L"] == 5:
        #fig2, ax = plot_factors_five(mean, X.cpu().detach().numpy(), moran_idx=None, size=2, s=0.7, alpha=0.9)
    #else:
       # fig2, ax = plot_factors(mean, X.cpu().detach().numpy(), moran_idx=None, size=2, s=0.7, alpha=0.9)
    #fig.suptitle(f'Factors | sigma: {kwargs["sigma"]}, lengthscale: {kwargs["lengthscale"]}')
    #fig2.savefig(f'{save_path}/{file_path}_plot.png')
    #fig.close()



