import torch
import matplotlib.pyplot as plt
from torch import optim, distributions, nn
import torch.nn.utils as nn_utils
from tqdm import tqdm
from gpzoo.gp import SVGP, VNNGP, GaussianPrior
from gpzoo.kernels import NSF_RBF, RBF
from gpzoo.likelihoods import NSF2, PNMF
from gpzoo.utilities import rescale_spatial_coords, dims_autocorr, regularized_nmf, anndata_to_train_val, add_jitter, scanpy_sizefactors #init_softplus
import squidpy as sq
import numpy as np
import time
import random
import scanpy as sc
import anndata as ad
from anndata import AnnData
from squidpy.gr import spatial_neighbors,spatial_autocorr
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from torch.distributions import Normal

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

def original_vnngp_kl(mz, Lz, Lu, mu, verbose=False):
    if verbose == True:
        print(f"mz shape: {mz.shape}")
        print(f"Lz shape: {Lz.shape}")
        print(f"Lu shape: {Lu.shape}")
        print(f"mu shape: {mu.shape}")
    M = mz.shape[0]
    #print(f"KL M: {M}")
    kl_divergence_sum=0

    for j in range(M):
        # Compute necessary parameters for each inducing point
        mj = mz[j]
        sj = Lz[j, j] ** 2  # Assuming Lz is the Cholesky factor of the covariance matrix
        bj = Lz[:, j]
        fj = bj[j]
        sn_j = torch.diag(Lz) ** 2
        mn_j = mu

        if verbose == True:
            print(f"KL shape mj: {mj.shape}")
            print(f"KL shape sj: {sj.shape}")
            print(f"KL shape bj: {bj.shape}")
            print(f"KL shape fj: {fj.shape}")
            print(f"KL shape sn_j: {sn_j.shape}")
            print(f"KL shape mN_j: {mn_j.shape}")

        # Compute the KL divergence for the j-th inducing point
        log_fj = torch.log(fj)
        log_sj = torch.log(sj)

        kl_terms = log_fj - log_sj - 1 + fj.reciprocal() * (sj + (bj**2).sum() * sn_j + (mj - bj @ mn_j)**2)

        kl_divergence_sum += 0.5 * kl_terms.sum()
        
    return kl_divergence_sum

import torch

def vnngp_kl(mz, Lz, Lu, mu, verbose=False):
    """
    Compute the KL divergence term for the VNNGP model.

    Parameters:
    mz : torch.Tensor
        Mean vector of the variational distribution.
    Lz : torch.Tensor
        Cholesky factor of the covariance matrix of the variational distribution.
    Lu : torch.Tensor
        Cholesky factor of the prior covariance matrix.
    mu : torch.Tensor
        Mean vector of the inducing points.
    verbose : bool
        If True, print the shapes of the tensors.

    Returns:
    torch.Tensor
        The computed KL divergence value.
    """
    if verbose:
        print(f"mz shape: {mz.shape}")
        print(f"Lz shape: {Lz.shape}")
        print(f"Lu shape: {Lu.shape}")
        print(f"mu shape: {mu.shape}")

    # Number of inducing points
    M = mz.shape[0]

    # Compute the necessary parameters for all inducing points
    sj = torch.diag(Lz) ** 2  # Variance of the variational distribution
    bj = Lz  # Coefficients for the conditional distribution
    fj = torch.diag(bj)  # Conditional variance for the inducing points
    sn_j = sj  # Variance of the variational approximation for the inducing points

    # Compute the mean and variance terms for the KL divergence
    log_fj = torch.log(fj)
    log_sj = torch.log(sj)

    # Compute the KL divergence terms in a vectorized manner
    kl_terms = log_fj - log_sj - 1 + fj.reciprocal() * (sj + torch.sum(bj**2, dim=0) * sn_j + (mz - torch.matmul(bj.T, mu))**2)

    # Sum the KL divergence terms and scale by 0.5
    kl_divergence_sum = 0.5 * torch.sum(kl_terms) 

    return kl_divergence_sum


def make_synthetic_data(D=3, N=3000):
    X = Normal(0, 1.0).sample((N,))*10
    Y = torch.zeros((D, N))
    for i in range(D):
        Y[i] = (i+1)*torch.sin(2**(i-1)*X) + Normal(0, 0.1).sample((N,))
    X = X[:, None]
    return X,Y

def init_softplus(mat, minval= 1e-5):
    mat2 = mat.copy()
    mask = mat2<20
    mat2[mask] = np.log(np.exp(mat2[mask])-1+minval)

    return mat2


def load_visium():
    adata = sq.datasets.visium_hne_adata()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    Y_sums = np.array(np.sum(adata.raw.X > 0, axis=0))[0]
    
    Y = np.array(adata.raw.X[:, Y_sums>200].todense(), dtype=int).T
    X = adata.obsm['spatial']
    X = X.astype('float64')
    Y = Y.astype('float64')
    X = rescale_spatial_coords(X)
    
    return X, Y

 
def load_slideseq():
    adata = sq.datasets.slideseqv2()
    adata = adata.raw.to_adata()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20] #from 53K to 45K
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=1)
    idx = list(range(adata.shape[0]))
    random.shuffle(idx)
    adata = adata[idx]
    
    Dtr, Dval = anndata_to_train_val(adata, sz="scanpy")
    Y = Dtr['Y'].T
    Y = Y[~adata.var.MT]
    X = Dtr['X']*50.0
    V = Dtr['sz']
    
    return X, (Y/V.squeeze())
    
def preloaded_nmf_factors(path):
    nmf_results = torch.load(path)
    factors = np.array(nmf_results['factors'])
    loadings = np.array(nmf_results['loadings'])
    
    return factors, loadings

def inducing_points_cluster_centers(X, M, random_state=256):
    return KMeans(n_clusters=M, random_state=random_state, n_init="auto").fit(X)
    
def build_model(X, Y, loadings=None, factors=None, model_type=None, **kwargs):
    """
    """
    V = scanpy_sizefactors(Y.T)
    kwargs = kwargs['kwargs']

    kmeans = inducing_points_cluster_centers(X, kwargs['M'], random_state=kwargs['rs'])
    Z = nn.Parameter(torch.tensor(kmeans.cluster_centers_).type(torch.float))
    kernel = NSF_RBF(L=kwargs['L'], sigma=kwargs['sigma'], lengthscale=kwargs['lengthscale'])
    
    gp = None
    if model_type == 'VNNGP':
        gp = VNNGP(kernel, M=kwargs['M'], jitter=kwargs['jtr'], K=kwargs['K'])
    elif model_type == 'SVGP':
        gp = SVGP(kernel, M=kwargs['M'], jitter=kwargs['jtr'])

    X = torch.tensor(X).type(torch.float)
    Y = torch.tensor(Y).type(torch.float)
    Kzx = kernel.forward(Z.to('cpu'), X.to('cpu'))
    Kxz = torch.transpose(Kzx, -2, -1)
    Kzz = kernel.forward(Z, Z)

    all_factors = torch.tensor(factors.T)[:, :, None].type(torch.float)
    L1 = torch.linalg.cholesky(add_jitter(Kzx @ Kxz, 1e-4))
    
    solved1 = torch.cholesky_solve(Kzx @ all_factors, L1)
    mu = (Kzz @ solved1).to(device)
    gp.mu = nn.Parameter(torch.squeeze(mu).clone().detach()).type(torch.float)
    Lu = 1e-2 * torch.eye(kwargs['M']).expand(kwargs['L'], kwargs['M'], kwargs['M'])
    gp.Lu = nn.Parameter(Lu.clone().detach())
    
    gp.Z = nn.Parameter(Z, requires_grad=False)

    model = NSF2(gp, Y, L=kwargs['L'])

    if loadings is not None:
        model.W = nn.Parameter(torch.tensor(init_softplus(loadings)[:, :kwargs['L']]).type(torch.float))
    model.V = nn.Parameter(torch.squeeze(torch.tensor(init_softplus(V)).type(torch.float)))
    
    model.to(device)
    return model

def buil_pnmf(Y, L):
    L = 15
    prior = GaussianPrior(Y, L=L)
    model = PNMF(prior, Y, L=L)
    model.to(device)

def model_grads(model):
    model.prior.kernel.sigma.requires_grad = True
    model.prior.kernel.lengthscale.requires_grad = True
    model.prior.Z.requires_grad=True
    model.prior.mu.requires_grad=False
    model.prior.Lu.requires_grad=True
    model.W.requires_grad=True
    
    model.V.requires_grad=False

# Plotting Functions
def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    max_val = np.percentile(factors, 95)
    min_val = np.percentile(factors, 5)
    
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

    if ax is None:
        fig, ax = plt.subplots(2, 5, figsize=(size*5, size*2), tight_layout=True)
        
    for i in range(L):
        
        curr_ax = ax[i//5, i%5]
        
        curr_ax.scatter(X[:, 0], X[:,1], c=factors[i], vmin=min_val, vmax=max_val, alpha=alpha, cmap='turbo', s=s)

        curr_ax.invert_yaxis()
        if names is not None:
            curr_ax.set_title(names[i], x=0.03, y=.88, fontsize="small", c="white",
                     ha="left", va="top")
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_facecolor('xkcd:gray')

def train(model, optimizer, y, device, steps=200, E=20, **kwargs):
    losses = []
    means = []
    scales = []
    
    for it in tqdm(range(steps)):
        
        optimizer.zero_grad()
        pY, qF, pF = model.forward(E=E, **kwargs)

        logpY = y*torch.log(pY.rate) - pY.rate

        ELBO = (logpY).mean(axis=0).sum()
        ELBO -= torch.sum(distributions.kl_divergence(qF, pF))

        loss = -ELBO
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (it%10)==0:
            means.append(torch.exp(qF.mean.detach().cpu()).numpy())
            scales.append(qF.scale.detach().cpu().numpy())
    
    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    return losses, means, scales


def train_hybrid_batched(model, optimizer, X, y, device, steps=200, E=20, verbose=False, batch_size=1000, **kwargs):
    kwargs = kwargs['kwargs']
    losses = []
    means = []
    scales = []
    idxs = []

    for it in tqdm(range(steps)):
        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)
        optimizer.zero_grad()
        pY, _ , qU, pU, qF, pF = model.forward_batched(X=X, idx=idx, E=E, **kwargs)
        
        logpY = y[:, idx]*torch.log((pY.rate).cpu()) - (pY.rate).cpu()


        ELBO = ((logpY).mean(axis=0).sum()).to(device)
        ELBO -= torch.sum(torch.vmap(vnngp_kl)(qU.mean, qU.scale_tril, model.prior.Lu, model.prior.mu))

        loss = -ELBO
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (it%10)==0:
            means.append(torch.exp(qF.mean.detach().cpu()).numpy())
            scales.append(qF.scale.detach().cpu().numpy())

    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()

    return losses, means, scales, idxs


def train_batched(model, optimizer, X, y, device, steps=200, E=20, verbose=False, batch_size=1000, **kwargs):
    kwargs = kwargs['kwargs']
    losses = []
    means = []
    scales = []
    idxs = []

    for it in tqdm(range(steps)):
        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)
        optimizer.zero_grad()
        pY, qF, qU, pU = model.forward_batched(X=X, idx=idx, E=E, verbose=verbose, kwargs=kwargs)
        
        logpY = y[:, idx]*torch.log(pY.rate) - (pY.rate)


        ELBO = ((logpY).mean(axis=0).sum()).to(device)
        ELBO -= torch.sum(torch.vmap(vnngp_kl)(qU.mean, qU.scale_tril, model.prior.Lu, model.prior.mu))

        loss = -ELBO
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (it%10)==0:
            means.append(torch.exp(qF.mean.detach().cpu()).numpy())
            scales.append(qF.scale.detach().cpu().numpy())

    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()

    return losses, means, scales, idxs


def train_batched_old(model, optimizer, X, y, device, steps=200, E=20, batch_size=1000, L=10, **kwargs):
    losses = []
    
    means = []
    scales = []
    idxs = []
    
    for it in tqdm(range(steps)):
        
        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)
        
        
        optimizer.zero_grad()
        pY, qF, qU, pU = model.forward_batched(X=X, idx=idx, E=E, **kwargs)

        logpY = y[:, idx]*torch.log(pY.rate) - pY.rate

        ELBO = (logpY).mean(axis=0).sum()
        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
        
        loss = -ELBO
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (it%10)==0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(torch.exp(qF.mean.detach().cpu()).numpy())
            scales.append(qF.scale.detach().cpu().numpy())
    
    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    return losses, means, scales, idxs



def run_experiment(data_func, save_path, steps=1000, batched=False, model_type=None, NMF=False, nmf_path=None, **kwargs):
    kwargs = kwargs['kwargs']
    file_path = model_type
    X, Y = data_func()
    #K=None
    if model_type == 'VNNGP':
        #K=kwargs['K']
        file_path += f"_K={kwargs['K']}"

        if kwargs['lkzz_build']:
            file_path += f"_lkzz={kwargs['lkzz_build']}"
        
    file_path += f"_M={kwargs['M']}_L={kwargs['L']}_lr={kwargs['lr']}_jtr={kwargs['jtr']}_iter={steps}"
    if batched:
        file_path += f"_bs={kwargs['batch_size']}"
    
    factors = None
    loadings = None
    
    if NMF:
        # with NMF initialization
        file_path += f"_NMFinit"
        factors, loadings = preloaded_nmf_factors(nmf_path)
        X = np.array(X)
        moran_idx, moranI = dims_autocorr(factors, X)
        factors=factors[:, moran_idx]
        loadings=loadings[:, moran_idx]
        model = build_model(X, Y, loadings=loadings, factors=factors, model_type=model_type, kwargs=kwargs)
    else:
        # without NMF initialization
        X = np.array(X)
        model = build_model(X, Y, model_type=model_type, kwargs=kwargs)
        
        

    #model = build_model(X, Y, loadings=loadings, factors=factors, model_type=model_type, kwargs=kwargs)
    model_grads(model)
    model.prior.jitter=kwargs['jtr']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
    
    model.to(device)
    X = torch.tensor(X).type(torch.float)
    Y = torch.tensor(Y).type(torch.float)
    X_train = X.to(device)
    #Y_train = Y.to(device)
    
    start_time = time.time()
    losses, means, scales, idxs = train_new_KL_batched(model, optimizer, X_train, Y, device,
                                                steps=steps, E=3, batch_size=kwargs['batch_size'], kwargs=kwargs)
    end_time = time.time()
    
    final_time = end_time - start_time
    
    torch.save(model.state_dict(), f'{save_path}/{file_path}_state_dict.pth')
    torch.save({'losses': losses,
                'means': means,
                'scales': scales,
                'time': final_time},
               f'{save_path}/{file_path}_model.pt')
    
    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.suptitle(f"{model_type} Loss")
    fig.savefig(f'{save_path}/{file_path}_loss.png')
    #fig.close()
    
    size=2
    fig, axes = plt.subplots(3, 5, figsize=(size*5, size*3), tight_layout=True)
    
    model.cpu()
    qF, _, _ = model.prior(X)
    mean = torch.exp(qF.mean).detach().numpy()
    
    plot_factors(mean, X, moran_idx=moran_idx, size=2, s=1, alpha=0.9, ax=axes)
    fig.suptitle(f'Factors')
    fig.savefig(f'{save_path}/{file_path}_plot.png')
    #fig.close()


def run_validation_experiment(X, Y, save_path, steps=1000, batched=False, model_type=None, NMF=False, nmf_path=None, **kwargs):
    kwargs = kwargs['kwargs']
    file_path = model_type

    if model_type == 'VNNGP':
        file_path += f"_K={kwargs['K']}"

        if kwargs['lkzz_build']:
            file_path += f"_lkzz={kwargs['lkzz_build']}"
        
    file_path += f"_M={kwargs['M']}_L={kwargs['L']}_lr={kwargs['lr']}_jtr={kwargs['jtr']}_iter={steps}"
    if batched:
        file_path += f"_bs={kwargs['batch_size']}"
    
    factors = None
    loadings = None
    
    if NMF:
        # with NMF initialization
        file_path += f"_NMFinit"
        factors, loadings = preloaded_nmf_factors(nmf_path)
        X_array = np.array(X.cpu())
        Y_array = np.array(Y.cpu())
        moran_idx, moranI = dims_autocorr(factors, X_array)
        factors=factors[:, moran_idx]
        loadings=loadings[:, moran_idx]
        model = build_model(X_array, Y_array, loadings=loadings, factors=factors, model_type=model_type, kwargs=kwargs)
    else:
        # without NMF initialization
        X = np.array(X)
        model = build_model(X, Y, model_type=model_type, kwargs=kwargs)
        
    #model = build_model(X, Y, loadings=loadings, factors=factors, model_type=model_type, kwargs=kwargs)
    model_grads(model)
    model.prior.jitter=kwargs['jtr']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])

    X = torch.tensor(X).type(torch.float)
    Y = torch.tensor(Y).type(torch.float)
    X_train = X.to(device)
    #Y_train = Y.to(device)
    
    model.to(device)
    X = torch.tensor(X, dtype=torch.float).to(device)
    Y = torch.tensor(Y, dtype=torch.float).to(device)
    start_time = time.time()
    losses, means, scales, idxs = train_batched(model, optimizer, X_train, Y, device,
                                                steps=steps, E=3, batch_size=kwargs['batch_size'], kwargs=kwargs)
    end_time = time.time()
    
    final_time = end_time - start_time
    
    torch.save(model.state_dict(), f'{save_path}/{file_path}_state_dict.pth')
    torch.save({'losses': losses,
                'means': means,
                'scales': scales,
                'time': final_time},
               f'{save_path}/{file_path}_model.pt')
    
    fig, ax = plt.subplots()
    ax.plot(losses)
    fig.suptitle(f"{model_type} Loss")
    fig.savefig(f'{save_path}/{file_path}_loss.png')
    #fig.close()
    
    size=2
    rows = int(kwargs['L']/5)
    fig, axes = plt.subplots(rows, 5, figsize=(size*5, size*rows), tight_layout=True)
    
    model.cpu()
    qF, _, _ = model.prior(X.cpu(), kwargs=kwargs)
    mean = torch.exp(qF.mean).detach().numpy()
    plot_factors(np.exp(mean), X.cpu().detach().numpy(), moran_idx=moran_idx, size=2, s=0.2, alpha=0.9, ax=axes)
    fig.suptitle(f'Factors | sigma: {kwargs["sigma"]}, lengthscale: {kwargs["lengthscale"]}')
    fig.savefig(f'{save_path}/{file_path}_plot.png')
    #fig.close()



def run_pnmf(X, Y, save_path, steps=1000, **kwargs):
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    kwargs = kwargs['kwargs']
    file_path = 'PNMF'
    file_path += f"_L={kwargs['L']}_lr={kwargs['lr']}_iter={steps}"
    prior = GaussianPrior(Y, L=kwargs['L'])
    model = PNMF(prior, Y, L=kwargs['L'])
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])

    X_train = X.to(device)
    Y_train = Y.to(device)

    model.to(device)
    start = time.time()
    losses, means, scales = train(model, optimizer, Y_train, device, steps=steps, E=1, kwargs=kwargs)
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
    fig, axes = plt.subplots(rows, 5, figsize=(size*5, size*rows), tight_layout=True)
    
                
    model.cpu()
    qF, _, _ = model.prior(X.cpu(), kwargs=kwargs)
    mean = torch.exp(qF.mean).detach().numpy()
    plot_factors(np.exp(mean), X.cpu().detach().numpy(), moran_idx=moran_idx, size=2, s=0.2, alpha=0.9, ax=axes)
    fig.suptitle(f'Factors | sigma: {kwargs["sigma"]}, lengthscale: {kwargs["lengthscale"]}')
    fig.savefig(f'{save_path}/{file_path}_plot.png')
    #fig.close()



def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    max_val = np.percentile(factors, 95)
    min_val = np.percentile(factors, 5)

    
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

    if ax is None:
        fig, ax = plt.subplots(2, 5, figsize=(size*5, size*2), tight_layout=True)
        
    for i in range(L):
        plt.subplot(L//5, 5, i+1)
        
        curr_ax = ax[i//5, i%5]
        
        curr_ax.scatter(X[:, 0], X[:,1], c=factors[i], vmin=min_val, vmax=max_val, alpha=alpha, cmap='turbo', s=s)

        curr_ax.invert_yaxis()
        if names is not None:
            curr_ax.set_title(names[i], x=0.03, y=.88, fontsize="small", c="white",
                     ha="left", va="top")
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_facecolor('xkcd:gray')


def plot_factors_five(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    max_val = np.percentile(factors, 95)
    min_val = np.percentile(factors, 5)

    
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

    if ax is None:
        fig, ax = plt.subplots(1, 5, figsize=(size*5, size*2), tight_layout=True)
        
    for i in range(L):
        plt.subplot(1, 5, i+1)
        
        curr_ax = ax[i]
        
        curr_ax.scatter(X[:, 0], X[:,1], c=factors[i], vmin=min_val, vmax=max_val, alpha=alpha, cmap='turbo', s=s)

        curr_ax.invert_yaxis()
        if names is not None:
            curr_ax.set_title(names[i], x=0.03, y=.88, fontsize="small", c="white",
                     ha="left", va="top")
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_facecolor('xkcd:gray')


    
    