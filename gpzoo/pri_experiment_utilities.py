import torch
import matplotlib.pyplot as plt
from torch import optim, distributions, nn
import torch.nn.utils as nn_utils
from tqdm.autonotebook import tqdm
from gpzoo.gp import SVGP, VNNGP, GaussianPrior
from gpzoo.kernels import NSF_RBF, RBF
from gpzoo.likelihoods import NSF2, PNMF, GaussianLikelihood, Hybrid_NSF2
from gpzoo.utilities import rescale_spatial_coords, dims_autocorr, regularized_nmf, anndata_to_train_val, add_jitter, scanpy_sizefactors 
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
from scipy.optimize import curve_fit
from os import path
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import acf


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

def dims_autocorr_timeseries(factors, sort=True, lags=10):
    """
    factors: (num observations) x (num latent dimensions) array
    sort: if True (default), returns the index and autocorrelation statistics in decreasing
    order of autocorrelation. If False, returns the index and autocorrelation statistics
    according to the ordering of factors.

    returns: an integer array of length (num latent dims), "idx"
    and a numpy array containing the ACF values for each dimension

    indexing factors[:,idx] will sort the factors in decreasing order of temporal
    autocorrelation.
    """
    num_factors = factors.shape[1]
    acf_values = np.zeros(num_factors)

    # Compute ACF for each latent dimension (factor)
    for i in range(num_factors):
        factor_series = factors[:, i]
        
        # Compute the autocorrelation function (ACF) for the current factor
        acf_result = acf(factor_series, nlags=lags)
        
        # Store the sum of autocorrelation values (can change this to any ACF-based metric)
        acf_values[i] = np.sum(np.abs(acf_result))

    # Get the sorted indices based on ACF values
    if sort:
        idx = np.argsort(-acf_values)  # Sort in descending order
    else:
        idx = np.arange(num_factors)  # Keep original order

    return idx, acf_values[idx]

def cubic_polynomial(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def generator():
  while True:
    yield

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

    # Compute the KL divergence terms 
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
    

def load_nmf(spath):
    factors_path = path.join(spath, 'nmf/factors.npy')
    loadings_path = path.join(spath, 'nmf/loadings.npy')
    
    factors = np.load(factors_path)
    loadings = np.load(loadings_path)

    return factors, loadings

def inducing_points_cluster_centers(X, M, random_state=256):
    return KMeans(n_clusters=M, random_state=random_state, n_init="auto").fit(X)

def remake_original(model, X, **kwargs):
    '''
    parameters:
    - models: trained model instance
    '''
    kwargs = kwargs['kwargs']
    with torch.no_grad():
        model.cpu()
        try: 
            qF, _, _ = model.prior(X, kwargs=kwargs)
            means = torch.exp(qF.mean).detach().numpy() # means = factors
            W = (model.W).cpu()
        except: # hybrid models
            qF, _, _ = model.sf.prior(X, kwargs=kwargs)
            means = torch.exp(qF.mean).detach().numpy() # means = factors
            W = (model.sf.W).cpu()
        W_transformed = nn.functional.softplus(W.T)
        W_transformed = W_transformed.detach().numpy()
        y_nmf = ((means.T)).dot(W_transformed)
    return y_nmf


def evaluate_model(model, X, Y, device, evaluation_metric=root_mean_squared_error, **kwargs):
    model.eval()  # Set the model to evaluation mode
    kwargs = kwargs['kwargs']
    y_nmf = remake_original(model, X, kwargs=kwargs)
    
    # Check for invalid values (NaNs or Infs) in y_nmf
    y_nmf = np.nan_to_num(y_nmf, nan=0.0, posinf=0.0, neginf=0.0)
    if Y.shape != y_nmf.shape:
        y_nmf = y_nmf.T

    try:
        Y = check_array(Y, ensure_2d=False, force_all_finite='allow-nan')
        y_nmf = check_array(y_nmf, ensure_2d=False, force_all_finite='allow-nan')
    except ValueError:
        # If invalid values are found, apply standard scaling
        scaler = StandardScaler()
        Y = scaler.fit_transform(Y.reshape(-1, 1)).flatten()
        y_nmf = scaler.fit_transform(y_nmf.reshape(-1, 1)).flatten()
    
    # Calculate the reconstruction error using the evaluation metric
    try:
        reconstruction_error = evaluation_metric(Y, y_nmf)
    except:
        reconstruction_error = evaluation_metric(Y.flatten(), y_nmf.flatten())

    return reconstruction_error

def evaluate_model_hybrid(model, X, Y, device, **kwargs):
    model.eval()  # Set the model to evaluation mode
    kwargs = kwargs['kwargs']
    with torch.no_grad():
        model.cpu()
        qF, _, _ = model.sf.prior(X, kwargs=kwargs)
        means = torch.exp(qF.mean).detach().numpy() # means = factors
        W = (model.sf.W).cpu()
        W_transformed = nn.functional.softplus(W.T)
        W_transformed = W_transformed.detach().numpy()
        y_nmf = ((means.T)).dot(W_transformed)
        reconstruction_error = root_mean_squared_error(Y, y_nmf.T)
        
    return reconstruction_error

def build_model(X, Y, loadings=None, factors=None, **kwargs):
    """
    Builds a Gaussian process (GP) model based on the provided data and specified model type.

    Parameters:
    - X (array-like): Input data (features).
    - Y (array-like): Output data (targets).
    - loadings (array-like, optional): Loadings matrix for the model, used to initialize the model's weights.
    - factors (array-like, optional): Latent factors for the model.
    - model_type (str, optional): Type of GP model to build ('VNNGP' or 'SVGP').
    - kwargs (dict): Additional parameters for the model such as number of inducing points (M), 
      random state (rs), lengthscale (L), sigma, jitter, etc.

    Returns:
    - model: The constructed model ready for training.
    """
    kwargs=kwargs['kwargs']
    # Compute size factors for the output data
    V = scanpy_sizefactors(Y.T)
    
    # Compute inducing points using k-means clustering
    kmeans = inducing_points_cluster_centers(X, kwargs['M'], random_state=kwargs['rs'])
    Z = nn.Parameter(torch.tensor(kmeans.cluster_centers_).type(torch.float))
    
    # Define the kernel
    kernel = NSF_RBF(L=kwargs['L'], sigma=kwargs['sigma'], lengthscale=kwargs['lengthscale'])
    
    # Initialize the Gaussian Process model based on the specified model type
    gp = None
    if kwargs['model'] == 'VNNGP':
        gp = VNNGP(kernel, M=kwargs['M'], jitter=kwargs['jtr'], K=kwargs['K'])
    elif kwargs['model'] == 'SVGP':
        gp = SVGP(kernel, M=kwargs['M'], jitter=kwargs['jtr'])
    else:
        raise ValueError("Invalid model type") 

    # Convert input and output data to torch tensors
    X = torch.tensor(X).type(torch.float)
    Y = torch.tensor(Y).type(torch.float)
    
    # Compute kernel matrices
    Kzx = kernel.forward(Z.to('cpu'), X.to('cpu'))
    Kxz = torch.transpose(Kzx, -2, -1)
    Kzz = kernel.forward(Z, Z)

    # Transform factors for GP model
    all_factors = torch.tensor(factors.T)[:, :, None].type(torch.float)
    
    # Compute Cholesky decomposition and solve for the mean of the GP
    L1 = torch.linalg.cholesky(add_jitter(Kzx @ Kxz, kwargs['L1_jitter']))
    solved1 = torch.cholesky_solve(Kzx @ all_factors, L1)
    mu = (Kzz @ solved1).to(device)
    gp.mu = nn.Parameter(torch.squeeze(mu).clone().detach()).type(torch.float)
    
    # Initialize lower triangular matrices
    Lu = 1e-2 * torch.eye(kwargs['M']).expand(kwargs['L'], kwargs['M'], kwargs['M'])
    gp.Lu = nn.Parameter(Lu.clone().detach())
    
    # Set inducing points
    gp.Z = nn.Parameter(Z, requires_grad=False)

    # Construct the full model
    model = NSF2(gp=gp, y=Y, L=kwargs['L'])

    # Initialize model weights if loadings are provided
    if loadings is not None:
        model.W = nn.Parameter(torch.tensor(init_softplus(loadings)[:, :kwargs['L']]).type(torch.float))
    
    model.V = nn.Parameter(torch.squeeze(torch.tensor(init_softplus(V)).type(torch.float)))
    model.to(device)
    
    return model

def build_model_hybrid(X, Y, loadings=None, factors=None, model_type=None, **kwargs):
    """
    """
    kwargs=kwargs['kwargs']
    V = scanpy_sizefactors(Y.T)
    L = kwargs['L']

    kmeans = inducing_points_cluster_centers(X, kwargs['M'], random_state=kwargs['rs'])
    Z = nn.Parameter(torch.tensor(kmeans.cluster_centers_).type(torch.float))
    kernel = NSF_RBF(L=kwargs['L'], sigma=1.0, lengthscale=1.0)

    gaussian_prior = GaussianPrior(Y, L=L-1)
    gaussian_prior.mean = nn.Parameter(torch.zeros(L-1, len(X)).type(torch.float))
    
    gp = None
    if kwargs['model'] == 'VNNGP':
        gp = VNNGP(kernel, M=kwargs['M'], jitter=kwargs['jtr'], K=kwargs['K'])
    elif kwargs['model'] == 'SVGP':
        gp = SVGP(kernel, M=kwargs['M'], jitter=kwargs['jtr'])
    else:
        raise ValueError("Invalid model type") 

    X = torch.tensor(X).type(torch.float)
    Y = torch.tensor(Y).type(torch.float)
    Kzx = kernel.forward(Z.to('cpu'), X.to('cpu'))
    Kxz = torch.transpose(Kzx, -2, -1)
    Kzz = kernel.forward(Z, Z)

    all_factors = torch.tensor((factors[:, :kwargs['L']]).T)[:, :, None].type(torch.float)
    L1 = torch.linalg.cholesky(add_jitter(Kzx @ Kxz, 1e-1)) # changed from 1e-4 to 1e-3
    
    solved1 = torch.cholesky_solve(Kzx @ all_factors, L1)
    mu = (Kzz @ solved1).to(device)
    gp.mu = nn.Parameter(torch.squeeze(mu).clone().detach()).type(torch.float)
    Lu = 1e-2 * torch.eye(kwargs['M']).expand(kwargs['L'], kwargs['M'], kwargs['M'])
    gp.Lu = nn.Parameter(Lu.clone().detach())
    
    gp.Z = nn.Parameter(Z, requires_grad=False)

    model = Hybrid_NSF2(gp=gp, prior=gaussian_prior, y=Y, L=L, T=L-1)

    if loadings is not None:
        model.sf.W = nn.Parameter(torch.tensor(init_softplus(loadings)[:, :kwargs['L']]).type(torch.float))
        #model.cf.W = nn.Parameter(torch.ones(len(Y), L-1))
        model.cf.W = nn.Parameter(torch.tensor(init_softplus(loadings)[:, kwargs['L']:]).type(torch.float)) # initialize nonspatial
        print("MODEL SHAPES")
        print(model.sf.W.shape)
        print(model.cf.W.shape)
        print(nn.Parameter(torch.ones(len(Y), L-1)).shape)
        
    model.cf.prior.scale_pf = 1e-1
    model.V = nn.Parameter(torch.squeeze(torch.tensor(init_softplus(V)).type(torch.float)))
    model.to(device)
    return model


def buil_pnmf(Y, L):
    prior = GaussianPrior(Y, L=L)
    model = PNMF(prior, Y, L=L)
    model.to(device)

def model_grads(model):
    model.prior.kernel.sigma.requires_grad = False
    model.prior.kernel.lengthscale.requires_grad = True
    model.prior.Z.requires_grad=False
    model.prior.mu.requires_grad=True
    model.prior.Lu.requires_grad=True
    model.W.requires_grad=False
    model.V.requires_grad=False

def model_grads_hybrid0(model):
    # 0. train both variances (3 hrs - 9 hrs)
    model.cf.prior.scale.requires_grad = True
    model.sf.prior.Lu.requires_grad=True
    
    model.sf.prior.kernel.lengthscale.requires_grad = False
    model.sf.prior.kernel.sigma.requires_grad = False
    model.sf.prior.mu.requires_grad=False
    model.sf.W.requires_grad=False
    model.cf.W.requires_grad=False
    model.cf.prior.mean.requires_grad=False

def model_grads_hybrid1(model):
    # 1. train mean and variance (spatial) until convergence  (6 hrs)
    model.sf.prior.Lu.requires_grad=True
    model.sf.prior.mu.requires_grad=True
    
    model.cf.prior.mean.requires_grad=False
    model.cf.prior.scale.requires_grad = False
    model.sf.prior.kernel.lengthscale.requires_grad = False
    model.sf.prior.kernel.sigma.requires_grad = False
    model.sf.W.requires_grad=False
    model.cf.W.requires_grad=False

def model_grads_hybrid2(model):
    # 2. train mean and variance until convergence + non-spatial variance (3 hrs)
    model.sf.prior.Lu.requires_grad=True
    model.sf.prior.mu.requires_grad=True
    model.cf.prior.scale.requires_grad = True
    
    model.cf.prior.mean.requires_grad=False
    model.sf.prior.kernel.lengthscale.requires_grad = False
    model.sf.prior.kernel.sigma.requires_grad = False
    model.sf.W.requires_grad=False
    model.cf.W.requires_grad=False

def model_grads_hybrid3(model):
    # 3. train mean, variance and loadings spatial + non-spatial variance (6hrs)
    model.sf.prior.Lu.requires_grad=True
    model.sf.prior.mu.requires_grad=True
    model.cf.prior.scale.requires_grad = True
    model.sf.W.requires_grad=True
    
    model.cf.prior.mean.requires_grad=False
    model.sf.prior.kernel.lengthscale.requires_grad = False
    model.sf.prior.kernel.sigma.requires_grad = False
    model.cf.W.requires_grad=False

def model_grads_hybrid4(model):
    # 4. train non-spatial mean, variance and loadings (6hrs)
    model.cf.prior.mean.requires_grad=True
    model.cf.prior.scale.requires_grad = True
    model.cf.prior.scale.requires_grad = True
    model.cf.W.requires_grad=True
    
    model.sf.prior.Lu.requires_grad=False
    model.sf.prior.mu.requires_grad=False
    model.sf.W.requires_grad=False
    model.sf.prior.kernel.lengthscale.requires_grad = False
    model.sf.prior.kernel.sigma.requires_grad = False

def model_grads_hybrid5(model):
    # 5. train everything (6hrs)
    model.cf.prior.mean.requires_grad=True
    model.cf.prior.scale.requires_grad = True
    model.cf.prior.scale.requires_grad = True
    model.cf.W.requires_grad=True
    model.sf.prior.Lu.requires_grad=True
    model.sf.prior.mu.requires_grad=True
    model.sf.W.requires_grad=True
    model.sf.prior.kernel.lengthscale.requires_grad = False
    model.sf.prior.kernel.sigma.requires_grad = False
    
def train(model, optimizer, X, y, device, steps=200, E=20, **kwargs):
    losses = []
    means = []
    scales = []
    
    for it in tqdm(range(steps)):
        
        optimizer.zero_grad()
        pY, qF, qU, pU = None, None, None, None
        try:
            pY, qF, qU, pU = model.forward(X=X, E=E, **kwargs)
        except:
            pY, qF, pF = model.forward(E=E, **kwargs)

        logpY = y*torch.log(pY.rate) - pY.rate

        ELBO = (logpY).mean(axis=0).sum()
        try:
            ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
        except:
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


def train_convergence(model, optimizer, X, y, device, steps=200, E=20, verbose=False, batch_size=1000, **kwargs):
    kwargs = kwargs['kwargs']
    losses = []
    means = []
    scales = []
    it = 0
    
    for _ in tqdm(generator()):
        it +=1 
        #idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)
        optimizer.zero_grad()
        pY, qF, qU, pU = model.forward(X=X, verbose=verbose, kwargs=kwargs)
        #logpY = y[:, idx]*torch.log(pY.rate) - (pY.rate)
        logpY = y*torch.log(pY.rate) - pY.rate


        ELBO = ((logpY).mean(axis=0).sum()).to(device)
        ELBO -= torch.sum(torch.vmap(vnngp_kl)(qU.mean, qU.scale_tril, model.prior.Lu, model.prior.mu))

        loss = -ELBO
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (it%10)==0:
            means.append(torch.exp(qF.mean.detach().cpu()).numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        # Smoothing strategy for convergence detection
        if it >= 110 and (it % 10) == 0:
            recent_losses = losses[-100:]
            if len(recent_losses) == 100:
                x_data = np.arange(100)
                y_data = recent_losses
                try:
                    popt, _ = curve_fit(cubic_polynomial, x_data, y_data)
                    smoothed_current = cubic_polynomial(99, *popt)
                    smoothed_previous = cubic_polynomial(98, *popt)
                    relative_change = abs(smoothed_current - smoothed_previous) / max(1e-10, abs(smoothed_previous))

                    if relative_change < 5e-5:
                        print(f"Convergence detected at iteration {it}")
                        break
                except RuntimeError:
                    # If curve fitting fails, continue without early stopping
                    pass

    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()

    return losses, means, scales

def train_hybrid(model, optimizer, X, y, device, steps=200, E=20, verbose=False, batch_size=1000, **kwargs):
    kwargs = kwargs['kwargs']
    losses = []
    means = []
    scales = []

    for it in tqdm(range(steps)):
        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)
        optimizer.zero_grad()
        #pY, qF, qU, pU, qF2, pF2 = model.forward_batched(X=X, idx=idx, E=E, verbose=verbose, kwargs=kwargs)
        pY, qF, qU, pU, qF2, pF2 = model.forward(X=X, E=E, kwargs=kwargs)
        
        #logpY = y[:, idx]*torch.log(pY.rate) - (pY.rate)
        logpY = y*torch.log(pY.rate) - pY.rate
        ELBO = ((logpY).mean(axis=0).sum()).to(device)
        KL1 = distributions.kl_divergence(qU, pU)
        KL2 = distributions.kl_divergence(qF2, pF2)


        ELBO -= torch.sum(KL1)
        ELBO -= torch.sum(KL2)

        loss = -ELBO
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        #logpys.append(logpY)
        if (it%10)==0:
            means.append(torch.exp(qF.mean.detach().cpu()).numpy())
            scales.append(qF.scale.detach().cpu().numpy())

    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()

    return losses, means, scales

def train_batched(model, optimizer, X, y, device, steps=200, E=20, verbose=False, **kwargs):
    losses = []
    means = []
    scales = []
    idxs = []
    kwargs = kwargs['kwargs']

    for it in tqdm(range(steps)):
        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=kwargs['batch_size'], replacement=False)
        optimizer.zero_grad()
        pY, qF, qU, pU = model.forward_batched(X=X, idx=idx, E=E, verbose=verbose, kwargs=kwargs)
        #print(f"y shape: {y.shape}")
        #print(f"idx shape: {idx.shape}")
        #print(f"y[:, idx] shape: {y[:, idx].shape}")
        #print(f"pY.rate shape: {pY.rate.shape}")
        
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

def train_batched_convergence(model, optimizer, X, y, device, steps=200, E=20, verbose=False, batch_size=1000, **kwargs):
    kwargs = kwargs['kwargs']
    losses = []
    means = []
    scales = []
    idxs = []
    it = 0
    
    for _ in tqdm(generator()):
        it +=1 
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

        # Smoothing strategy for convergence detection
        if it >= 110 and (it % 10) == 0:
            recent_losses = losses[-100:]
            if len(recent_losses) == 100:
                x_data = np.arange(100)
                y_data = recent_losses
                try:
                    popt, _ = curve_fit(cubic_polynomial, x_data, y_data)
                    smoothed_current = cubic_polynomial(99, *popt)
                    smoothed_previous = cubic_polynomial(98, *popt)
                    relative_change = abs(smoothed_current - smoothed_previous) / max(1e-10, abs(smoothed_previous))

                    if relative_change < 5e-5:
                        print(f"Convergence detected at iteration {it}")
                        break
                except RuntimeError:
                    # If curve fitting fails, continue without early stopping
                    pass

    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()

    return losses, means, scales, idxs

def train_hybrid_batched(model, optimizer, X, y, device, steps=200, E=20, verbose=False, batch_size=1000, **kwargs):
    kwargs = kwargs['kwargs']
    losses = []
    means = []
    scales = []
    idxs = []

    for it in tqdm(range(steps)):
        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)
        optimizer.zero_grad()
        pY, qF, qU, pU, qF2, pF2 = model.forward_batched(X=X, idx=idx, E=E, kwargs=kwargs)
        
        logpY = y[:, idx]*torch.log(pY.rate) - (pY.rate)
        
        ELBO = ((logpY).mean(axis=0).sum()).to(device)
        #KL1 = distributions.kl_divergence(qU, pU)
        ELBO -= torch.sum(torch.vmap(vnngp_kl)(qU.mean, qU.scale_tril, model.sf.prior.Lu, model.sf.prior.mu))
        KL2 = distributions.kl_divergence(qF2, pF2)

        #ELBO -= torch.sum(KL1)
        ELBO -= torch.sum(KL2)
        
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
    losses, means, scales = train(model, optimizer, X_train, Y_train, device, steps=steps, E=1, kwargs=kwargs)
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
    
    #fig2, axes = plt.subplots(rows, 5, figsize=(size*5, size*rows), tight_layout=True)
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



def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    max_val = np.percentile(factors, 95)
    min_val = np.percentile(factors, 5)

    
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

    rows = int(L/5)
    fig, ax = plt.subplots(rows, 5, figsize=(size*5, size*rows), tight_layout=True)
        
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
    return fig, ax


def plot_factors_five(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    max_val = np.percentile(factors, 95)
    min_val = np.percentile(factors, 5)

    
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

   
    fig, ax = plt.subplots(1, 5, figsize=(size*5, size*1), tight_layout=True)
        
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

    return fig, ax

def plot_umap_factors_by_age(loadings, adata, dot_size=10):  # Added dot_size parameter
    L = loadings.shape[1]
    size = 2
    unique_ages = sorted(adata.obs['Age'].unique())  # Sort ages from youngest to oldest
    print(unique_ages)
    num_ages = len(unique_ages)
    vmin = loadings.min().min()
    vmax = loadings.max().max()

    
    # Create subplots
    fig, axes = plt.subplots(L, num_ages + 1, figsize=(size * (num_ages + 1), size * L), tight_layout=True)
    
    for i in range(L):
        # Plot the original UMAP for the factor
        sc.pl.embedding(adata, basis='umap_har', color=f'Factor_{i+1}', ax=axes[i, 0], show=False, cmap='viridis', size=dot_size, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Factor {i+1} - Original')
        
        # Plot UMAPs for each age
        for j, age in enumerate(unique_ages):
            adata_age = adata[adata.obs['Age'] == age]
            sc.pl.embedding(
            adata_age,
            basis='umap_har',
            color=f'Factor_{i+1}',
            ax=axes[i, j + 1],
            show=False,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,  # Set the global color limits
            size=dot_size
            )
            axes[i, j + 1].set_title(f'Age {age}')
    
    plt.tight_layout()
    plt.show()
    return fig

def calculate_eval_metric(X, Y, param_dict, datapath, eval_function=root_mean_squared_error, 
                          train=False, hybrid=False, model_type='VNNGP'):
    """
    General function to evaluate models for both VNNGP and SVGP model types based on param_dict.
    
    Parameters:
    - X: Input features.
    - Y: Target values.
    - param_dict: Dictionary containing hyperparameters (L, M, K, sigma, lengthscale, etc.).
    - datapath: Directory path to load/save model files.
    - eval_function: Evaluation function for model performance.
    - train: If True, evaluate on training and validation sets.
    - hybrid: If True, use a hybrid model.
    - model_type: Model type to specify ('VNNGP' or 'SVGP').
    
    Returns:
    - data_dict: Dictionary containing evaluation metrics (RMSE) for different combinations of L, M, and K.
    """
    
    if train:
        X_train, X_val, y_train, y_val = train_test_split(X, Y.T, test_size=0.05, random_state=param_dict['rs'])
    
    L = param_dict['L']
    M = param_dict['M']
    K = param_dict.get('K', [None])  # K may not be used for SVGP
    data_dict = {str(l): [] for l in L}
    
    for l in L:
        inducing_point_rmse = []
        for m in M:
            neighbor_rmse = []
            for k in (K if model_type == 'VNNGP' else [None]):
                
                # Model parameter dictionary
                dicts = {
                    'L': l, 
                    'M': m, 
                    'K': k,  # VNNGP specific, not used for SVGP
                    'sigma': param_dict['sigma'], 
                    'lengthscale': param_dict['lengthscale'], 
                    'jtr': param_dict['jtr'],
                    'batch_size': 128,
                    'lr': param_dict['lr'],
                    'rs': param_dict['rs'],
                    'lkzz_build': 1,
                    'model': param_dict['model'],
                    'L1_jitter': param_dict['L1_jitter']
                }

                # Load NMF factors and loadings
                nmf_save_path = path.join(datapath, 'nmf')
                factors_file = f"nmf_factors_iter=1000_rs=256_L={l}.npy"
                loadings_file = f"nmf_loadings_iter=1000_rs=256_L={l}.npy"

                if train:
                    factors_file = f"train_{factors_file}"
                    loadings_file = f"train_{loadings_file}"
                
                factors = np.load(path.join(nmf_save_path, factors_file))
                loadings = np.load(path.join(nmf_save_path, loadings_file))
                
                # Build model based on the type and whether hybrid or not
                if not hybrid:
                    if train:
                        model = build_model(X_train, y_train, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                    else:
                        model = build_model(X, Y, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                else:
                    if trian:
                        model = build_model(X_train, y_train, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                    else:
                        model = build_model(X, Y, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                
                # Define model file path based on the type of model
                if model_type == 'VNNGP':
                    modelpath = path.join(datapath, 'nnnsf')
                    filepath = f"VNNGP_K={k}_lkzz=1_M={m}_L={l}_lr={dicts['lr']}_jtr={dicts['jtr']}_ls={dicts['lengthscale']}_sigma={dicts['sigma']}_bs=128_NMFinit_state_dict.pth"
                else:  # model_type == 'SVGP'
                    modelpath = path.join(datapath, 'nsf')
                    filepath = f"SVGP_M={m}_L={l}_lr={dicts['lr']}_jtr={dicts['jtr']}_ls={dicts['lengthscale']}_sigma={dicts['sigma']}_bs={dicts['batch_size']}_NMFinit_state_dict.pth"
                
                dictpath = path.join(modelpath, filepath)
                print(dictpath)

                if not path.exists(dictpath):
                    print("Does not exist.")
                    neighbor_rmse.append(np.nan)
                    continue

                # Load model state
                model.load_state_dict(torch.load(dictpath))

                # Evaluate model
                if train:
                    X_data = (X_train, X_val)
                    Y_data = (y_train.T, y_val.T)
                else:
                    X_data = (X,)
                    Y_data = (Y.T,)

                loss = []
                for X_set, y_set in zip(X_data, Y_data):
                    X_set = torch.tensor(X_set).type(torch.float)
                    #X_set = X_set.clone().detach().type(torch.float)
                    loss.append(evaluate_model(model, X_set.cpu(), y_set, device, evaluation_metric=eval_function, kwargs=dicts))

                if train:
                    loss = (loss[0], loss[1])  # (train_loss, val_loss)
                else:
                    loss = loss[0]  # test loss only

                print(f"With {l} factors and {m} neighbors/IPs: ", loss)
                neighbor_rmse.append(loss)

            inducing_point_rmse.append(neighbor_rmse)
        data_dict[str(l)] = inducing_point_rmse
        print(f"DONE with factor {l}")
    
    return data_dict

    
    