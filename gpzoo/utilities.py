import torch
from tqdm.autonotebook import tqdm
from torch import distributions
from functools import partial
from torch.nn.utils import clip_grad_norm_
from sklearn.decomposition import NMF
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from math import ceil
import matplotlib.pyplot as plt
import gc



def estimate_lcgp_rank(lengthscale, domain_range, dim=2, p=0.9, j_max=1000):
    """Estimate the effective spectral rank of a Matérn 3/2 kernel.

    Returns the minimum number of Fourier modes needed to capture fraction p
    of the kernel's variance, based on the Matérn 3/2 power spectral density
    evaluated on a Fourier grid over the domain.

    Args:
        lengthscale: Kernel lengthscale l.
        domain_range: Tuple (min, max) defining the domain; L = max - min.
        dim: Spatial dimension (1 or 2). Default 2.
        p: Variance threshold (default 0.99).
        j_max: Maximum number of Fourier modes per axis (default 1000).

    Returns:
        Integer R, the estimated effective rank.
    """
    L = domain_range[1] - domain_range[0]
    l = lengthscale

    if dim == 1:
        js = np.arange(0, j_max + 1)
        freqs = np.pi * js / L
        # Matérn 3/2 spectral density: exponent is 2 (nu=3/2 → (nu + d/2) = 2)
        lambdas = (3.0 / l**2 + freqs**2) ** (-2)
        total = lambdas.sum()
        cumsum = np.cumsum(lambdas)
        R = int(np.searchsorted(cumsum, p * total)) + 1
    elif dim == 2:
        js = np.arange(0, j_max + 1)
        freqs = np.pi * js / L
        # 2D grid of eigenvalues: exponent is 3 (nu=3/2 → (nu + d/2) = 3)
        f1, f2 = np.meshgrid(freqs, freqs, indexing='ij')
        lambdas = (3.0 / l**2 + f1**2 + f2**2) ** (-3)
        flat = lambdas.ravel()
        total = flat.sum()
        # Sort descending to find minimum k largest modes covering fraction p
        flat_sorted = np.sort(flat)[::-1]
        cumsum = np.cumsum(flat_sorted)
        R = int(np.searchsorted(cumsum, p * total)) + 1
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")

    return R


def init_Lu_nsf(kernel, X, Z, K, niter=100, use_cholesky=False, jitter=1e-5):
    """Initialize Lu using SVD or Cholesky decomposition.

    Args:
        kernel: Kernel function
        X: Input points
        Z: Inducing points
        K: Rank for low-rank approximation (ignored if use_cholesky=True)
        niter: Number of iterations for low-rank SVD
        use_cholesky: If True, use Cholesky decomposition of K(Z, Z) instead of SVD.
                      This is the preferred method for SVGP models where X=Z.
        jitter: Jitter to add to diagonal for numerical stability (only used with Cholesky)

    Returns:
        Lu initialization matrix
    """
    if use_cholesky:
        # For SVGP: compute Cholesky of K(Z, Z)
        Kzz = kernel.forward(X=Z, Z=Z)
        Kzz = Kzz.contiguous()
        Kzz = add_jitter(Kzz, jitter)
        Lu = torch.linalg.cholesky(Kzz)
        return Lu.detach()

    Kxz = kernel.forward(X=X, Z=Z)
    N = len(X)
    M = len(Z)
    ratio = M/N

    # Use full SVD when K >= M (all components), otherwise use low-rank
    if K >= M:
        U, S, Vh = torch.linalg.svd(Kxz, full_matrices=False)
        U = U[:, :K]
        S = S[:K]
    else:
        U, S, V = torch.svd_lowrank(Kxz, q=K, niter=niter)

    S = S/(ratio**0.5)
    W = U*(S**0.5).detach()
    return W

def init_Lu(kernel, X, groupsX, Z, groupsZ, K, use_cholesky=False, jitter=1e-5):
    """Initialize Lu using SVD or Cholesky decomposition for MGGP models.

    Args:
        kernel: MGGP kernel function
        X: Input points
        groupsX: Group labels for X
        Z: Inducing points
        groupsZ: Group labels for Z
        K: Rank for low-rank approximation (ignored if use_cholesky=True)
        use_cholesky: If True, use Cholesky decomposition of K(Z, Z) instead of SVD.
                      This is the preferred method for SVGP models where X=Z.
        jitter: Jitter to add to diagonal for numerical stability (only used with Cholesky)

    Returns:
        Lu initialization matrix
    """
    if use_cholesky:
        # For SVGP: compute Cholesky of K(Z, Z)
        Kzz = kernel.forward(X=Z, groupsX=groupsZ, Z=Z, groupsZ=groupsZ)
        Kzz = Kzz.contiguous()
        Kzz = add_jitter(Kzz, jitter)
        Lu = torch.linalg.cholesky(Kzz)
        return Lu.detach()

    Kxz = kernel.forward(X=X, groupsX=groupsX, Z=Z, groupsZ=groupsZ)
    N = len(X)
    M = len(Z)
    ratio = M/N

    # Use full SVD when K >= M (all components), otherwise use low-rank
    if K >= M:
        U, S, Vh = torch.linalg.svd(Kxz, full_matrices=False)
        U = U[:, :K]
        S = S[:K]
    else:
        U, S, V = torch.svd_lowrank(Kxz, q=K, niter=100)

    S = S/(ratio**0.5)
    W = U*(S**0.5).detach()
    return W


def build_sparse_from_Kzx_knnidx(Kzx, knn_idx, N):
    """
    Build an (N x N) sparse matrix from Kzx and knn_idx.

    Kzx:     (N, K, 1)  -- kernel values from each x_i to its K neighbors
    knn_idx: (N, K)     -- neighbor indices for each x_i
    N:       int        -- total number of points

    Returns: sparse_coo_tensor of shape (N, N)
    """
    device = Kzx.device
    N, K = knn_idx.shape

    # Flatten i, j coordinates and values
    row_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, K)  # (N, K)
    row_idx = row_idx.reshape(-1)                                        # (N*K,)
    col_idx = knn_idx.reshape(-1)                                        # (N*K,)
    values  = Kzx.squeeze().reshape(-1)                                  # (N*K,)
    
    indices = torch.stack([row_idx, col_idx], dim=0)                    # (2, N*K)
    A_sparse = torch.sparse_coo_tensor(indices, values, size=(N, N))

    return A_sparse.coalesce()


def symmetrize_by_copy(A: torch.Tensor) -> torch.Tensor:
    """
    Make A symmetric by copying missing transposed entries.
    If (i,j) exists in A, its value is kept.
    If (i,j) is missing but (j,i) exists, we add (i,j) with value A[j,i].
    """
    assert A.is_sparse
    A = A.coalesce()
    AT = A.transpose(0, 1).coalesce()

    N = A.size(0)
    ai, av = A.indices(), A.values()     # (2, nnzA), (nnzA,)
    ati, atv = AT.indices(), AT.values() # (2, nnzAT), (nnzAT,)

    # linearize coordinates to find which AT entries are new
    keys_A  = ai[0] * N + ai[1]
    keys_AT = ati[0] * N + ati[1]

    # mask of AT entries not present in A
    if hasattr(torch, "isin"):
        mask_new = ~torch.isin(keys_AT, keys_A)
    else:
        # fallback without torch.isin
        keys_sorted, _ = torch.sort(keys_A)
        pos = torch.searchsorted(keys_sorted, keys_AT)
        in_bounds = pos < keys_sorted.numel()
        match = in_bounds & (keys_sorted[pos.clamp_max(keys_sorted.numel()-1)] == keys_AT)
        mask_new = ~match

    add_idx = ati[:, mask_new]
    add_val = atv[mask_new]

    # union: keep A's entries, plus new mirrored ones
    idx = torch.cat([ai, add_idx], dim=1)
    val = torch.cat([av, add_val], dim=0)

    B = torch.sparse_coo_tensor(idx, val, size=A.size(), device=A.device).coalesce()
    return B

def train_vnngp_batched_with_tracking(optimizer, model, X, groupsX, y, device, steps=200, x_batch_size=1000, y_batch_size=1000, E=20, **kwargs):
    losses = []

    means = []
    scales = []
    idxs = []

    N = len(X)
    J = len(y)

    full_knn_idx = model.prior.calculate_knn(X).clone()

    for it in tqdm(range(steps)):

        # Sampling indices for minibatching
        idx = torch.multinomial(torch.ones(N), num_samples=x_batch_size, replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=y_batch_size, replacement=False)

        X_batch = X[idx]
        y_batch = y[idy][:, idx]
        groupsX_batch = groupsX[idx]

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        groupsX_batch = groupsX_batch.to(device)
        knn_idx = (full_knn_idx[idx]).to(device)
        
        model.prior.knn_idx = knn_idx

        optimizer.zero_grad()
        

        pY, qF, qZ, _ = model.forward_batched(X_batch.squeeze(), idx=idx, groupsX=groupsX_batch, idy=idy)


        logpY = y_batch*torch.log(pY.rate) - pY.rate

        L1 = (logpY).mean(axis=0).sum()
        # L1 = (L1 * N) / batch_size
        L2 = torch.sum(model.prior.kl_divergence(qZ))

        log_prob = L1 - L2
        loss = -log_prob

        loss.backward()
        optimizer.step()
        losses.append(loss.item())


        # Only track these every few steps to avoid memory overload
        if (it % 10) == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())  # or torch.exp(qZ.mean) if Poisson-like
            scales.append(qF.scale.detach().cpu().numpy())


        # Cleanup
        del pY, qF, qZ, logpY, L1, L2, log_prob, loss
        model.prior.knn_idx = None

        del X_batch, y_batch, groupsX_batch, knn_idx

        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        gc.collect()

    return losses, means, scales, idxs


def is_lower_triangular(mat, tol=1e-6):
    upper_part = torch.triu(mat, diagonal=1)
    return torch.all(upper_part.abs() < tol)

def build_group_distances(X, groupsX):
    N = len(torch.unique(groupsX))
    average_position = torch.zeros((N, 2), dtype=torch.float)
    for i in range(N):
        group_mask = groupsX==i
        average_position[i] = torch.mean(X[group_mask])

    distance_mat = torch.cdist(average_position, average_position)

    return distance_mat

    # mu^T@inv(K)mu, K = L^T L
    # mu_k @ inv(K_kk) @ mu_k


def whitened_KL(mz, Lz):

    Lz_diag = torch.diagonal(Lz)
    log_Lz_diag = torch.log(Lz_diag)

    M = len(mz)

    kl_term = -2*torch.sum(log_Lz_diag) + torch.sum(Lz**2) + torch.sum(mz**2) - M

    return 0.5*kl_term

def init_softplus(mat, minval= 1e-5):
    mat2 = mat.copy()
    mask = mat2<20
    mat2[mask] = np.log(np.exp(mat2[mask])-1+minval)

    return mat2


'''
code from original NSF paper https://github.com/willtownes/nsf-paper/tree/main
'''

def smooth_spatial_factors(F,Z,X=None):
  """
  F: real-valued factors (ie on the log scale for NSF)
  Z: inducing point locations
  X: spatial coordinates
  """
  M = Z.shape[0]
  if X is None: #no spatial coordinates, just use the mean
    beta0 = F.mean(axis=0)
    U = np.tile(beta0,[M,1])
    beta = None
  else: #spatial coordinates
    lr = LinearRegression().fit(X,F)
    beta0 = lr.intercept_
    beta = lr.coef_
    nn = max(2, ceil(X.shape[0]/M))
    knn = KNeighborsRegressor(n_neighbors=nn).fit(X,F)
    U = knn.predict(Z)
  return U,beta0,beta


def rescale_spatial_coords(X,box_side=4):
    """
    X is an NxD matrix of spatial coordinates
    Returns a rescaled version of X such that aspect ratio is preserved
    But data are centered at zero and area of equivalent bounding box set to
    box_side^D
    Goal is to rescale X to be similar to a N(0,1) distribution in all axes
    box_side=4 makes the data fit in range (-2,2)
    """
    xmin = X.min(axis=0)
    X -= xmin
    x_gmean = np.exp(np.mean(np.log(X.max(axis=0))))
    X *= box_side/x_gmean
    return X - X.mean(axis=0)

def anndata_to_train_val(ad, layer=None, nfeat=None, train_frac=0.95,
                         sz="constant", dtp="float32", flip_yaxis=False):
    """
    Convert anndata object ad to a training data dictionary
    and a validation data dictionary
    Requirements:
    * rows of ad are pre-shuffled to ensure random split of train/test
    * spatial coordinates in ad.obsm['spatial']
    * features (cols) of ad sorted in decreasing importance (eg with deviance)
    """
    if nfeat is not None: ad = ad[:,:nfeat]
    N = ad.shape[0]
    Ntr = round(train_frac*N)
    X = ad.obsm["spatial"].copy().astype(dtp)
    if flip_yaxis: X[:,1] = -X[:,1]
    X = rescale_spatial_coords(X)
    if layer is None: Y = ad.X
    else: Y = ad.layers[layer]
    
    Y = Y.toarray() #in case Y is a sparse matrix
    Y = Y.astype(dtp)
    Dtr = {"X":X[:Ntr,:], "Y":Y[:Ntr,:]}
    Dval = {"X":X[Ntr:,:], "Y":Y[Ntr:,:]}
    if sz=="constant":
        Dtr["sz"] = np.ones((Ntr,1),dtype=dtp)
        Dval["sz"] = np.ones((N-Ntr,1),dtype=dtp)
    elif sz=="mean":
        Dtr["sz"] = Dtr["Y"].mean(axis=1,keepdims=True)
        Dval["sz"] = Dval["Y"].mean(axis=1,keepdims=True)

    elif sz=="scanpy":
        Dtr["sz"] = scanpy_sizefactors(Dtr["Y"])
        Dval["sz"] = scanpy_sizefactors(Dval["Y"])
    else:
        raise ValueError("unrecognized size factors 'sz'")
    
    Dtr["idx"] = np.arange(Ntr)
    Dval["idx"] = np.arange(Ntr, N)
    if Ntr>=N: Dval = None #avoid returning an empty array
    return Dtr,Dval

def scanpy_sizefactors(Y):
    sz = Y.sum(axis=1,keepdims=True)
    return sz/np.median(sz)

def dims_autocorr(factors,coords,sort=True):
    """
    factors: (num observations) x (num latent dimensions) array
    coords: (num observations) x (num spatial dimensions) array
    sort: if True (default), returns the index and I statistics in decreasing
    order of autocorrelation. If False, returns the index and I statistics
    according to the ordering of factors.

    returns: an integer array of length (num latent dims), "idx"
    and a numpy array containing the Moran's I values for each dimension

    indexing factors[:,idx] will sort the factors in decreasing order of spatial
    autocorrelation.
    """
    from anndata import AnnData
    print('here_andata')
    from squidpy.gr import spatial_neighbors,spatial_autocorr

    ad = AnnData(X=factors,obsm={"spatial":coords})
    spatial_neighbors(ad)
    df = spatial_autocorr(ad,mode="moran",copy=True)
    if not sort: #revert to original sort order
        df.sort_index(inplace=True)
    
    idx = np.array([int(i) for i in df.index])
    return idx,df["I"].to_numpy()

def lnormal_approx_dirichlet(L):
    """
    Approximate a symmetric, flat Dirichlet (alpha=L) of dimension L
    by L independent lognormal distributions.

    The approximation is by matching the marginal means and variances.

    Returns the tuple of (mu,sigma) lognormal parameters
    """

    sigma2 = np.log(2*L)-np.log(L+1) #note this is zero if L=1
    mu = -np.log(L)-sigma2/2.0 #also zero if L=1
    return mu, np.sqrt(sigma2)


'''
code from original NSF paper https://github.com/willtownes/nsf-paper/tree/main
'''

def rescale_spatial_coords(X,box_side=4):
    """
    X is an NxD matrix of spatial coordinates
    Returns a rescaled version of X such that aspect ratio is preserved
    But data are centered at zero and area of equivalent bounding box set to
    box_side^D
    Goal is to rescale X to be similar to a N(0,1) distribution in all axes
    box_side=4 makes the data fit in range (-2,2)
    """
    xmin = X.min(axis=0)
    X -= xmin
    x_gmean = np.exp(np.mean(np.log(X.max(axis=0))))
    X *= box_side/x_gmean
    return X - X.mean(axis=0)

def anndata_to_train_val(ad, layer=None, nfeat=None, train_frac=0.95,
                         sz="constant", dtp="float32", flip_yaxis=False):
    """
    Convert anndata object ad to a training data dictionary
    and a validation data dictionary
    Requirements:
    * rows of ad are pre-shuffled to ensure random split of train/test
    * spatial coordinates in ad.obsm['spatial']
    * features (cols) of ad sorted in decreasing importance (eg with deviance)
    """
    if nfeat is not None: ad = ad[:,:nfeat]
    N = ad.shape[0]
    Ntr = round(train_frac*N)
    X = ad.obsm["spatial"].copy().astype(dtp)
    if flip_yaxis: X[:,1] = -X[:,1]
    X = rescale_spatial_coords(X)
    if layer is None: Y = ad.X
    else: Y = ad.layers[layer]
    
    Y = Y.toarray() #in case Y is a sparse matrix
    Y = Y.astype(dtp)
    Dtr = {"X":X[:Ntr,:], "Y":Y[:Ntr,:]}
    Dval = {"X":X[Ntr:,:], "Y":Y[Ntr:,:]}
    if sz=="constant":
        Dtr["sz"] = np.ones((Ntr,1),dtype=dtp)
        Dval["sz"] = np.ones((N-Ntr,1),dtype=dtp)
    elif sz=="mean":
        Dtr["sz"] = Dtr["Y"].mean(axis=1,keepdims=True)
        Dval["sz"] = Dval["Y"].mean(axis=1,keepdims=True)

    elif sz=="scanpy":
        Dtr["sz"] = scanpy_sizefactors(Dtr["Y"])
        Dval["sz"] = scanpy_sizefactors(Dval["Y"])
    else:
        raise ValueError("unrecognized size factors 'sz'")
    
    Dtr["idx"] = np.arange(Ntr)
    if Ntr>=N: Dval = None #avoid returning an empty array
    return Dtr,Dval

def scanpy_sizefactors(Y):
    sz = Y.sum(axis=1,keepdims=True)
    return sz/np.median(sz)


def lnormal_approx_dirichlet(L):
    """
    Approximate a symmetric, flat Dirichlet (alpha=L) of dimension L
    by L independent lognormal distributions.

    The approximation is by matching the marginal means and variances.

    Returns the tuple of (mu,sigma) lognormal parameters
    """

    sigma2 = np.log(2*L)-np.log(L+1) #note this is zero if L=1
    mu = -np.log(L)-sigma2/2.0 #also zero if L=1
    return mu, np.sqrt(sigma2)



def regularized_nmf(Y, L, sz=1, pseudocount=1e-2, factors=None, 
                    loadings=None, shrinkage=0.2, **kwargs):
    """
    Run nonnegative matrix factorization on (obs x feat) matrix Y
    The factors and loadings matrices are shrunk toward an approximately
    symmetric Dirichlet distribution (equal weight to all features and topics).
    Factors are converted to log scale.

    Parameters
    ----------
    Y : numpy array
    Nonnegative matrix
    L : integer
    Number of nonnegative components
    sz : numeric, optional
    size factors. The default is 1.
    pseudocount : numeric, optional
    Small number to add to nonnegative factors before log transform.
    The default is 1e-2.
    factors : numpy array, optional
    User provided factor matrix. The default is None.
    loadings : numpy array, optional
    User provided loadings matrix. The default is None.
    shrinkage : numeric between zero and one, optional
    How much to shrink toward symmetric Dirichlet. The default is 0.2.
    **kwargs : additional keyword arguments passed to sklearn.decomposition.NMF

    Returns
    -------
    Factors on the log-scale and loadings on the nonnegative scale
    """
    eF = factors
    W = loadings
    if eF is None or W is None:
        nmf = NMF(L,**kwargs)
        eF = nmf.fit_transform(Y)#/sz
        W = nmf.components_.T
    W = shrink_loadings(W, shrinkage=shrinkage)
    wsum = W.sum(axis=0)
    eF = shrink_factors(eF*wsum, shrinkage=shrinkage)
    F = np.log(pseudocount+eF)-np.log(sz)
    prior_mu, prior_sigma = lnormal_approx_dirichlet(max(L,1.1))
    beta0 = prior_mu*np.ones(L)
    wt_to_W = F.mean(axis=0)- beta0
    F-= wt_to_W
    W*= np.exp(wt_to_W-np.log(wsum))
    return F,W

def shrink_factors(F,shrinkage=0.2):
    a = shrinkage
    if 0<a<1:
        fsum = F.sum(axis=1,keepdims=True)
        F = F*(1-a)+a*fsum/float(F.shape[1]) #preserve rowsums
    return F

def shrink_loadings(W,shrinkage=0.2):
    a = shrinkage
    if 0<a<1:
        wsum = W.sum(axis=0)
        W = W*(1-a)+a*wsum/float(W.shape[0]) #preserve colsums
    return W

def regularized_nmf(Y, L, sz=1, pseudocount=1e-2, factors=None, 
                    loadings=None, shrinkage=0.2, **kwargs):
    """
    Run nonnegative matrix factorization on (obs x feat) matrix Y
    The factors and loadings matrices are shrunk toward an approximately
    symmetric Dirichlet distribution (equal weight to all features and topics).
    Factors are converted to log scale.

    Parameters
    ----------
    Y : numpy array
    Nonnegative matrix
    L : integer
    Number of nonnegative components
    sz : numeric, optional
    size factors. The default is 1.
    pseudocount : numeric, optional
    Small number to add to nonnegative factors before log transform.
    The default is 1e-2.
    factors : numpy array, optional
    User provided factor matrix. The default is None.
    loadings : numpy array, optional
    User provided loadings matrix. The default is None.
    shrinkage : numeric between zero and one, optional
    How much to shrink toward symmetric Dirichlet. The default is 0.2.
    **kwargs : additional keyword arguments passed to sklearn.decomposition.NMF

    Returns
    -------
    Factors on the log-scale and loadings on the nonnegative scale
    """
    eF = factors
    W = loadings
    if eF is None or W is None:
        nmf = NMF(L,**kwargs)
        eF = nmf.fit_transform(Y)#/sz
        W = nmf.components_.T
    W = shrink_loadings(W, shrinkage=shrinkage)
    wsum = W.sum(axis=0)
    eF = shrink_factors(eF*wsum, shrinkage=shrinkage)
    F = np.log(pseudocount+eF)-np.log(sz)
    prior_mu, prior_sigma = lnormal_approx_dirichlet(max(L,1.1))
    beta0 = prior_mu*np.ones(L)
    wt_to_W = F.mean(axis=0)- beta0
    F-= wt_to_W
    W*= np.exp(wt_to_W-np.log(wsum))
    return F,W

def shrink_factors(F,shrinkage=0.2):
    a = shrinkage
    if 0<a<1:
        fsum = F.sum(axis=1,keepdims=True)
        F = F*(1-a)+a*fsum/float(F.shape[1]) #preserve rowsums
    return F

def shrink_loadings(W,shrinkage=0.2):
    a = shrinkage
    if 0<a<1:
        wsum = W.sum(axis=0)
        W = W*(1-a)+a*wsum/float(W.shape[0]) #preserve colsums
    return W

def reshape_param(param):
    param_shape = param.shape
    param = param.view(-1, param_shape[-2], param_shape[-1])
    return param

def svgp_forward(Kxx: torch.Tensor, Kzz: torch.Tensor, W: torch.Tensor, inducing_mean: torch.Tensor, inducing_cov: torch.Tensor)-> (torch.Tensor, torch.Tensor):
    '''
        Kxx: Tensor of shape (L x N)
        Kzz: Tensor of shape (L x M x M)
        W: Tensor of shape (L x N x M)
        inducing_mean: Tensor of shape (L x M)
        inducing_cov: Tensor of shape (L x M x M)

        output: Tensor of shape (L x N x 1), Tensor of shape (L x N)
    '''
    mean = W@ (inducing_mean.unsqueeze(-1))
    diff = inducing_cov-Kzz #shape L x M x M

    cov = Kxx + torch.sum((W @ diff)* W, dim=-1) #shape L x N
    
    return mean, cov

def _squared_dist(X, Z):
    
    X2 = (X**2).sum(1, keepdim=True)
    Z2 = (Z**2).sum(1, keepdim=True)
    XZ = X.matmul(Z.t())
    r2 = X2 - 2 * XZ + Z2.t()
    return r2.clamp(min=0)

# def add_jitter(K, jitter=1e-3):
#     mat = K

#     if mat.dim()==2:
#         N, _ =  mat.shape
#         mat.view(-1)[::N+1] += jitter
#         return mat

#     if mat.dim()==3:
#         L, N, _ = mat.shape
#         mat.view(L, -1)[:, ::N+1] += jitter
#         return mat

def add_jitter(K, jitter=1e-3):
    """
    Adds jitter to the diagonal of the last two dimensions of a tensor.
    Uses `view` for speed. Assumes square matrices along the last two dims.
    """
    *batch_dims, N, M = K.shape
    assert N == M, "Last two dimensions must form square matrices"

    # Flatten all leading batch dimensions into one
    B = int(torch.tensor(batch_dims).prod()) if batch_dims else 1
    K.view(B, N * N)[:, ::N + 1] += jitter

    return K


def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    
    max_val = np.percentile(factors, 95)
    min_val = np.percentile(factors, 5)
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

    if ax is None:
        fig, ax = plt.subplots(L//5, 5, figsize=(size*5, size*(L//5)), tight_layout=True)
        
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

def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


def _embed_distance_matrix(distance_matrix):

    # Code adapted from https://github.com/andrewcharlesjones/multi-group-GP
    N = len(distance_matrix)
    D2 = distance_matrix**2
    C = torch.eye(N) - (1/N) *torch.ones(size=(N, N))
    B = -0.5*(C @ D2 @ C)
    L, Q = torch.linalg.eigh(B)
    L[L < 0] = 0
    embedding = Q @ torch.diag(_torch_sqrt(L, 1e-6))
    return embedding

def train(model, optimizer, X, y, device, steps=200, E=20, **kwargs):
    losses = []
    for _ in tqdm(range(steps)):

        optimizer.zero_grad()
        pY, _ , qU, pU = model(X=X, E=E, **kwargs)


        ELBO = (pY.log_prob(y)).mean(axis=0).sum()

        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))


        loss = -ELBO
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    return losses




def train_hybrid_batched(model, optimizer, X, y, device, steps=200, E=20, batch_size=1000, **kwargs):
    losses = []
    for it in tqdm(range(steps)):
        

        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)


        optimizer.zero_grad()
        pY, _ , qU, pU, qF, pF = model.forward_batched(X=X, idx=idx, E=E, **kwargs)

        # logpY = pY.log_prob(y[:, idx])
        logpY = y[:, idx]*torch.log(pY.rate) - pY.rate
        # print(logpY.shape)

        ELBO = (logpY).mean(axis=0).sum()

        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
        ELBO -= torch.sum(distributions.kl_divergence(qF, pF))

        loss = -ELBO
        loss.backward()
        optimizer.step()

        #keep W and W2 positive after updates
        model.W.data = torch.clamp(model.W.data, min=0.0)
        model.W2.data = torch.clamp(model.W2.data, min=0.0)

        losses.append(loss.item())
    
    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    return losses


def train_hybrid(model, optimizer, X, y, device, steps=200, E=20, **kwargs):
    losses = []
    for _ in tqdm(range(steps)):

        optimizer.zero_grad()
        pY, _ , qU, pU, qF, pF = model(X=X, E=E, **kwargs)


        ELBO = (pY.log_prob(y)).mean(axis=0).sum()

        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
        ELBO -= torch.sum(distributions.kl_divergence(qF, pF))

        loss = -ELBO
        loss.backward()
        optimizer.step()
        
        #keep W and W2 positive after updates
        model.W.data = torch.clamp(model.W.data, min=0.0)
        model.W2.data = torch.clamp(model.W2.data, min=0.0)

        
        losses.append(loss.item())
    
    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    return losses


def train_closure_batched(model, optimizer, X, groupsX, y, device, steps=200, E=20, batch_size=1000):
    losses = []

    
    def closure(idx):
        print('here_')
        optimizer.zero_grad()
        pY, _ , qU, pU = model.forward_batched(X, groupsX, idx, E=E)
        logpY = pY.log_prob(y[:, idx])
        ELBO = (logpY).mean(axis=0).sum()
        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
        loss = -ELBO
        loss.backward()
        losses.append(loss.item())
        # clip_grad_norm_(model.parameters(), 1.0)
        return loss
    
    
    for it in tqdm(range(steps)):
        print('step')

        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)

        optimizer.step(partial(closure, idx))


    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()

    return losses



def train_batched(model, optimizer, X, y, device, steps=200, E=20, batch_size=1000, **kwargs):
    losses = []
    for it in tqdm(range(steps)):
        

        idx = torch.multinomial(torch.ones(X.shape[0]), num_samples=batch_size, replacement=False)


        optimizer.zero_grad()
        pY, _ , qU, pU = model.forward_batched(X=X, idx=idx, E=E, **kwargs)

        logpY = pY.log_prob(y[:, idx])
        # print(logpY.shape)

        ELBO = (logpY).mean(axis=0).sum()

        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))


        loss = -ELBO
        loss.backward()
        optimizer.step()
        #keep W and W2 positive after updates
        model.W.data = torch.clamp(model.W.data, min=0.0)
        # model.W2.data = torch.clamp(model.W2.data, min=0.0)

        losses.append(loss.item())
    
    with torch.no_grad():
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    return losses