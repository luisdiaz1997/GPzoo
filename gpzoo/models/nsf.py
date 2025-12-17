"""
Convenience classes for NSF (Non-negative Spatial Factorization) models.

These classes provide a high-level API that handles all initialization while
still being built on the modular components (kernels, GPs, likelihoods).

Usage:
    # Simple usage - batteries included
    model = SVGP_NSF(X, Y, L=12, lengthscale=8.0)

    # With more control
    model = MGGP_VNNGP_NSF(X, Y, groupsX, L=12, lengthscale=8.0, K=50)

    # Or use the modular components directly
    kernel = Matern32(lengthscale=8.0)
    gp = SVGP(kernel, M=1000)
    model = NSF2(gp, Y, L=12)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Literal
import torch
from torch import nn

from ..kernels import batched_Matern32, batched_MGGP_Matern32, batched_RBF, batched_MGGP_RBF
from ..gp import SVGP, VNNGP, MGGP_SVGP, MGGP_VNNGP, LCGP, MGGP_LCGP
from ..likelihoods import NSF2
from ..utilities import init_Lu_nsf, init_Lu
from ..model_utilities import (
    mggp_kmeans_inducing_points,
    _init_mu_from_lu,
    _default_device,
    init_loadings,
)
from ..modules import CholeskyParameter


# =============================================================================
# Abstract Base Class
# =============================================================================

class NSF_Model(nn.Module, ABC):
    """
    Abstract base class for NSF models.
    
    Provides common interface and reduces boilerplate across model variants.
    Subclasses must implement _build_model() to construct the specific GP and likelihood.
    """
    
    # Whether this model uses multi-group kernels (requires groupsX in forward)
    _is_mggp: bool = False
    
    def __init__(self):
        super().__init__()
        self._model: NSF2 = None
    
    # -------------------------------------------------------------------------
    # Properties (common to all models)
    # -------------------------------------------------------------------------
    
    @property
    def prior(self):
        return self._model.prior

    @property
    def W(self):
        return self._model.W.data

    @W.setter
    def W(self, value):
        # Handle both CholeskyParameter and raw parameter cases
        if hasattr(self._model.W, '_raw'):
            self._model.W._raw.data = value
        else:
            self._model.W.data = value

    @property
    def V(self):
        return self._model.V

    @V.setter
    def V(self, value):
        self._model.V = value
    
    # -------------------------------------------------------------------------
    # Forward methods
    # -------------------------------------------------------------------------
    
    def forward(self, X, E=10, verbose=False, groupsX=None, **kwargs):
        if self._is_mggp:
            return self._model.forward(X, E=E, verbose=verbose, groupsX=groupsX, **kwargs)
        return self._model.forward(X, E=E, verbose=verbose, **kwargs)

    def forward_batched(self, X, idx, idy=None, E=10, verbose=False, groupsX=None, **kwargs):
        if self._is_mggp:
            return self._model.forward_batched(X, idx, idy=idy, E=E, verbose=verbose, groupsX=groupsX, **kwargs)
        return self._model.forward_batched(X, idx, idy=idy, E=E, verbose=verbose, **kwargs)

    def forward_batched_train(self, X, idx, idy=None, E=10, verbose=False, groupsX=None, **kwargs):
        if self._is_mggp:
            return self._model.forward_batched_train(X, idx, idy=idy, E=E, verbose=verbose, groupsX=groupsX, **kwargs)
        return self._model.forward_batched_train(X, idx, idy=idy, E=E, verbose=verbose, **kwargs)
    
    # -------------------------------------------------------------------------
    # nn.Module delegation
    # -------------------------------------------------------------------------
    
    def to(self, device):
        self._model.to(device)
        return self

    def parameters(self):
        return self._model.parameters()

    def named_parameters(self):
        return self._model.named_parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        return self._model.load_state_dict(state_dict, strict=strict)

    def project_parameters(self):
        """Apply projection to parameters if using projected mode."""
        self._model.project_parameters()
    
    # -------------------------------------------------------------------------
    # Helpers for subclasses
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _get_kernel_cls(kernel_type: str, is_mggp: bool):
        """Get the appropriate kernel class."""
        if is_mggp:
            return batched_MGGP_Matern32 if kernel_type == "matern32" else batched_MGGP_RBF
        return batched_Matern32 if kernel_type == "matern32" else batched_RBF
    
    @staticmethod
    def _init_size_factors(Y: torch.Tensor, V: Optional[torch.Tensor]) -> nn.Parameter:
        """Initialize size factors V."""
        if V is None:
            V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
        return nn.Parameter(V.clone())
    
    def _init_loadings(
        self,
        Y: torch.Tensor,
        L: int,
        loadings_init: str,
        seed: Optional[int],
        init_mu_from_factors: bool = False,
    ):
        """Initialize W from data, optionally initialize GP mu from factors."""
        if loadings_init in ("svd", "pca", "nmf"):
            if init_mu_from_factors:
                W_init, F_init = init_loadings(Y, L, mode=loadings_init, seed=seed, return_factors=True)
                self._model.W.data = W_init
                return F_init  # Return for caller to use
            else:
                W_init = init_loadings(Y, L, mode=loadings_init, seed=seed, return_factors=False)
                self._model.W.data = W_init
        return None
    
    def _init_lu_for_svgp(
        self,
        gp,
        kernel,
        Z: torch.Tensor,
        L: int,
        M: int,
        lu_rank: Optional[int],
        lu_init_iters: int,
        scale_multiplier: float,
        jitter: float,
        cholesky_mode: str,
        diagonal_only: bool,
        seed: Optional[int],
        groupsZ: Optional[torch.Tensor] = None,
    ):
        """Initialize Lu for SVGP-style models (using CholeskyParameter)."""
        rank = lu_rank or M
        
        if groupsZ is not None:
            # MGGP case
            Lu_base = init_Lu(
                kernel, Z, groupsZ, Z, groupsZ, K=rank,
                use_cholesky=True, jitter=jitter
            )
        else:
            Lu_base = init_Lu_nsf(
                kernel, Z, Z, K=rank, niter=lu_init_iters,
                use_cholesky=True, jitter=jitter
            )
        
        if Lu_base.dim() == 2:
            Lu_base = Lu_base.unsqueeze(0)
        
        Lu_target = (scale_multiplier * Lu_base).expand(L, -1, -1).contiguous()
        
        # Replace the single M-sized CholeskyParameter with L x M x M batched version
        del gp.Lu
        gp.Lu = CholeskyParameter((L, M), mode=cholesky_mode, diagonal_only=diagonal_only)
        gp.Lu.data = Lu_target
        
        gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))
    
    def _init_lu_for_vnngp(
        self,
        gp,
        kernel,
        X: torch.Tensor,
        Z: torch.Tensor,
        L: int,
        K: int,
        lu_rank: Optional[int],
        lu_init_iters: int,
        scale_multiplier: float,
        subset_step: int,
        seed: Optional[int],
        groupsZ: Optional[torch.Tensor] = None,
    ):
        """Initialize Lu for VNNGP-style models (using raw nn.Parameter)."""
        ref = X[::max(subset_step, 1)]
        rank = lu_rank or K
        
        if groupsZ is not None:
            # MGGP case
            ref_groups = groupsZ[::max(subset_step, 1)]
            Lu_base = init_Lu(kernel, Z, groupsZ, ref, ref_groups, K=rank)
        else:
            Lu_base = init_Lu_nsf(kernel, X, ref, K=rank, niter=lu_init_iters)
        
        if Lu_base.dim() == 2:
            Lu_base = Lu_base.unsqueeze(0)
        
        Lu = Lu_base.expand(L, -1, -1).contiguous()
        gp.Lu = nn.Parameter((scale_multiplier * Lu).clone().detach())
        gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))
    
    def _setup_knn(self, X: torch.Tensor):
        """Precompute KNN indices for VNNGP models."""
        knn_idx = self._model.prior.calculate_knn(X)[:, :-1]
        self._model.prior.knn_idx = knn_idx
        knn_idz = self._model.prior.calculate_knn(self._model.prior.Z)[:, 1:]
        self._model.prior.knn_idz = knn_idz


# =============================================================================
# Concrete Implementations
# =============================================================================

class SVGP_NSF(NSF_Model):
    """
    Sparse Variational GP with NSF (Poisson) likelihood.
    """
    
    _is_mggp = False

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        L: int = 12,
        lengthscale: float = 8.0,
        sigma: float = 1.0,
        num_inducing: Optional[int] = None,
        inducing_points: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        kernel_type: Literal["matern32", "rbf"] = "matern32",
        lu_rank: Optional[int] = None,
        lu_init_iters: int = 10,
        scale_multiplier: float = 1e-6,
        cholesky_mode: str = "exp",
        diagonal_only: bool = False,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
        loadings_init: str = "pca",
    ):
        super().__init__()

        kernel = self._get_kernel_cls(kernel_type, is_mggp=False)(sigma=sigma, lengthscale=lengthscale)

        if inducing_points is None:
            num_inducing = num_inducing or min(4000, X.shape[0])
            Z = X[:num_inducing].clone()
        else:
            Z = inducing_points.clone()
        M = Z.shape[0]

        gp = SVGP(kernel, M=M, jitter=jitter, cholesky_mode=cholesky_mode, diagonal_only=diagonal_only)
        gp.Z = nn.Parameter(Z, requires_grad=False)

        self._init_lu_for_svgp(
            gp, kernel, Z, L, M, lu_rank, lu_init_iters,
            scale_multiplier, jitter, cholesky_mode, diagonal_only, seed
        )

        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.V = self._init_size_factors(Y, V)

        # Initialize loadings from data
        self._init_loadings(Y, L, loadings_init, seed, init_mu_from_factors=False)

        self._model.to(_default_device(X, device))


class VNNGP_NSF(NSF_Model):
    """
    Variational Nearest-Neighbor GP with NSF (Poisson) likelihood.
    """

    _is_mggp = False

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        L: int = 12,
        K: int = 50,
        lengthscale: float = 8.0,
        sigma: float = 1.0,
        num_inducing: Optional[int] = None,
        inducing_points: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        kernel_type: Literal["matern32", "rbf"] = "matern32",
        lu_rank: Optional[int] = None,
        lu_init_iters: int = 10,
        scale_multiplier: float = 1e-6,
        subset_step: int = 10,
        precompute_knn: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
        loadings_init: str = "pca",
    ):
        super().__init__()

        kernel = self._get_kernel_cls(kernel_type, is_mggp=False)(sigma=sigma, lengthscale=lengthscale)

        Z = X.clone() if inducing_points is None else inducing_points.clone()
        M = Z.shape[0]

        gp = VNNGP(kernel, M=M, jitter=jitter, K=K)
        gp.Z = nn.Parameter(Z, requires_grad=True)

        self._init_lu_for_vnngp(
            gp, kernel, X, Z, L, K, lu_rank, lu_init_iters,
            scale_multiplier, subset_step, seed
        )

        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.prior.K = K
        self._model.V = self._init_size_factors(Y, V)

        # Initialize loadings from data and get factors for mu
        F_init = self._init_loadings(Y, L, loadings_init, seed, init_mu_from_factors=True)

        if F_init is not None:
            # F_init is (L, N) in log-space, mu is (L, M) where M=N for VNNGP
            self._model.prior.mu.data = F_init.to(self._model.prior.mu.device)

        if precompute_knn:
            self._setup_knn(X)

        self._model.to(_default_device(X, device))


class MGGP_SVGP_NSF(NSF_Model):
    """
    Multi-Group SVGP with NSF (Poisson) likelihood.
    """
    
    _is_mggp = True

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        groupsX: torch.Tensor,
        *,
        L: int = 12,
        lengthscale: float = 8.0,
        sigma: float = 1.0,
        group_diff_param: float = 10.0,
        num_inducing: Optional[int] = None,
        inducing_points: Optional[torch.Tensor] = None,
        inducing_groups: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        kernel_type: Literal["matern32", "rbf"] = "matern32",
        scale_multiplier: float = 1e-6,
        cholesky_mode: str = "exp",
        diagonal_only: bool = False,
        inducing_method: Literal["kmeans", "random"] = "kmeans",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
        loadings_init: str = "pca",
    ):
        super().__init__()

        n_groups = int(groupsX.max().item() + 1)
        kernel = self._get_kernel_cls(kernel_type, is_mggp=True)(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        if inducing_points is None or inducing_groups is None:
            num_inducing = num_inducing or min(4000, X.shape[0])
            Z, groupsZ = mggp_kmeans_inducing_points(
                X, groupsX, num_inducing, seed=seed or 123,
                allocation="proportional" if inducing_method == "kmeans" else "equal"
            )
        else:
            Z, groupsZ = inducing_points.clone(), inducing_groups.clone()
        M = Z.shape[0]

        gp = MGGP_SVGP(kernel, M=M, n_groups=n_groups, jitter=jitter,
                       cholesky_mode=cholesky_mode, diagonal_only=diagonal_only)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        self._init_lu_for_svgp(
            gp, kernel, Z, L, M, None, 10,
            scale_multiplier, jitter, cholesky_mode, diagonal_only, seed,
            groupsZ=groupsZ
        )

        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.V = self._init_size_factors(Y, V)

        # Initialize loadings from data
        self._init_loadings(Y, L, loadings_init, seed, init_mu_from_factors=False)

        self._model.to(_default_device(X, device))


class MGGP_VNNGP_NSF(NSF_Model):
    """
    Multi-Group VNNGP with NSF (Poisson) likelihood.

    Combines multi-group kernel modulation with nearest-neighbor sparsity
    for scalable multi-sample spatial transcriptomics.

    Args:
        X: Spatial coordinates, shape (N, D)
        Y: Gene expression counts, shape (genes, N)
        groupsX: Group assignments for each cell, shape (N,)
        L: Number of latent factors
        K: Number of nearest neighbors for sparse approximation
        lengthscale: RBF/Matern kernel lengthscale
        sigma: Kernel variance (default 1.0)
        group_diff_param: Controls group similarity (higher = more different)
        num_inducing: Number of inducing points (default: N)
        inducing_points: Optional pre-specified inducing points
        inducing_groups: Group assignments for inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        precompute_knn: Whether to precompute KNN indices
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode

    Example:
        >>> model = MGGP_VNNGP_NSF(X, Y, groupsX, L=12, K=50, lengthscale=8.0)
        >>> model.to("cuda")
        >>> model.prior.knn_idx = full_knn_idx[batch_idx]
        >>> pY, qF, qZ, pZ = model.forward_batched_train(
        ...     X_batch, idx=batch_idx, groupsX=groups_batch
        ... )
    """
    
    _is_mggp = True

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        groupsX: torch.Tensor,
        *,
        L: int = 12,
        K: int = 50,
        lengthscale: float = 8.0,
        sigma: float = 1.0,
        group_diff_param: float = 10.0,
        num_inducing: Optional[int] = None,
        inducing_points: Optional[torch.Tensor] = None,
        inducing_groups: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        kernel_type: Literal["matern32", "rbf"] = "matern32",
        scale_multiplier: float = 1e-6,
        subset_step: int = 10,
        precompute_knn: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
        loadings_init: str = "pca",
    ):
        super().__init__()

        n_groups = int(groupsX.max().item() + 1)
        kernel = self._get_kernel_cls(kernel_type, is_mggp=True)(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        if inducing_points is None or inducing_groups is None:
            Z, groupsZ = X.clone(), groupsX.clone()
        else:
            Z, groupsZ = inducing_points.clone(), inducing_groups.clone()
        M = Z.shape[0]

        gp = MGGP_VNNGP(kernel, M=M, n_groups=n_groups, jitter=jitter, K=K)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        self._init_lu_for_vnngp(
            gp, kernel, X, Z, L, K, None, 10,
            scale_multiplier, subset_step, seed,
            groupsZ=groupsZ
        )

        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.prior.K = K
        self._model.V = self._init_size_factors(Y, V)

        # Initialize loadings from data and get factors for mu
        F_init = self._init_loadings(Y, L, loadings_init, seed, init_mu_from_factors=True)

        if F_init is not None:
            # F_init is (L, N) in log-space, mu is (L, M) where M=N for MGGP_VNNGP
            self._model.prior.mu.data = F_init.to(self._model.prior.mu.device)

        if precompute_knn:
            self._setup_knn(X)

        self._model.to(_default_device(X, device))



class LCGP_NSF(NSF_Model):
    """
    Locally Conditioned GP with NSF (Poisson) likelihood.
    
    Uses low-rank plus diagonal variational covariance S = D + VV^T,
    which provides O(MR) parameters instead of O(M²) while still
    capturing correlations between inducing points.
    
    The KL divergence uses the locally conditioned approximation:
        KL(q(U) || p(U)) ≈ Σ_j E_{q(U_{n(j)})}[KL(q(U_j|U_{n(j)}) || p(U_j|U_{n(j)}))]
    
    Args:
        X: Spatial coordinates, shape (N, D)
        Y: Gene expression counts, shape (genes, N)
        L: Number of latent factors
        K: Number of nearest neighbors for local conditioning
        rank: Rank of low-rank component V (default: min(M, K+5))
        lengthscale: RBF/Matern kernel lengthscale
        sigma: Kernel variance (default 1.0)
        num_inducing: Number of inducing points (default: N, i.e., all points)
        inducing_points: Optional pre-specified inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        diag_mode: 'softplus' or 'exp' for diagonal constraint
        precompute_knn: Whether to precompute KNN indices
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode
        loadings_init: Initialization method for W ('pca', 'svd', 'nmf')
    """
    
    _is_mggp = False

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        *,
        L: int = 12,
        K: int = 50,
        rank: Optional[int] = None,
        lengthscale: float = 8.0,
        sigma: float = 1.0,
        num_inducing: Optional[int] = None,
        inducing_points: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        kernel_type: Literal["matern32", "rbf"] = "matern32",
        diag_mode: str = "softplus",
        scale_multiplier: float = 1e-6,
        precompute_knn: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
        loadings_init: str = "pca",
    ):
        super().__init__()

        kernel = self._get_kernel_cls(kernel_type, is_mggp=False)(sigma=sigma, lengthscale=lengthscale)

        # LCGP typically uses all points as inducing points (like VNNGP)
        Z = X.clone() if inducing_points is None else inducing_points.clone()
        M = Z.shape[0]
        
        # Default rank
        if rank is None:
            rank = min(M, K + 5)

        gp = LCGP(kernel, M=M, jitter=jitter, K=K, rank=rank, diag_mode=diag_mode)
        gp.Z = nn.Parameter(Z, requires_grad=False)

        # Initialize LowRankPlusDiagonal for batched factors
        self._init_lu_for_lcgp(
            gp, L, M, rank, scale_multiplier, diag_mode, seed
        )

        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.prior.K = K
        self._model.V = self._init_size_factors(Y, V)

        # Initialize loadings from data and get factors for mu
        F_init = self._init_loadings(Y, L, loadings_init, seed, init_mu_from_factors=True)

        if F_init is not None:
            self._model.prior.mu.data = F_init.to(self._model.prior.mu.device)

        if precompute_knn:
            self._setup_knn(X)

        self._model.to(_default_device(X, device))
    
    def _init_lu_for_lcgp(
        self,
        gp,
        L: int,
        M: int,
        rank: int,
        scale_multiplier: float,
        diag_mode: str,
        seed: Optional[int],
    ):
        """Initialize LowRankPlusDiagonal for LCGP with batched factors."""
        from ..modules import LowRankPlusDiagonal
        
        # Replace the single-factor Lu with batched version
        del gp.Lu
        gp.Lu = LowRankPlusDiagonal(m=M, rank=rank, batch_size=L, diag_mode=diag_mode)
        
        # Initialize diagonal to small positive values
        if diag_mode == 'softplus':
            init_val = torch.log(torch.exp(torch.tensor(scale_multiplier)) - 1)
            gp.Lu.diag._raw.data.fill_(init_val)
        else:  # exp
            gp.Lu.diag._raw.data.fill_(torch.log(torch.tensor(scale_multiplier)))
        
        # Initialize V to small random values
        if seed is not None:
            torch.manual_seed(seed)
        gp.Lu.V._raw.data = torch.randn(L, M, rank) * 0.01
        
        # Initialize mu
        gp.mu = nn.Parameter(torch.randn(L, M) * 0.1)


class MGGP_LCGP_NSF(NSF_Model):
    """
    Multi-Group LCGP with NSF (Poisson) likelihood.
    
    Combines multi-group kernel modulation with LCGP's low-rank plus diagonal 
    covariance parameterization.
    
    Args:
        X: Spatial coordinates, shape (N, D)
        Y: Gene expression counts, shape (genes, N)
        groupsX: Group assignments for each cell, shape (N,)
        L: Number of latent factors
        K: Number of nearest neighbors for local conditioning
        rank: Rank of low-rank component V (default: min(M, K+5))
        lengthscale: RBF/Matern kernel lengthscale
        sigma: Kernel variance (default 1.0)
        group_diff_param: Controls group similarity (higher = more different)
        num_inducing: Number of inducing points (default: N)
        inducing_points: Optional pre-specified inducing points
        inducing_groups: Group assignments for inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        diag_mode: 'softplus' or 'exp' for diagonal constraint
        scale_multiplier: float = 1e-6
        precompute_knn: Whether to precompute KNN indices
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode
        loadings_init: Initialization method for W ('pca', 'svd', 'nmf')
    """
    
    _is_mggp = True

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        groupsX: torch.Tensor,
        *,
        L: int = 12,
        K: int = 50,
        rank: Optional[int] = None,
        lengthscale: float = 8.0,
        sigma: float = 1.0,
        group_diff_param: float = 10.0,
        num_inducing: Optional[int] = None,
        inducing_points: Optional[torch.Tensor] = None,
        inducing_groups: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        jitter: float = 1e-5,
        kernel_type: Literal["matern32", "rbf"] = "matern32",
        diag_mode: str = "softplus",
        scale_multiplier: float = 1e-6,
        precompute_knn: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
        loadings_init: str = "pca",
    ):
        super().__init__()

        n_groups = int(groupsX.max().item() + 1)
        kernel = self._get_kernel_cls(kernel_type, is_mggp=True)(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        if inducing_points is None or inducing_groups is None:
            Z, groupsZ = X.clone(), groupsX.clone()
        else:
            Z, groupsZ = inducing_points.clone(), inducing_groups.clone()
        
        M = Z.shape[0]
        # Default rank
        if rank is None:
            rank = min(M, K + 5)

        gp = MGGP_LCGP(kernel, M=M, n_groups=n_groups, jitter=jitter, K=K, rank=rank, diag_mode=diag_mode)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        # Initialize batched LowRankPlusDiagonal for MGGP-LCGP
        # We need to adapt _init_lu_for_lcgp to handle MGGP case if needed, 
        # but LCGP's Lu structure is independent of groupsZ generally, it's about the covariance.
        # However, LCGP_NSF uses _init_lu_for_lcgp which defines Lu.
        # Let's inspect _init_lu_for_lcgp in LCGP_NSF.
        # It takes 'gp' and replaces gp.Lu.
        # This should work fine for MGGP_LCGP too since it wraps LCGP and exposes Lu.
        
        # We can reuse the logic from LCGP_NSF by calling it or copying it.
        # Since it's a method of LCGP_NSF, we can't call it easily unless we mixin or duplicate.
        # Duplication is safer to avoid MRO issues or making LCGP_NSF a base.
        
        from ..modules import LowRankPlusDiagonal
        
        del gp.Lu
        gp.Lu = LowRankPlusDiagonal(m=M, rank=rank, batch_size=L, diag_mode=diag_mode)
        
        if diag_mode == 'softplus':
            init_val = torch.log(torch.exp(torch.tensor(scale_multiplier)) - 1)
            gp.Lu.diag._raw.data.fill_(init_val)
        else:
            gp.Lu.diag._raw.data.fill_(torch.log(torch.tensor(scale_multiplier)))
        
        if seed is not None:
            torch.manual_seed(seed)
        gp.Lu.V._raw.data = torch.randn(L, M, rank) * 0.01
        
        gp.mu = nn.Parameter(torch.randn(L, M) * 0.1)

        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.prior.K = K
        self._model.V = self._init_size_factors(Y, V)

        # Initialize loadings from data
        F_init = self._init_loadings(Y, L, loadings_init, seed, init_mu_from_factors=True)
        if F_init is not None:
            self._model.prior.mu.data = F_init.to(self._model.prior.mu.device)

        if precompute_knn:
            self._setup_knn(X)

        self._model.to(_default_device(X, device))