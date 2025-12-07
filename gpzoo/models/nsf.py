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
from ..gp import SVGP, VNNGP, MGGP_SVGP, MGGP_VNNGP
from ..likelihoods import NSF2
from ..utilities import init_Lu_nsf, init_Lu
from ..model_utilities import (
    mggp_kmeans_inducing_points,
    _init_mu_from_lu,
    _default_device,
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
        self._model: NSF2 = None  # Set by subclass in _build_model
    
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

    This is the standard SVGP model for spatial transcriptomics data.
    Suitable for datasets up to ~50k cells with ~4k inducing points.

    Args:
        X: Spatial coordinates, shape (N, D)
        Y: Gene expression counts, shape (genes, N)
        L: Number of latent factors
        lengthscale: RBF/Matern kernel lengthscale
        sigma: Kernel variance (default 1.0)
        num_inducing: Number of inducing points (default: min(4000, N))
        inducing_points: Optional pre-specified inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        cholesky_mode: 'exp' or 'softplus' for Lu diagonal constraint
        diagonal_only: If True, use diagonal covariance for Lu
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode

    Example:
        >>> model = SVGP_NSF(X, Y, L=12, lengthscale=8.0)
        >>> model.to("cuda")
        >>> pY, qF, qZ, pZ = model.forward_batched_train(X_batch, idx=batch_idx)
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
    ):
        super().__init__()

        kernel = self._get_kernel_cls(kernel_type, is_mggp=False)(sigma=sigma, lengthscale=lengthscale)

        # Inducing points
        if inducing_points is None:
            num_inducing = num_inducing or min(4000, X.shape[0])
            Z = X[:num_inducing].clone()
        else:
            Z = inducing_points.clone()
        M = Z.shape[0]

        # Create GP
        gp = SVGP(kernel, M=M, jitter=jitter, cholesky_mode=cholesky_mode, diagonal_only=diagonal_only)
        gp.Z = nn.Parameter(Z, requires_grad=False)

        # Initialize Lu
        self._init_lu_for_svgp(
            gp, kernel, Z, L, M, lu_rank, lu_init_iters,
            scale_multiplier, jitter, cholesky_mode, diagonal_only, seed
        )

        # Create model
        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.V = self._init_size_factors(Y, V)
        self._model.to(_default_device(X, device))


class VNNGP_NSF(NSF_Model):
    """
    Variational Nearest-Neighbor GP with NSF (Poisson) likelihood.

    Uses K-nearest neighbor sparsity for scalability to large datasets.
    Suitable for datasets with 100k+ cells.

    Args:
        X: Spatial coordinates, shape (N, D)
        Y: Gene expression counts, shape (genes, N)
        L: Number of latent factors
        K: Number of nearest neighbors for sparse approximation
        lengthscale: RBF/Matern kernel lengthscale
        sigma: Kernel variance (default 1.0)
        num_inducing: Number of inducing points (default: N)
        inducing_points: Optional pre-specified inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        precompute_knn: Whether to precompute KNN indices
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode

    Example:
        >>> model = VNNGP_NSF(X, Y, L=12, K=50, lengthscale=8.0)
        >>> model.to("cuda")
        >>> model.prior.knn_idx = full_knn_idx[batch_idx]
        >>> pY, qF, qZ, pZ = model.forward_batched_train(X_batch, idx=batch_idx)
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
    ):
        super().__init__()

        kernel = self._get_kernel_cls(kernel_type, is_mggp=False)(sigma=sigma, lengthscale=lengthscale)

        # Inducing points (default: all points for VNNGP)
        Z = X.clone() if inducing_points is None else inducing_points.clone()
        M = Z.shape[0]

        # Create GP
        gp = VNNGP(kernel, M=M, jitter=jitter, K=K)
        gp.Z = nn.Parameter(Z, requires_grad=True)

        # Initialize Lu
        self._init_lu_for_vnngp(
            gp, kernel, X, Z, L, K, lu_rank, lu_init_iters,
            scale_multiplier, subset_step, seed
        )

        # Create model
        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.prior.K = K
        self._model.V = self._init_size_factors(Y, V)

        if precompute_knn:
            self._setup_knn(X)

        self._model.to(_default_device(X, device))


class MGGP_SVGP_NSF(NSF_Model):
    """
    Multi-Group SVGP with NSF (Poisson) likelihood.

    Extends SVGP with group-specific kernel modulation for multi-sample
    or multi-condition spatial transcriptomics.

    Args:
        X: Spatial coordinates, shape (N, D)
        Y: Gene expression counts, shape (genes, N)
        groupsX: Group assignments for each cell, shape (N,)
        L: Number of latent factors
        lengthscale: RBF/Matern kernel lengthscale
        sigma: Kernel variance (default 1.0)
        group_diff_param: Controls group similarity (higher = more different)
        num_inducing: Number of inducing points
        inducing_points: Optional pre-specified inducing points
        inducing_groups: Group assignments for inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        cholesky_mode: 'exp' or 'softplus' for Lu diagonal constraint
        diagonal_only: If True, use diagonal covariance for Lu
        inducing_method: "kmeans" or "random"
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode

    Example:
        >>> model = MGGP_SVGP_NSF(X, Y, groupsX, L=12, lengthscale=8.0)
        >>> model.to("cuda")
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
    ):
        super().__init__()

        n_groups = int(groupsX.max().item() + 1)
        kernel = self._get_kernel_cls(kernel_type, is_mggp=True)(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        # Inducing points
        if inducing_points is None or inducing_groups is None:
            num_inducing = num_inducing or min(4000, X.shape[0])
            Z, groupsZ = mggp_kmeans_inducing_points(
                X, groupsX, num_inducing, seed=seed or 123,
                allocation="proportional" if inducing_method == "kmeans" else "equal"
            )
        else:
            Z, groupsZ = inducing_points.clone(), inducing_groups.clone()
        M = Z.shape[0]

        # Create GP
        gp = MGGP_SVGP(kernel, M=M, n_groups=n_groups, jitter=jitter,
                       cholesky_mode=cholesky_mode, diagonal_only=diagonal_only)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        # Initialize Lu
        self._init_lu_for_svgp(
            gp, kernel, Z, L, M, None, 10,
            scale_multiplier, jitter, cholesky_mode, diagonal_only, seed,
            groupsZ=groupsZ
        )

        # Create model
        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.V = self._init_size_factors(Y, V)
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
    ):
        super().__init__()

        n_groups = int(groupsX.max().item() + 1)
        kernel = self._get_kernel_cls(kernel_type, is_mggp=True)(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        # Inducing points (default: all points for VNNGP)
        if inducing_points is None or inducing_groups is None:
            Z, groupsZ = X.clone(), groupsX.clone()
        else:
            Z, groupsZ = inducing_points.clone(), inducing_groups.clone()
        M = Z.shape[0]

        # Create GP
        gp = MGGP_VNNGP(kernel, M=M, n_groups=n_groups, jitter=jitter, K=K)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        # Initialize Lu
        self._init_lu_for_vnngp(
            gp, kernel, X, Z, L, K, None, 10,
            scale_multiplier, subset_step, seed,
            groupsZ=groupsZ
        )

        # Create model
        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)
        self._model.prior.K = K
        self._model.V = self._init_size_factors(Y, V)

        if precompute_knn:
            self._setup_knn(X)

        self._model.to(_default_device(X, device))