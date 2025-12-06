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
    _log_diagonals,
    _default_device,
)


class SVGP_NSF(nn.Module):
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
        lu_use_cholesky: Use Cholesky initialization for Lu
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode: 'softplus', 'exp', 'exp_sum', or 'projected'

    Example:
        >>> model = SVGP_NSF(X, Y, L=12, lengthscale=8.0)
        >>> model.to("cuda")
        >>> # Training loop...
        >>> pY, qF, qZ, pZ = model.forward_batched_train(X_batch, idx=batch_idx)
    """

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
        lu_use_cholesky: bool = True,
        lu_rank: Optional[int] = None,
        lu_init_iters: int = 10,
        scale_multiplier: float = 1e-6,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
    ):
        super().__init__()

        # Create kernel
        kernel_cls = batched_Matern32 if kernel_type == "matern32" else batched_RBF
        kernel = kernel_cls(sigma=sigma, lengthscale=lengthscale)

        # Determine inducing points
        if inducing_points is None:
            if num_inducing is None:
                num_inducing = min(4000, X.shape[0])
            Z = X[:num_inducing].clone()
        else:
            Z = inducing_points.clone()
        M = Z.shape[0]

        # Create GP
        gp = SVGP(kernel, M=M, jitter=jitter)
        gp.Z = nn.Parameter(Z, requires_grad=False)

        # Initialize Lu
        rank = lu_rank or M
        Lu_base = init_Lu_nsf(
            kernel, Z, Z, K=rank, niter=lu_init_iters,
            use_cholesky=lu_use_cholesky, jitter=jitter
        )
        if Lu_base.dim() == 2:
            Lu_base = Lu_base.unsqueeze(0)
        Lu = Lu_base.expand(L, -1, -1).contiguous()
        Lu_param = _log_diagonals(scale_multiplier * Lu) if lu_use_cholesky else scale_multiplier * Lu
        gp.Lu = nn.Parameter(Lu_param.clone().detach())
        gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

        # Create NSF likelihood wrapper with loadings_mode
        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)

        if V is None:
            V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
        self._model.V = nn.Parameter(V.clone())

        # Move to device
        target_device = _default_device(X, device)
        self._model.to(target_device)

    @property
    def prior(self):
        return self._model.prior

    @property
    def W(self):
        return self._model.W.value

    @W.setter
    def W(self, value):
        self._model.W._raw.data = value

    @property
    def V(self):
        return self._model.V

    @V.setter
    def V(self, value):
        self._model.V = value

    def forward(self, X, E=10, verbose=False, **kwargs):
        return self._model.forward(X, E=E, verbose=verbose, **kwargs)

    def forward_batched(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched(X, idx, idy=idy, E=E, verbose=verbose, **kwargs)

    def forward_batched_train(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched_train(X, idx, idy=idy, E=E, verbose=verbose, **kwargs)

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


class VNNGP_NSF(nn.Module):
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
        num_inducing: Number of inducing points (default: N, i.e., all points)
        inducing_points: Optional pre-specified inducing points
        V: Optional size factors, shape (N,)
        jitter: Numerical stability term
        kernel_type: "matern32" or "rbf"
        precompute_knn: Whether to precompute KNN indices
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode: 'softplus', 'exp', 'exp_sum', or 'projected'

    Example:
        >>> model = VNNGP_NSF(X, Y, L=12, K=50, lengthscale=8.0)
        >>> model.to("cuda")
        >>> # Set KNN indices for batch before forward pass
        >>> model.prior.knn_idx = full_knn_idx[batch_idx]
        >>> pY, qF, qZ, pZ = model.forward_batched_train(X_batch, idx=batch_idx)
    """

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

        # Create kernel
        kernel_cls = batched_Matern32 if kernel_type == "matern32" else batched_RBF
        kernel = kernel_cls(sigma=sigma, lengthscale=lengthscale)

        # Determine inducing points (default: all points for VNNGP)
        if inducing_points is None:
            Z = X.clone()
        else:
            Z = inducing_points.clone()
        M = Z.shape[0]

        # Create GP
        gp = VNNGP(kernel, M=M, jitter=jitter, K=K)
        gp.Z = nn.Parameter(Z, requires_grad=True)

        # Initialize Lu
        ref = X[::max(subset_step, 1)]
        rank = lu_rank or K
        Lu_base = init_Lu_nsf(kernel, X, ref, K=rank, niter=lu_init_iters)
        if Lu_base.dim() == 2:
            Lu_base = Lu_base.unsqueeze(0)
        Lu = Lu_base.expand(L, -1, -1).contiguous()
        gp.Lu = nn.Parameter((scale_multiplier * Lu).clone().detach())
        gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

        # Create NSF likelihood wrapper
        self._model = NSF2(gp, Y, L=L)
        self._model.projection_mode = loadings_mode  # Use specified loadings mode
        self._model.prior.K = K
        D = Y.shape[0]
        self._model.W = nn.Parameter(torch.rand(D, L, device=Y.device, dtype=Y.dtype))

        if V is None:
            V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
        self._model.V = nn.Parameter(V.clone())

        # Precompute KNN
        if precompute_knn:
            knn_idx = self._model.prior.calculate_knn(X)[:, :-1]
            self._model.prior.knn_idx = knn_idx
            knn_idz = self._model.prior.calculate_knn(self._model.prior.Z)[:, 1:]
            self._model.prior.knn_idz = knn_idz

        # Move to device
        target_device = _default_device(X, device)
        self._model.to(target_device)

    @property
    def prior(self):
        return self._model.prior

    @property
    def W(self):
        return self._model.W.value

    @W.setter
    def W(self, value):
        self._model.W._raw.data = value

    @property
    def V(self):
        return self._model.V

    @V.setter
    def V(self, value):
        self._model.V = value

    def forward(self, X, E=10, verbose=False, **kwargs):
        return self._model.forward(X, E=E, verbose=verbose, **kwargs)

    def forward_batched(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched(X, idx, idy=idy, E=E, verbose=verbose, **kwargs)

    def forward_batched_train(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched_train(X, idx, idy=idy, E=E, verbose=verbose, **kwargs)

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


class MGGP_SVGP_NSF(nn.Module):
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
        inducing_method: "kmeans" or "random"
        device: Target device
        seed: Random seed for reproducibility
        loadings_mode: Loadings W transformation mode: 'softplus', 'exp', 'exp_sum', or 'projected'

    Example:
        >>> model = MGGP_SVGP_NSF(X, Y, groupsX, L=12, lengthscale=8.0)
        >>> model.to("cuda")
        >>> pY, qF, qZ, pZ = model.forward_batched_train(
        ...     X_batch, idx=batch_idx, groupsX=groups_batch
        ... )
    """

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
        lu_use_cholesky: bool = True,
        scale_multiplier: float = 1e-6,
        inducing_method: Literal["kmeans", "random"] = "kmeans",
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        loadings_mode: str = "softplus",
    ):
        super().__init__()

        n_groups = int(groupsX.max().item() + 1)

        # Create kernel
        kernel_cls = batched_MGGP_Matern32 if kernel_type == "matern32" else batched_MGGP_RBF
        kernel = kernel_cls(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        # Determine inducing points
        if inducing_points is None or inducing_groups is None:
            if num_inducing is None:
                num_inducing = min(4000, X.shape[0])
            Z, groupsZ = mggp_kmeans_inducing_points(
                X, groupsX, num_inducing,
                seed=seed or 123,
                allocation="proportional" if inducing_method == "kmeans" else "equal"
            )
        else:
            Z = inducing_points.clone()
            groupsZ = inducing_groups.clone()
        M = Z.shape[0]

        # Create GP
        gp = MGGP_SVGP(kernel, M=M, n_groups=n_groups, jitter=jitter)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        # Initialize Lu
        Lu_base = init_Lu(
            kernel, Z, groupsZ, Z, groupsZ, K=M,
            use_cholesky=lu_use_cholesky, jitter=jitter
        )
        if Lu_base.dim() == 2:
            Lu_base = Lu_base.unsqueeze(0)
        Lu = Lu_base.expand(L, -1, -1).contiguous()
        Lu_param = _log_diagonals(scale_multiplier * Lu) if lu_use_cholesky else scale_multiplier * Lu
        gp.Lu = nn.Parameter(Lu_param.clone().detach())
        gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

        # Create NSF likelihood wrapper with loadings_mode
        self._model = NSF2(gp, Y, L=L, loadings_mode=loadings_mode)

        if V is None:
            V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
        self._model.V = nn.Parameter(V.clone())

        # Move to device
        target_device = _default_device(X, device)
        self._model.to(target_device)

    @property
    def prior(self):
        return self._model.prior

    @property
    def W(self):
        return self._model.W.value

    @W.setter
    def W(self, value):
        self._model.W._raw.data = value

    @property
    def V(self):
        return self._model.V

    @V.setter
    def V(self, value):
        self._model.V = value

    def forward(self, X, groupsX, E=10, verbose=False, **kwargs):
        return self._model.forward(X, E=E, verbose=verbose, groupsX=groupsX, **kwargs)

    def forward_batched(self, X, idx, groupsX, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched(X, idx, idy=idy, E=E, verbose=verbose, groupsX=groupsX, **kwargs)

    def forward_batched_train(self, X, idx, groupsX, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched_train(X, idx, idy=idy, E=E, verbose=verbose, groupsX=groupsX, **kwargs)

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


class MGGP_VNNGP_NSF(nn.Module):
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
        loadings_mode: Loadings W transformation mode: 'softplus', 'exp', 'exp_sum', or 'projected'

    Example:
        >>> model = MGGP_VNNGP_NSF(X, Y, groupsX, L=12, K=50, lengthscale=8.0)
        >>> model.to("cuda")
        >>> # Set KNN indices for batch
        >>> model.prior.knn_idx = full_knn_idx[batch_idx]
        >>> pY, qF, qZ, pZ = model.forward_batched_train(
        ...     X_batch, idx=batch_idx, groupsX=groups_batch
        ... )
    """

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

        # Create kernel
        kernel_cls = batched_MGGP_Matern32 if kernel_type == "matern32" else batched_MGGP_RBF
        kernel = kernel_cls(
            sigma=sigma, lengthscale=lengthscale,
            group_diff_param=group_diff_param, n_groups=n_groups
        )

        # Determine inducing points (default: all points for VNNGP)
        if inducing_points is None or inducing_groups is None:
            Z = X.clone()
            groupsZ = groupsX.clone()
        else:
            Z = inducing_points.clone()
            groupsZ = inducing_groups.clone()
        M = Z.shape[0]

        # Create GP
        gp = MGGP_VNNGP(kernel, M=M, n_groups=n_groups, jitter=jitter, K=K)
        gp.Z = nn.Parameter(Z, requires_grad=False)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

        # Initialize Lu
        ref_Z = Z[::max(subset_step, 1)]
        ref_groups = groupsZ[::max(subset_step, 1)]
        Lu_base = init_Lu(kernel, Z, groupsZ, ref_Z, ref_groups, K=K)
        if Lu_base.dim() == 2:
            Lu_base = Lu_base.unsqueeze(0)
        Lu = Lu_base.expand(L, -1, -1).contiguous()
        gp.Lu = nn.Parameter((scale_multiplier * Lu).clone().detach())
        gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

        # Create NSF likelihood wrapper
        self._model = NSF2(gp, Y, L=L)
        self._model.projection_mode = loadings_mode  # Use specified loadings mode
        self._model.prior.K = K
        D = Y.shape[0]
        self._model.W = nn.Parameter(torch.rand(D, L, device=Y.device, dtype=Y.dtype))

        if V is None:
            V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
        self._model.V = nn.Parameter(V.clone())

        # Precompute KNN
        if precompute_knn:
            knn_idx = self._model.prior.calculate_knn(X)[:, :-1]
            self._model.prior.knn_idx = knn_idx
            knn_idz = self._model.prior.calculate_knn(self._model.prior.Z)[:, 1:]
            self._model.prior.knn_idz = knn_idz

        # Move to device
        target_device = _default_device(X, device)
        self._model.to(target_device)

    @property
    def prior(self):
        return self._model.prior

    @property
    def W(self):
        return self._model.W.value

    @W.setter
    def W(self, value):
        self._model.W._raw.data = value

    @property
    def V(self):
        return self._model.V

    @V.setter
    def V(self, value):
        self._model.V = value

    def forward(self, X, groupsX, E=10, verbose=False, **kwargs):
        return self._model.forward(X, E=E, verbose=verbose, groupsX=groupsX, **kwargs)

    def forward_batched(self, X, idx, groupsX, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched(X, idx, idy=idy, E=E, verbose=verbose, groupsX=groupsX, **kwargs)

    def forward_batched_train(self, X, idx, groupsX, idy=None, E=10, verbose=False, **kwargs):
        return self._model.forward_batched_train(X, idx, idy=idy, E=E, verbose=verbose, groupsX=groupsX, **kwargs)

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
