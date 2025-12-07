"""Model utility functions for initialization and inducing point selection.

This module provides helper functions used by the convenience model classes
in gpzoo.models.nsf. For most users, the convenience classes (SVGP_NSF, etc.)
are the recommended way to create models.
"""

import torch
import numpy as np

from typing import Optional
from sklearn.cluster import KMeans



def init_loadings(
    Y: torch.Tensor,
    L: int,
    mode: str = "pca",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Initialize loadings matrix W from data.
    
    Args:
        Y: Gene expression counts, shape (D, N) where D=genes, N=cells
        L: Number of latent factors
        mode: 'pca' (default, faster) or 'nmf'
        seed: Random seed for reproducibility
    
    Returns:
        W_init: Initialized loadings, shape (D, L), non-negative
    """
    import numpy as np
    
    Y_np = Y.detach().cpu().numpy().astype(np.float64)
    D, N = Y_np.shape
    
    if seed is not None:
        np.random.seed(seed)
    
    if mode == "pca":
        from sklearn.decomposition import PCA
        
        # PCA expects (n_samples, n_features), so transpose: (N, D)
        # We want components that explain variance across cells
        pca = PCA(n_components=L, random_state=seed)
        pca.fit(Y_np.T)  # Fit on (N, D)
        
        # components_ is (L, D), we want (D, L)
        W_init = pca.components_.T
        
        # Clamp to non-negative
        W_init = np.maximum(W_init, 0.0)
        
        # Handle case where clamping zeros out entire factors
        # Add small noise to zero columns
        col_sums = W_init.sum(axis=0)
        zero_cols = col_sums < 1e-10
        if zero_cols.any():
            W_init[:, zero_cols] = np.abs(np.random.randn(D, zero_cols.sum())) * 0.01
    
    elif mode == "nmf":
        from sklearn.decomposition import NMF
        
        # NMF needs non-negative input
        Y_nn = np.maximum(Y_np, 0.0)
        
        # NMF expects (n_samples, n_features), so transpose: (N, D)
        # But we want W (D, L), so we fit on (D, N) and take components
        # Actually: Y ≈ W @ H where Y is (D, N), W is (D, L), H is (L, N)
        # sklearn NMF: X ≈ W @ H where X is (n_samples, n_features)
        # So fit on Y.T (N, D) gives us H.T (D, L) as the loadings
        
        # Alternative: fit on Y (D, N) directly
        nmf = NMF(
            n_components=L,
            init='nndsvda',
            max_iter=200,
            random_state=seed,
        )
        W_init = nmf.fit_transform(Y_nn)  # (D, L)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'pca' or 'nmf'")
    
    # Scale to reasonable magnitude
    # Normalize so each factor has unit norm, then scale
    norms = np.linalg.norm(W_init, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    W_init = W_init / norms
    
    return torch.tensor(W_init, dtype=Y.dtype, device=Y.device)


def _default_device(reference: torch.Tensor, device: Optional[torch.device]) -> torch.device:
    """Determine the target device, defaulting to reference tensor's device."""
    if device is not None:
        return device
    if isinstance(reference, torch.Tensor):
        return reference.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_tensor(value, *, dtype=None, device=None):
    """Convert value to tensor if needed."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device if device is not None else value.device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _init_mu_from_lu(Lu_base: torch.Tensor, L: int, device: torch.device, seed: Optional[int] = None):
    """Initialize mu from Lu base using random projection."""
    if seed is not None:
        torch.manual_seed(seed)
    rank = Lu_base.shape[-1]
    noise = torch.randn(L, rank, 1, device=Lu_base.device)
    mu = torch.squeeze(Lu_base @ noise)
    return mu.to(device)


def _log_diagonals(Lu: torch.Tensor) -> torch.Tensor:
    """Convert Lu diagonal to log-space for storage as parameter.

    apply_constraints() expects diagonal in log-space (applies exp),
    so we store log(diag) while keeping lower triangle as-is.
    """
    diag = torch.diagonal(Lu, dim1=-2, dim2=-1)
    log_diag = torch.log(diag)
    lower = torch.tril(Lu, diagonal=-1)
    return lower + torch.diag_embed(log_diag)


def mggp_kmeans_inducing_points(
    X: torch.Tensor,
    groupsX: torch.Tensor,
    target_points: int,
    *,
    seed: int = 123,
    n_init: int = 10,
    allocation: str = "proportional",
):
    """Select inducing points using KMeans clustering per group.

    Args:
        X: Spatial coordinates, shape (N, D)
        groupsX: Group assignments, shape (N,)
        target_points: Total number of inducing points to select
        seed: Random seed for KMeans
        n_init: Number of KMeans initializations
        allocation: "proportional" or "equal" allocation across groups

    Returns:
        Z: Inducing point coordinates
        groupsZ: Group assignments for inducing points
    """
    if target_points <= 0:
        raise ValueError("target_points must be positive.")

    X_cpu = X.detach().cpu()
    groups_cpu = groupsX.detach().cpu()
    X_np = X_cpu.numpy()
    groups_np = groups_cpu.numpy()

    total_points = min(len(X_np), target_points)
    unique_groups, group_counts = np.unique(groups_np, return_counts=True)
    total_available = group_counts.sum()
    if total_points == 0 or total_available == 0:
        raise ValueError("No points available for initializing inducing locations.")

    total_points = min(total_points, total_available)

    if allocation not in {"proportional", "equal"}:
        raise ValueError("allocation must be 'proportional' or 'equal'.")

    if allocation == "proportional":
        desired = group_counts / total_available * total_points
    else:  # equal
        desired = np.full_like(group_counts, fill_value=total_points / len(unique_groups), dtype=float)

    fractional, integral = np.modf(desired)
    targets = integral.astype(int)
    targets = np.minimum(targets, group_counts)
    assigned = targets.sum()
    leftover = total_points - assigned

    def _distribute_leftover(ordering):
        nonlocal leftover
        for idx in ordering:
            if leftover <= 0:
                break
            capacity = group_counts[idx] - targets[idx]
            if capacity <= 0:
                continue
            take = min(capacity, leftover)
            targets[idx] += take
            leftover -= take

    if leftover > 0:
        priority = np.argsort(-fractional)
        _distribute_leftover(priority)

    if leftover > 0:
        remaining_capacity = np.argsort(-(group_counts - targets))
        _distribute_leftover(remaining_capacity)

    if leftover != 0:
        raise RuntimeError("Failed to allocate inducing points across groups.")

    centers_list = []
    group_list = []

    for g, n_clusters in zip(unique_groups, targets):
        if n_clusters <= 0:
            continue
        mask = groups_np == g
        count = mask.sum()
        if count == 0:
            continue
        n_clusters = min(n_clusters, count)
        if n_clusters <= 0:
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
        kmeans.fit(X_np[mask])
        centers_list.append(kmeans.cluster_centers_)
        group_list.append(np.full(n_clusters, g, dtype=np.int64))

    if not centers_list:
        # fallback to random sample
        idx = torch.randperm(len(X_cpu))[:total_points]
        return X.cpu()[idx], groups_cpu[idx]

    centers = np.concatenate(centers_list, axis=0)
    groupsZ = np.concatenate(group_list, axis=0)

    Z = torch.tensor(centers, dtype=X.dtype, device=X.device)
    groupsZ_t = torch.tensor(groupsZ, dtype=groupsX.dtype, device=groupsX.device)
    return Z, groupsZ_t


def init_mggp_inducing_points(
    X: torch.Tensor,
    groupsX: torch.Tensor,
    num_inducing: int,
    *,
    method: str = "kmeans",
    seed: int = 123,
    **kwargs,
):
    """Initialize inducing points for MGGP models.

    Args:
        X: Spatial coordinates
        groupsX: Group assignments
        num_inducing: Number of inducing points
        method: "kmeans" or "random"
        seed: Random seed
        **kwargs: Additional arguments for the method

    Returns:
        Z: Inducing point coordinates
        groupsZ: Group assignments for inducing points
    """
    if method == "kmeans":
        return mggp_kmeans_inducing_points(
            X,
            groupsX,
            num_inducing,
            seed=seed,
            n_init=kwargs.get("n_init", 10),
            allocation=kwargs.get("allocation", "proportional"),
        )
    if method == "random":
        idx = torch.randperm(len(X), device=X.device)[:num_inducing]
        return X[idx], groupsX[idx]
    raise ValueError(f"Unknown method '{method}' for init_mggp_inducing_points.")
