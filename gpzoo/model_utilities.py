"""Model utility functions for initialization and inducing point selection.

This module provides helper functions used by the convenience model classes
in gpzoo.models.nsf. For most users, the convenience classes (SVGP_NSF, etc.)
are the recommended way to create models.
"""

import torch
import numpy as np

from typing import Optional
from sklearn.cluster import KMeans


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
