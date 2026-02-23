"""Model utility functions for initialization and inducing point selection.

This module provides helper functions used by the convenience model classes
in gpzoo.models.nsf. For most users, the convenience classes (SVGP_NSF, etc.)
are the recommended way to create models.
"""

import torch
import numpy as np

from typing import Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import warnings



def init_loadings(
    Y: torch.Tensor,
    L: int,
    mode: str = "svd",
    seed: Optional[int] = None,
    return_factors: bool = False,
) -> torch.Tensor:
    """
    Initialize loadings matrix W (and optionally factors F) from data.
    
    For SVD mode (default), decomposes Y = U @ S @ V.T without centering:
        W = U @ S^{1/2}  (shape D, L)
        F = S^{1/2} @ V.T  (shape L, N)
    
    Args:
        Y: Gene expression counts, shape (D, N) where D=genes, N=cells
        L: Number of latent factors
        mode: 'svd' (default), 'pca', or 'nmf'
        seed: Random seed for reproducibility
        return_factors: If True, also return F for initializing GP mean (useful for VNNGP)
    
    Returns:
        W_init: Initialized loadings, shape (D, L), non-negative
        F_init: (only if return_factors=True) Factors in log-space, shape (L, N)
                Ready to initialize GP mu as log(F + eps)
    """

    
    Y_np = Y.detach().cpu().numpy().astype(np.float64)
    D, N = Y_np.shape
    
    if seed is not None:
        np.random.seed(seed)
    
    if mode == "svd":
        # Use truncated SVD for efficiency with large matrices
        from sklearn.utils.extmath import randomized_svd

        # randomized_svd returns truncated U, S, Vt directly
        # This is much more efficient than full SVD for large matrices
        U_L, S_L, Vt_L = randomized_svd(Y_np, n_components=L, random_state=seed)
        # U_L: (D, L), S_L: (L,), Vt_L: (L, N)

        S_sqrt = np.sqrt(S_L)

        # W = U @ S^{1/2}
        W_init = U_L * S_sqrt[np.newaxis, :]  # (D, L)

        # F = S^{1/2} @ V.T
        F_init = S_sqrt[:, np.newaxis] * Vt_L  # (L, N)
        
        # Clamp W to non-negative
        W_init = np.maximum(W_init, 0.0)
        
        # Handle zero columns
        col_norms = np.linalg.norm(W_init, axis=0)
        zero_cols = col_norms < 1e-10
        if zero_cols.any():
            n_zero = zero_cols.sum()
            W_init[:, zero_cols] = np.abs(np.random.randn(D, n_zero)) * 0.01
        
        # F needs to be positive for log transform, clamp then log
        F_init = np.maximum(F_init, 1e-6)
        F_log = np.log(F_init)
    
    elif mode == "pca":
        # PCA = SVD on centered data
        from sklearn.utils.extmath import randomized_svd

        Y_mean = Y_np.mean(axis=1, keepdims=True)
        Y_centered = Y_np - Y_mean

        # Use truncated SVD for efficiency
        U_L, S_L, Vt_L = randomized_svd(Y_centered, n_components=L, random_state=seed)
        # U_L: (D, L), S_L: (L,), Vt_L: (L, N)

        S_sqrt = np.sqrt(S_L)

        # W = U @ S^{1/2}
        W_init = U_L * S_sqrt[np.newaxis, :]  # (D, L)

        # F = S^{1/2} @ V.T (from centered data)
        F_init = S_sqrt[:, np.newaxis] * Vt_L  # (L, N)

        # Clamp W to non-negative
        W_init = np.maximum(W_init, 0.0)

        # Handle zero columns
        col_norms = np.linalg.norm(W_init, axis=0)
        zero_cols = col_norms < 1e-10
        if zero_cols.any():
            n_zero = zero_cols.sum()
            W_init[:, zero_cols] = np.abs(np.random.randn(D, n_zero)) * 0.01

        # F needs to be positive for log transform, clamp then log
        F_init = np.maximum(F_init, 1e-6)
        F_log = np.log(F_init)
    
    elif mode == "nmf":
        from sklearn.decomposition import NMF
        
        Y_nn = np.maximum(Y_np, 0.0)
        
        nmf = NMF(
            n_components=L,
            init='nndsvda',
            max_iter=200,
            random_state=seed,
        )
        W_init = nmf.fit_transform(Y_nn)  # (D, L)
        F_init = nmf.components_  # (L, N) - already non-negative
        
        F_init = np.maximum(F_init, 1e-6)
        F_log = np.log(F_init)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'svd', 'pca', or 'nmf'")
    
    W_out = torch.tensor(W_init, dtype=Y.dtype, device=Y.device)
    
    if return_factors:
        F_out = torch.tensor(F_log, dtype=Y.dtype, device=Y.device)
        return W_out, F_out
    
    return W_out



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


def kmeans_inducing_points(
    X: torch.Tensor,
    target_points: int,
    *,
    seed: int = 123,
    n_init: int = 10,
):
    """Select inducing points using KMeans clustering (no groups).

    Args:
        X: Spatial coordinates, shape (N, D)
        target_points: Number of inducing points to select
        seed: Random seed for KMeans
        n_init: Number of KMeans initializations

    Returns:
        Z: Inducing point coordinates, shape (M, D)
    """
    if target_points <= 0:
        raise ValueError("target_points must be positive.")

    X_np = X.detach().cpu().numpy()
    N = len(X_np)
    M = min(target_points, N)

    if M == 0:
        raise ValueError("No points available for initializing inducing locations.")

    kmeans = KMeans(n_clusters=M, random_state=seed, n_init=n_init)
    kmeans.fit(X_np)

    Z = torch.tensor(kmeans.cluster_centers_, dtype=X.dtype, device=X.device)
    return Z


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
        allocation: Allocation strategy across groups:
            - "proportional": Allocate points proportionally to group sizes (default)
            - "equal": Allocate equal points to each group
            - "derived": Run K-means on all data for optimal spatial coverage,
              then use KNN (k=5, distance-weighted) to classify centroids to groups.
              Falls back to proportional for groups with no assigned points.

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

    if allocation not in {"proportional", "equal", "derived"}:
        raise ValueError("allocation must be 'proportional', 'equal', or 'derived'.")

    # ========== 'derived' mode: K-means on all data + KNN classification ==========
    if allocation == "derived":
        # Step 1: Run K-means on ALL data (ignoring groups) for optimal spatial coverage
        kmeans = KMeans(n_clusters=total_points, random_state=seed, n_init=n_init)
        kmeans.fit(X_np)
        Z_derived = kmeans.cluster_centers_  # (M, 2) - optimal spatial coverage

        # Step 2: Train KNN classifier to predict group from coordinates
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn.fit(X_np, groups_np)

        # Step 3: Predict groups for centroids
        groupsZ_derived = knn.predict(Z_derived)  # (M,)

        # Step 4: Validation - ensure each group has at least some inducing points
        derived_groups = np.unique(groupsZ_derived)
        missing_groups = set(unique_groups) - set(derived_groups)

        if missing_groups:
            warnings.warn(
                f"allocation='derived' failed to assign inducing points to groups "
                f"{sorted(missing_groups)}. Sampling donor points to fill missing groups.",
                UserWarning
            )

            rng = np.random.default_rng(seed)

            # Mutable work lists; positions here are stable indices (we never reorder).
            # Positions 0..M-1 come from Z_derived; later positions are appended fallbacks.
            Z_work = list(Z_derived)
            G_work = list(groupsZ_derived.tolist())

            # group_to_pos[g] = set of currently-active positions for group g
            group_to_pos = {}
            for pos, g in enumerate(G_work):
                group_to_pos.setdefault(int(g), set()).add(pos)

            removed_pos = set()  # positions to exclude when reconstructing

            # donor_threshold: a donor group must have strictly more than this many
            # active points.  Start at 1 so donors always keep ≥ 1 point after giving.
            donor_threshold = 1

            for missing_g in sorted(missing_groups):
                # Lower threshold until we can find at least one donor.
                t = donor_threshold
                while t >= 0:
                    donors = [
                        g for g, pos_set in group_to_pos.items()
                        if len(pos_set) > t
                    ]
                    if donors:
                        break
                    t -= 1

                if not donors:
                    warnings.warn(
                        f"No donors available for group {missing_g}; "
                        f"it will have no inducing points.",
                        UserWarning
                    )
                    continue

                n_donors = len(donors)

                # Remove 1 point from each donor group (random choice).
                for donor_g in donors:
                    avail = list(group_to_pos[donor_g])
                    pick = int(rng.integers(len(avail)))
                    pos = avail[pick]
                    removed_pos.add(pos)
                    group_to_pos[donor_g].discard(pos)

                # Sample n_donors data points from the missing group and append.
                X_mg = X_np[groups_np == missing_g]
                new_pts = X_mg[
                    rng.choice(len(X_mg), size=n_donors, replace=(n_donors > len(X_mg)))
                ]
                start = len(Z_work)
                Z_work.extend(new_pts)
                G_work.extend([missing_g] * n_donors)
                group_to_pos.setdefault(missing_g, set()).update(
                    range(start, start + n_donors)
                )

            # Reconstruct: keep every position not in removed_pos.
            keep = [i for i in range(len(Z_work)) if i not in removed_pos]
            Z_derived = np.array([Z_work[i] for i in keep])
            groupsZ_derived = np.array([G_work[i] for i in keep], dtype=groupsZ_derived.dtype)

        return (
            torch.tensor(Z_derived, dtype=X.dtype, device=X.device),
            torch.tensor(groupsZ_derived, dtype=groupsX.dtype, device=groupsX.device)
        )

    # ========== 'proportional' and 'equal' modes ==========
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
