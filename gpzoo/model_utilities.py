import torch
from torch import nn
import numpy as np

from typing import Optional
from sklearn.cluster import KMeans

from .kernels import (
    batched_Matern32,
    batched_MGGP_RBF,
)
from .gp import (
    SVGP,
    VNNGP,
    MGGP_SVGP,
    MGGP_VNNGP,
)
from .likelihoods import NSF2
from .utilities import (
    init_Lu_nsf,
    init_Lu,
)


def _default_device(reference: torch.Tensor, device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if isinstance(reference, torch.Tensor):
        return reference.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_tensor(value, *, dtype=None, device=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device if device is not None else value.device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _init_mu_from_lu(Lu_base: torch.Tensor, L: int, device: torch.device, seed: Optional[int] = None):
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


def build_nsf_svgp(
    X: torch.Tensor,
    Y: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    *,
    L: int = 4,
    kernel=None,
    jitter: float = 1e-5,
    inducing_points: Optional[torch.Tensor] = None,
    lu_rank: Optional[int] = None,
    lu_init_iters: int = 10,
    lu_use_cholesky: bool = True,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
):
    kernel = kernel or batched_Matern32(sigma=1.0, lengthscale=0.3)
    inducing = inducing_points if inducing_points is not None else X
    inducing = inducing.clone()
    M = inducing.shape[0]

    gp = SVGP(kernel, M=M, jitter=jitter)
    gp.Z = nn.Parameter(inducing, requires_grad=False)

    rank = lu_rank or M
    Lu_base = init_Lu_nsf(kernel, inducing, inducing, K=rank, niter=lu_init_iters,
                          use_cholesky=lu_use_cholesky, jitter=jitter)
    if Lu_base.dim() == 2:
        Lu_base = Lu_base.unsqueeze(0)
    Lu = Lu_base.expand(L, -1, -1).contiguous()
    # Convert diagonal to log-space if using cholesky (apply_constraints will exp it back)
    Lu_param = _log_diagonals(Lu) if lu_use_cholesky else Lu
    gp.Lu = nn.Parameter(Lu_param.clone().detach())
    gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

    model = NSF2(gp, Y, L=L)
    D = Y.shape[0]
    model.W = nn.Parameter(torch.randn(D, L, device=Y.device, dtype=Y.dtype))

    if V is None:
        V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
    model.V = nn.Parameter(V.clone())

    target_device = _default_device(X, device)
    model.to(target_device)
    return model


def build_nsf_vnngp(
    X: torch.Tensor,
    Y: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    *,
    L: int = 4,
    kernel=None,
    jitter: float = 1e-5,
    K: int = 50,
    inducing_points: Optional[torch.Tensor] = None,
    lu_reference_points: Optional[torch.Tensor] = None,
    subset_step: int = 10,
    lu_rank: Optional[int] = None,
    lu_init_iters: int = 10,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    precompute_knn: bool = True,
):
    kernel = kernel or batched_Matern32(sigma=1.0, lengthscale=0.3)
    inducing = inducing_points if inducing_points is not None else X
    inducing = inducing.clone()

    gp = VNNGP(kernel, M=inducing.shape[0], jitter=jitter, K=K)
    gp.Z = nn.Parameter(inducing, requires_grad=True)

    ref = lu_reference_points
    if ref is None:
        ref = X[::max(subset_step, 1)]
    rank = lu_rank or K
    Lu_base = init_Lu_nsf(kernel, X, ref, K=rank, niter=lu_init_iters)
    if Lu_base.dim() == 2:
        Lu_base = Lu_base.unsqueeze(0)
    Lu = Lu_base.expand(L, -1, -1).contiguous()
    gp.Lu = nn.Parameter(Lu.clone().detach())
    gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

    model = NSF2(gp, Y, L=L)
    model.prior.K = K
    D = Y.shape[0]
    model.W = nn.Parameter(torch.randn(D, L, device=Y.device, dtype=Y.dtype))

    if V is None:
        V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
    model.V = nn.Parameter(V.clone())

    if precompute_knn:
        knn_idx = model.prior.calculate_knn(X)[:, :-1]
        model.prior.knn_idx = knn_idx
        knn_idz = model.prior.calculate_knn(model.prior.Z)[:, 1:]
        model.prior.knn_idz = knn_idz

    target_device = _default_device(X, device)
    model.to(target_device)
    return model


def build_mggp_nsf_svgp(
    X: torch.Tensor,
    groupsX: torch.Tensor,
    Y: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    *,
    L: int = 10,
    kernel=None,
    jitter: float = 1e-5,
    inducing_points: torch.Tensor,
    inducing_groups: torch.Tensor,
    lu_use_cholesky: bool = True,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
):
    kernel = kernel or batched_MGGP_RBF(sigma=1.0, lengthscale=2.0, group_diff_param=10.0, n_groups=len(torch.unique(groupsX)))
    Z = inducing_points.clone()
    groupsZ = inducing_groups.clone()
    M = Z.shape[0]

    gp = MGGP_SVGP(kernel, M=M, n_groups=int(groupsZ.max().item() + 1), jitter=jitter)
    gp.Z = nn.Parameter(Z, requires_grad=False)
    gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

    Lu_base = init_Lu(kernel, gp.Z.detach(), gp.groupsZ.detach(), gp.Z.detach(), gp.groupsZ.detach(), K=M,
                      use_cholesky=lu_use_cholesky, jitter=jitter)
    if Lu_base.dim() == 2:
        Lu_base = Lu_base.unsqueeze(0)
    Lu = Lu_base.expand(L, -1, -1).contiguous()
    # Convert diagonal to log-space if using cholesky (apply_constraints will exp it back)
    Lu_param = _log_diagonals(Lu) if lu_use_cholesky else Lu
    gp.Lu = nn.Parameter(Lu_param.clone().detach())
    gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

    model = NSF2(gp, Y, L=L)
    D = Y.shape[0]
    model.W = nn.Parameter(torch.randn(D, L, device=Y.device, dtype=Y.dtype))

    if V is None:
        V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
    model.V = nn.Parameter(V.clone())

    target_device = _default_device(X, device)
    model.to(target_device)
    return model


def build_mggp_nsf_vnngp(
    X: torch.Tensor,
    groupsX: torch.Tensor,
    Y: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    *,
    L: int = 10,
    kernel=None,
    jitter: float = 1e-5,
    K: int = 50,
    inducing_points: Optional[torch.Tensor] = None,
    inducing_groups: Optional[torch.Tensor] = None,
    lu_reference_points: Optional[torch.Tensor] = None,
    lu_reference_groups: Optional[torch.Tensor] = None,
    subset_step: int = 10,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    precompute_knn: bool = True,
):
    kernel = kernel or batched_MGGP_RBF(sigma=1.0, lengthscale=2.0, group_diff_param=10.0, n_groups=int(groupsX.max().item() + 1))
    Z = inducing_points.clone() if inducing_points is not None else X.clone()
    groupsZ = inducing_groups.clone() if inducing_groups is not None else groupsX.clone()

    gp = MGGP_VNNGP(kernel, M=Z.shape[0], n_groups=int(groupsZ.max().item() + 1), jitter=jitter, K=K)
    gp.Z = nn.Parameter(Z, requires_grad=False)
    gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

    ref_Z = lu_reference_points if lu_reference_points is not None else gp.Z.detach()[::max(subset_step, 1)]
    ref_groups = lu_reference_groups if lu_reference_groups is not None else gp.groupsZ.detach()[::max(subset_step, 1)]
    Lu_base = init_Lu(kernel, gp.Z.detach(), gp.groupsZ.detach(), ref_Z, ref_groups, K=K)
    if Lu_base.dim() == 2:
        Lu_base = Lu_base.unsqueeze(0)
    Lu = Lu_base.expand(L, -1, -1).contiguous()
    gp.Lu = nn.Parameter(Lu.clone().detach())
    gp.mu = nn.Parameter(_init_mu_from_lu(Lu_base.squeeze(0), L, gp.Lu.device, seed))

    model = NSF2(gp, Y, L=L)
    model.prior.K = K
    D = Y.shape[0]
    model.W = nn.Parameter(torch.randn(D, L, device=Y.device, dtype=Y.dtype))

    if V is None:
        V = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)
    model.V = nn.Parameter(V.clone())

    if precompute_knn:
        knn_idx = model.prior.calculate_knn(X).clone()[:, :-1]
        model.prior.knn_idx = knn_idx
        knn_idz = model.prior.calculate_knn(model.prior.Z).clone()[:, 1:]
        model.prior.knn_idz = knn_idz

    target_device = _default_device(X, device)
    model.to(target_device)
    return model
