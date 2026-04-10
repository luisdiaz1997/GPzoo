"""KNN utilities for local GP priors (LCGP, VNNGP, MGGP variants).

Two strategies for selecting K conditioning neighbors per query point:
- ``knn``: deterministic FAISS L2 K-nearest neighbors.
- ``probabilistic``: sample K neighbors per row weighted by the kernel
  ``k(x_i, z_j)``, using the Gumbel-max trick (sampling without replacement).

Both strategies return ``(N, K+1)`` index tensors with column 0 = self
(when ``X = Z`` at training time, or self-as-query at inference). Callers
slice ``[:, :-1]`` for the inference path (self-inclusive K entries) and
``[:, 1:]`` for the KL path (self-exclusive K entries).
"""

from __future__ import annotations

import faiss
import torch


def _faiss_knn(X: torch.Tensor, Z: torch.Tensor, K: int) -> torch.Tensor:
    """K+1 nearest neighbors via FAISS L2. Returns (N, K+1) long tensor on CPU."""
    X_np = X.detach().cpu().float().numpy()
    Z_np = Z.detach().cpu().float().numpy()
    index = faiss.IndexFlatL2(Z_np.shape[-1])
    index.add(Z_np)
    _, indices = index.search(X_np, K + 1)
    return torch.tensor(indices, dtype=torch.long, device="cpu")


def _sample_without_replacement(weights: torch.Tensor, K: int) -> torch.Tensor:
    """Sample K column indices per row without replacement (Gumbel-max trick)."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(weights) + 1e-20) + 1e-20)
    log_weights = torch.log(weights.clamp(min=1e-20))
    _, topk_indices = (log_weights + gumbel_noise).topk(K, dim=1)
    return topk_indices


def _probabilistic_knn(
    X: torch.Tensor,
    Z: torch.Tensor,
    K: int,
    kernel,
    multigroup: bool = False,
    groupsX: torch.Tensor | None = None,
    groupsZ: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample K neighbors per row weighted by kernel(X, Z), prepend self.

    Returns ``(N, K+1)`` long tensor with column 0 = self index.
    """
    N = X.shape[0]
    K = min(K, Z.shape[0] - 1)

    # Run on CPU: kernel params and inputs may be on different devices at
    # construction time (kernel is created on CPU, inputs may already be on GPU).
    X = X.detach().cpu()
    Z = Z.detach().cpu()
    if groupsX is not None:
        groupsX = groupsX.detach().cpu()
    if groupsZ is not None:
        groupsZ = groupsZ.detach().cpu()
    kernel_was = next(kernel.parameters()).device
    kernel.cpu()

    with torch.no_grad():
        if multigroup:
            weights = kernel(X, Z, groupsX, groupsZ)
        else:
            weights = kernel(X, Z)

    kernel.to(kernel_was)

    weights = weights.squeeze()  # ensure (N, M)

    # Zero out self when querying training points (X is Z)
    if N == Z.shape[0]:
        weights[torch.arange(N), torch.arange(N)] = 0.0

    row_sums = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    weights = weights / row_sums

    sampled = _sample_without_replacement(weights, K)  # (N, K)
    self_idx = torch.arange(N, device=sampled.device).unsqueeze(1)
    return torch.cat([self_idx, sampled], dim=1).to(torch.long).cpu()


def calculate_knn(
    model,
    X: torch.Tensor,
    strategy: str = "knn",
    multigroup: bool = False,
    groupsX: torch.Tensor | None = None,
    groupsZ: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute KNN indices for a local-GP prior model.

    Extracts ``Z`` and ``K`` from ``model``. Dispatches to FAISS (``'knn'``)
    or kernel-weighted sampling (``'probabilistic'``).

    Returns ``(N, K+1)`` with column 0 = self for both strategies. Callers
    slice ``[:, :-1]`` (self-inclusive) for the inference path and
    ``[:, 1:]`` (self-exclusive) for the KL path.
    """
    Z = model.Z
    K = model.K

    if strategy == "knn":
        return _faiss_knn(X, Z, K)
    elif strategy == "probabilistic":
        return _probabilistic_knn(X, Z, K, model.kernel, multigroup, groupsX, groupsZ)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'knn' or 'probabilistic'.")
