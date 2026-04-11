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
    mem_gb: float = 50.0,
) -> torch.Tensor:
    """Sample K neighbors per row weighted by kernel(X, Z), prepend self.

    Returns ``(N, K+1)`` long tensor with column 0 = self index.

    For large N (e.g. LCGP where M=N), the full (N, M) kernel matrix can
    exceed GPU/CPU memory. ``mem_gb`` sets a budget; rows are processed in
    chunks of size ``B = floor(mem_gb * 2.5e8 / M)`` so peak memory stays
    within budget.
    """
    import math

    N = X.shape[0]
    M = Z.shape[0]
    K = min(K, M - 1)

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

    # Determine chunk size based on memory budget.
    B = int(math.floor(mem_gb * 2.5e8 / M))
    B = max(1, min(B, N))

    def _process_chunk(x_chunk, start, end):
        gX = groupsX[start:end] if groupsX is not None else None
        with torch.no_grad():
            if multigroup:
                w = kernel(x_chunk, Z, gX, groupsZ)
            else:
                w = kernel(x_chunk, Z)
        w = w.squeeze()  # (chunk_size, M)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        # Zero out self-connections when X == Z (training time)
        if N == M:
            rows = torch.arange(end - start)
            cols = torch.arange(start, end)
            w[rows, cols] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return _sample_without_replacement(w, K)  # (chunk_size, K)

    if B >= N:
        # Single-shot path — no chunking overhead for small datasets.
        sampled = _process_chunk(X, 0, N)
    else:
        chunks = []
        for start in range(0, N, B):
            end = min(start + B, N)
            chunks.append(_process_chunk(X[start:end], start, end))
        sampled = torch.cat(chunks, dim=0)  # (N, K)

    kernel.to(kernel_was)

    self_idx = torch.arange(N, device=sampled.device).unsqueeze(1)
    return torch.cat([self_idx, sampled], dim=1).to(torch.long).cpu()


def calculate_knn(
    model,
    X: torch.Tensor,
    strategy: str = "knn",
    multigroup: bool = False,
    groupsX: torch.Tensor | None = None,
    groupsZ: torch.Tensor | None = None,
    mem_gb: float = 50.0,
) -> torch.Tensor:
    """Compute KNN indices for a local-GP prior model.

    Extracts ``Z`` and ``K`` from ``model``. Dispatches to FAISS (``'knn'``)
    or kernel-weighted sampling (``'probabilistic'``).

    Returns ``(N, K+1)`` with column 0 = self for both strategies. Callers
    slice ``[:, :-1]`` (self-inclusive) for the inference path and
    ``[:, 1:]`` (self-exclusive) for the KL path.

    ``mem_gb`` controls the memory budget for the probabilistic path (default
    50 GB). The kernel matrix is evaluated in row-chunks so peak memory stays
    within budget. Has no effect on the FAISS path.
    """
    Z = model.Z
    K = model.K

    if strategy == "knn":
        return _faiss_knn(X, Z, K)
    elif strategy == "probabilistic":
        return _probabilistic_knn(
            X, Z, K, model.kernel, multigroup, groupsX, groupsZ, mem_gb=mem_gb
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'knn' or 'probabilistic'.")
