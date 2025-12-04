"""Shared utilities for Slideseq single-model training scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch

from gpzoo.utilities import rescale_spatial_coords, scanpy_sizefactors

from .config import (
    DEVICE,
    OUTPUT_DIR,
    SPATIAL_SCALE,
)




def load_slideseq_with_groups() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Load the SlideseqV2 dataset and return tensors used for training."""

    adata = sq.datasets.slideseqv2()
    adata = adata.raw.to_adata()

    adata.var["mt"] = adata.var_names.str.lower().str.startswith("mt-")
    adata.var["MT"] = adata.var["mt"]

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20].copy()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=10)

    gene_mask = ~adata.var["MT"].values

    X_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X_np = rescale_spatial_coords(X_np) * SPATIAL_SCALE
    Y_matrix = adata[:, gene_mask].X
    if hasattr(Y_matrix, "toarray"):
        Y_matrix = Y_matrix.toarray()
    Y_np = np.asarray(Y_matrix, dtype=np.float32).T
    V_np = scanpy_sizefactors(Y_np.T).reshape(-1)

    if "cluster" not in adata.obs:
        raise RuntimeError("Slideseq AnnData is missing obs['cluster'] needed for MGGP groups.")

    clusters = adata.obs["cluster"].astype("category")
    cluster_codes = clusters.cat.codes.to_numpy()
    groups_np = cluster_codes
    n_groups = len(clusters.cat.categories)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    Y_t = torch.tensor(Y_np, dtype=torch.float32)
    groups_t = torch.tensor(groups_np, dtype=torch.long)
    V_t = torch.ones_like(torch.tensor(V_np, dtype=torch.float32))

    return X_t, Y_t, groups_t, n_groups, V_t


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)


def _peak_mem_bytes() -> Dict[str, int]:
    if torch.cuda.is_available():
        return {
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(DEVICE)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(DEVICE)),
        }
    return {"max_memory_allocated": 0, "max_memory_reserved": 0}


def _save_losses(losses, weights_path: Path) -> Dict[str, str]:
    base = Path(weights_path)
    csv_path = base.with_name(base.stem + "_losses.csv")
    npy_path = base.with_name(base.stem + "_losses.npy")

    df = pd.DataFrame(
        {"step": np.arange(len(losses), dtype=int), "loss": np.asarray(losses, dtype=float)}
    )
    df.to_csv(csv_path, index=False)
    np.save(npy_path, np.asarray(losses, dtype=float))
    return {"losses_csv": str(csv_path), "losses_npy": str(npy_path)}


def run_training(
    label: str,
    model,
    optimizer,
    train_fn,
    *,
    save_path: Path,
    **train_kwargs,
) -> Dict[str, object]:
    """Execute training and persist weights/losses, mirroring benchmark logs."""

    _cuda_sync()
    _reset_peak_mem()
    t0 = time.perf_counter()

    losses, means, scales, idxs = train_fn(optimizer=optimizer, model=model, **train_kwargs)

    _cuda_sync()
    wall = time.perf_counter() - t0
    mem = _peak_mem_bytes()

    torch.save(model.state_dict(), save_path)
    loss_artifacts = _save_losses(losses, save_path)

    result = {
        "label": label,
        "steps": train_kwargs.get("steps"),
        "wall_time_sec": wall,
        "sec_per_step": wall / max(train_kwargs.get("steps", 1), 1),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "device": str(DEVICE),
        **mem,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "weights_path": str(save_path),
        **loss_artifacts,
    }

    artifacts_path = save_path.with_suffix(".json")
    with open(artifacts_path, "w") as fh:
        json.dump(result, fh, indent=2)
    result["artifacts_json"] = str(artifacts_path)
    return result


__all__ = [
    "DEVICE",
    "OUTPUT_DIR",
    "load_slideseq_with_groups",
    "run_training",
]
