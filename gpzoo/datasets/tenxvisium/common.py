"""Shared utilities for 10x Visium single-model training scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from gpzoo.utilities import rescale_spatial_coords, scanpy_sizefactors

from .config import (
    DATA_DIR,
    DEVICE,
    H5AD_PATTERN,
    OUTPUT_DIR,
    REGION_COLUMN,
    SPATIAL_SCALE,
)


def _prepare_adata(adata: sc.AnnData) -> sc.AnnData:
    """Return a copy of the AnnData without additional QC filtering."""

    return adata.copy()
def load_visium_with_regions(
    data_dir: Path = DATA_DIR,
    h5ad_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Load a single 10x Visium slide; expects obs['Region'] annotations."""

    data_dir = Path(data_dir)
    if h5ad_path is not None:
        files = [Path(h5ad_path)]
    else:
        files = sorted(data_dir.glob(H5AD_PATTERN))
    if not files:
        raise FileNotFoundError(f"No Visium h5ad files found in {data_dir}")
    if len(files) > 1 and h5ad_path is None:
        raise ValueError("Multiple Visium files found; specify h5ad_path to load one slide.")

    path = files[0]
    adata = sc.read_h5ad(path)
    if REGION_COLUMN not in adata.obs:
        raise RuntimeError(f"{path} missing obs['{REGION_COLUMN}']")

    mask = ~adata.obs[REGION_COLUMN].isna()
    adata = adata[mask].copy()
    if adata.n_obs == 0:
        raise RuntimeError(f"{path} is empty after filtering.")

    adata = _prepare_adata(adata)
    genes = adata.var_names

    X_np = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X_np = rescale_spatial_coords(X_np) * SPATIAL_SCALE

    Y_matrix = adata[:, genes].X
    if hasattr(Y_matrix, "toarray"):
        Y_matrix = Y_matrix.toarray()
    Y_np = np.asarray(Y_matrix, dtype=np.float32).T

    V_np = scanpy_sizefactors(Y_np.T).reshape(-1)
    regions = adata.obs[REGION_COLUMN]
    if not hasattr(regions.values, "codes"):
        regions = regions.astype("category")
    groups_np = regions.values.codes.astype(int)
    n_groups = int(groups_np.max().item()) + 1

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
    start_step: int = 0,
    save_optimizer_state: bool = False,
    **train_kwargs,
) -> Dict[str, object]:
    """Execute training and persist weights/losses, mirroring benchmark logs."""

    train_steps = int(train_kwargs.pop("steps", 0))
    train_steps = max(train_steps, 0)

    _cuda_sync()
    _reset_peak_mem()
    t0 = time.perf_counter()

    losses, means, scales, idxs = train_fn(
        optimizer=optimizer, model=model, start_step=start_step, steps=train_steps, **train_kwargs
    )

    _cuda_sync()
    wall = time.perf_counter() - t0
    mem = _peak_mem_bytes()

    torch.save(model.state_dict(), save_path)
    if save_optimizer_state and optimizer is not None:
        opt_path = save_path.with_suffix(".opt.pth")
        torch.save(optimizer.state_dict(), opt_path)
    loss_artifacts = _save_losses(losses, save_path)

    result = {
        "label": label,
        "steps": train_steps,
        "start_step": start_step,
        "wall_time_sec": wall,
        "sec_per_step": wall / max(train_steps, 1),
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
    "load_visium_with_regions",
    "run_training",
]
