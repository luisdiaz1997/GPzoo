"""Shared utilities for liver MERFISH single-model training scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

from gpzoo.utilities import rescale_spatial_coords, scanpy_sizefactors

from .config import (
    CELL_TYPE_COLUMN,
    DATA_PATH,
    DEVICE,
    OUTPUT_DIR,
    SPATIAL_SCALE,
)


def load_liver_with_groups(
    data_path: Path = DATA_PATH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Load the liver MERFISH dataset and return tensors used for training."""

    adata = ad.read_h5ad(data_path)

    if CELL_TYPE_COLUMN not in adata.obs:
        raise RuntimeError(f"{data_path} missing obs['{CELL_TYPE_COLUMN}']")

    # Extract spatial coordinates
    X_np = np.asarray(adata.obsm["X_spatial"], dtype=np.float32)
    X_np = rescale_spatial_coords(X_np) * SPATIAL_SCALE

    # Extract gene expression matrix
    Y_matrix = adata.raw.X
    if hasattr(Y_matrix, "toarray"):
        Y_matrix = Y_matrix.toarray()
    Y_np = np.asarray(Y_matrix, dtype=np.float32).T

    # Compute size factors
    V_np = scanpy_sizefactors(Y_np.T).reshape(-1)

    # Extract cell type groups
    cell_types = adata.obs[CELL_TYPE_COLUMN].astype("category")
    groups_np = cell_types.cat.codes.to_numpy()
    n_groups = len(cell_types.cat.categories)

    # Convert to tensors
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
    "load_liver_with_groups",
    "run_training",
]
