#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================
# Train + Save + Benchmark
# =========================
import os, time, json
from datetime import datetime

import numpy as np
import pandas as pd


import torch
from torch import optim

# ---- Project / third-party you already use ----
import scanpy as sc
import anndata as ad

# gpzoo bits you use
import gpzoo
from gpzoo.kernels import batched_Matern32
from gpzoo.utilities import scanpy_sizefactors
from gpzoo.model_utilities import (
    build_nsf_svgp,
    build_nsf_vnngp,
)
from gpzoo.training_utilities import (
    train_svgp_batched_with_tracking,
    train_vnngp_batched_with_tracking,
)

# ------------------
# Config
# ------------------
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/nsf_benchmarks"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 123
STEPS = 2000
X_BATCH = 1296
Y_BATCH = 200
L_FACTORS = 4
LR = 2e-2
K_LIST = [5, 10, 25, 50, 100]

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------
# Data
# ------------------
adata = sc.read_h5ad("/gladstone/engelhardt/home/lchumpitaz/gitclones/nsf-paper/simulations/bm_sp/data/S1.h5ad")
X_np = adata.obsm["spatial"]
Y_np = adata.X.T
V_np = scanpy_sizefactors(Y_np.T)

X = torch.tensor(X_np, dtype=torch.float)
Y = torch.tensor(Y_np, dtype=torch.float)
V_INIT = torch.squeeze(torch.ones_like(torch.tensor(V_np, dtype=torch.float)))

def model_grads(model):
    model.prior.kernel.lengthscale.requires_grad = False
    model.prior.kernel.sigma.requires_grad = False
    model.prior.Z.requires_grad = False
    model.prior.mu.requires_grad = True
    model.prior.Lu.requires_grad = True
    model.W.requires_grad = True
    model.V.requires_grad = False

# ------------------
# Benchmark helpers
# ------------------


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

def _peak_mem_bytes():
    if torch.cuda.is_available():
        return {
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    return {"max_memory_allocated": 0, "max_memory_reserved": 0}

def _save_losses(losses, weights_path):
    """
    Save loss curves next to the weights file.
    - CSV: one row per step with columns [step, loss]
    - NPY: raw float array for fast reload
    """
    base, _ = os.path.splitext(weights_path)
    csv_path = base + "_losses.csv"
    npy_path = base + "_losses.npy"

    # CSV with explicit step index
    df = pd.DataFrame({"step": np.arange(len(losses), dtype=int), "loss": np.asarray(losses, dtype=float)})
    df.to_csv(csv_path, index=False)

    # Raw array
    np.save(npy_path, np.asarray(losses, dtype=float))

    return {"losses_csv": csv_path, "losses_npy": npy_path}

def benchmark_and_save(label, model, optimizer, train_fn, save_path):


    _cuda_sync()
    _reset_peak_mem()
    t0 = time.perf_counter()

    # train and collect full loss history
    losses, means, scales, idxs = train_fn(
        optimizer, model, X, Y, device,
        steps=STEPS, x_batch_size=X_BATCH, y_batch_size=Y_BATCH
    )

    _cuda_sync()
    wall = time.perf_counter() - t0
    mem = _peak_mem_bytes()

    # Save model weights
    torch.save(model.state_dict(), save_path)

    # Save the losses right next to the weights
    loss_artifacts = _save_losses(losses, save_path)

    return {
        "label": label,
        "steps": STEPS,
        "wall_time_sec": wall,
        "sec_per_step": wall / STEPS,
        "final_loss": float(losses[-1]) if len(losses) else float("nan"),
        "device": str(device),
        **mem,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights_path": save_path,
        **loss_artifacts,   # <- paths to CSV/NPY
    }

# ------------------
# Runs
# ------------------
def main():
    results = []
    lr = LR
    # Full (SVGP) NSF
    nsf_model = build_nsf_svgp(
        X=X,
        Y=Y,
        V=V_INIT,
        L=L_FACTORS,
        kernel=batched_Matern32(sigma=1.0, lengthscale=0.3),
        jitter=1e-5,
        inducing_points=X,
        device=device,
        seed=12,
    )
    model_grads(nsf_model)
    nsf_optimizer = optim.Adam(filter(lambda p: p.requires_grad, nsf_model.parameters()), lr=lr)

    out_path = os.path.join(OUTPUT_DIR, "toy_nsf.pth")
    res = benchmark_and_save(
        label="SVGP_Matern32",
        model=nsf_model,
        optimizer=nsf_optimizer,
        train_fn=train_svgp_batched_with_tracking,
        save_path=out_path
    )
    results.append(res)

    # VNNGP variants
    for K in K_LIST:
        mdl = build_nsf_vnngp(
            X=X,
            Y=Y,
            V=V_INIT,
            L=L_FACTORS,
            kernel=batched_Matern32(sigma=1.0, lengthscale=0.3),
            jitter=1e-5,
            K=K,
            lu_reference_points=X[::10],
            subset_step=10,
            device=device,
            seed=0,
            precompute_knn=True,
        )
        model_grads(mdl)
        opt = optim.Adam(filter(lambda p: p.requires_grad, mdl.parameters()), lr=lr)

        out_path = os.path.join(OUTPUT_DIR, f"toy_nsf_k={K}.pth")
        res = benchmark_and_save(
            label=f"VNNGP_Matern32_K={K}",
            model=mdl,
            optimizer=opt,
            train_fn=train_vnngp_batched_with_tracking,
            save_path=out_path
        )
        results.append(res)

    # Persist a CSV + JSON summary
    import pandas as pd
    df = pd.DataFrame(results)
    csv_name = os.path.join(OUTPUT_DIR, f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_name, index=False)

    json_name = os.path.join(OUTPUT_DIR, f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_name, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved weights and logs to:", OUTPUT_DIR)
    print(df)

if __name__ == "__main__":
    main()
