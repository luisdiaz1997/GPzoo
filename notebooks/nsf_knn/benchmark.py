#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =========================
# Train + Save + Benchmark
# =========================
import os, gc, time, json
from datetime import datetime

import numpy as np
import pandas as pd


import torch
from torch import nn, optim

# ---- Project / third-party you already use ----
import scanpy as sc
import anndata as ad

# gpzoo bits you use
import gpzoo
from gpzoo.kernels import batched_Matern32
from gpzoo.gp import SVGP, VNNGP
from gpzoo.likelihoods import NSF2
from gpzoo.utilities import (
    scanpy_sizefactors, init_Lu_nsf
)

# ------------------
# Config
# ------------------
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 123
STEPS = 100
X_BATCH = 1296
Y_BATCH = 200
L_FACTORS = 4
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

# ------------------
# Model builders (same behavior as in your notebook)
# ------------------
def build_nsf_model(Y, L=4):
    D, N = Y.shape
    M = N
    kernel = batched_Matern32(sigma=1.0, lengthscale=0.3)
    gp = SVGP(kernel, M=M, jitter=1e-5)

    torch.manual_seed(12)
    Lu_init = init_Lu_nsf(kernel, X, X, K=M, niter=10)
    Lu = Lu_init.expand(L, -1, -1)
    gp.Lu = nn.Parameter(Lu.clone().detach())

    mu = torch.squeeze(Lu_init @ torch.randn(L, M, 1))
    gp.mu = nn.Parameter(mu)

    gp.Z = nn.Parameter(torch.tensor(X), requires_grad=False)

    model = NSF2(gp, Y, L=L)
    W = torch.randn(D, L)
    model.W = nn.Parameter(W)
    model.V = nn.Parameter(torch.squeeze(torch.ones_like(torch.tensor(V_np, dtype=torch.float))))
    model.to(device)
    return model

def build_nn_nsf_model(Y, L=4, K=50):
    D, N = Y.shape
    M = N
    kernel = batched_Matern32(sigma=1.0, lengthscale=0.3)
    gp = VNNGP(kernel, M=M, jitter=1e-5, K=K)

    torch.manual_seed(0)
    Lu_init = init_Lu_nsf(kernel, X, X[::10], K)
    Lu = Lu_init.expand(L, -1, -1)
    gp.Lu = nn.Parameter(Lu.clone().detach())
    mu = torch.squeeze(Lu_init @ torch.randn(L, K, 1))
    gp.mu = nn.Parameter(mu)

    gp.Z = nn.Parameter(torch.tensor(X), requires_grad=True)

    model = NSF2(gp, Y, L=L)
    model.prior.K = K

    W = torch.randn(D, L)
    model.W = nn.Parameter(W)
    model.V = nn.Parameter(torch.squeeze(torch.ones_like(torch.tensor(V_np, dtype=torch.float))))

    knn_idx = model.prior.calculate_knn(X)[:, :-1]
    model.prior.knn_idx = knn_idx
    knn_idz = model.prior.calculate_knn(model.prior.Z)[:, 1:]
    model.prior.knn_idz = knn_idz

    model.to(device)
    return model

def model_grads(model):
    model.prior.kernel.lengthscale.requires_grad = False
    model.prior.kernel.sigma.requires_grad = False
    model.prior.Z.requires_grad = False
    model.prior.mu.requires_grad = True
    model.prior.Lu.requires_grad = True
    model.W.requires_grad = True
    model.V.requires_grad = False

# ------------------
# Training wrappers (call your existing functions from the notebook)
# ------------------
# Re-implement minimal versions that match your signatures to avoid importing a .py that executes top-level code.
from tqdm import trange

def train_svgp_batched_with_tracking(optimizer, model, X, y, device,
                                     steps=200, x_batch_size=1000, y_batch_size=1000, **_):
    losses, means, scales, idxs = [], [], [], []
    N = len(X); J = len(y)
    for it in trange(steps, desc="SVGP"):
        idx = torch.multinomial(torch.ones(N), num_samples=x_batch_size, replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=y_batch_size, replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)

        optimizer.zero_grad()
        # identical calls as your version:
        pY, qF, qZ, _ = model.forward_batched_train(Xb.squeeze(), idx=idx, idy=idy)

        logpY = yb * torch.log(pY.rate) - pY.rate
        L1 = (logpY).mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence(qZ))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (it % 10) == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return losses, means, scales, idxs

def train_vnngp_batched_with_tracking(optimizer, model, X, y, device,
                                      steps=200, x_batch_size=1000, y_batch_size=1000, **_):
    losses, means, scales, idxs = [], [], [], []
    N = len(X); J = len(y)
    full_knn_idx = model.prior.calculate_knn(X).clone()[:, :-1]
    for it in trange(steps, desc="VNNGP"):
        idx = torch.multinomial(torch.ones(N), num_samples=x_batch_size, replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=y_batch_size, replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)
        knn_idx = (full_knn_idx[idx]).to(device)
        model.prior.knn_idx = knn_idx

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched(Xb.squeeze(), idx=idx, idy=idy)

        logpY = yb * torch.log(pY.rate) - pY.rate
        L1 = (logpY).mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence_full(qZ, idx=idx))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (it % 10) == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb, knn_idx
        model.prior.knn_idx = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return losses, means, scales, idxs

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
    lr = 1e-2
    # Full (SVGP) NSF
    nsf_model = build_nsf_model(Y, L=L_FACTORS)
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
        mdl = build_nn_nsf_model(Y, L=L_FACTORS, K=K)
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
