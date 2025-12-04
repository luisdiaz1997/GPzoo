#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark MGGP NSF models on SlideseqV2.
Mirrors notebooks/mggp_knn/benchmark.py but loads the real Slideseq data,
keeps cluster labels as groups, and compares full MGGP-SVGP vs MGGP-VNNGP.
"""

import os
import gc
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd

import torch
from torch import optim

from gpzoo.datasets.slideseq.common import load_slideseq_with_groups

from gpzoo.models import build_model
from gpzoo.training_utilities import (
    freeze_kernel_and_inputs,
    train_mggp_svgp_with_tracking,
    train_mggp_vnngp_with_tracking,
)


# ------------------
# Config (adapt to taste)
# ------------------
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/mggp_slideseq_benchmarks")
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 123
STEPS = 15000
X_BATCH = 6000
Y_BATCH = 1000
L_FACTORS = 12
LR = 1e-2
K_LIST = [5, 10, 25, 50, 75]
SPATIAL_SCALE = 50.0
LENGTHSCALE = 20.0
JITTER = 1e-5
SVGP_INDUCING = 4000
INDUCING_ALLOCATION = "equal"
SVGP_CHECKPOINT = None  # set to a .pth file to resume the SVGP run
# Optional mapping of K -> checkpoint path to resume selected VNNGP runs.
VNNGP_CHECKPOINTS = {}

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------
# Data (same prep as Slideseq notebooks)
# ------------------
X, Y, GROUPS_X, N_GROUPS, V_PARAM_INIT = load_slideseq_with_groups()
D_OBS, N_CELLS = Y.shape


# ------------------
# Benchmark bookkeeping (same helpers as benchmark.py)
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
    base, _ = os.path.splitext(weights_path)
    csv_path = base + "_losses.csv"
    npy_path = base + "_losses.npy"

    df = pd.DataFrame(
        {"step": np.arange(len(losses), dtype=int), "loss": np.asarray(losses, dtype=float)}
    )
    df.to_csv(csv_path, index=False)
    np.save(npy_path, np.asarray(losses, dtype=float))
    return {"losses_csv": csv_path, "losses_npy": npy_path}


def benchmark_and_save(label, model, optimizer, train_fn, save_path, **train_kwargs):
    _cuda_sync()
    _reset_peak_mem()
    t0 = time.perf_counter()

    losses, means, scales, idxs = train_fn(
        optimizer=optimizer,
        model=model,
        X=X,
        groupsX=GROUPS_X,
        y=Y,
        device=device,
        steps=STEPS,
        x_batch_size=X_BATCH,
        y_batch_size=Y_BATCH,
        **train_kwargs,
    )

    _cuda_sync()
    wall = time.perf_counter() - t0
    mem = _peak_mem_bytes()

    torch.save(model.state_dict(), save_path)
    loss_artifacts = _save_losses(losses, save_path)

    return {
        "label": label,
        "steps": STEPS,
        "wall_time_sec": wall,
        "sec_per_step": wall / max(STEPS, 1),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "device": str(device),
        **mem,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights_path": save_path,
        **loss_artifacts,
    }


# ------------------
# Entry point
# ------------------
def main():
    results = []

    target = min(SVGP_INDUCING, N_CELLS)
    svg_model, _ = build_model(
        "slideseq/svgp_mggp_nsf",
        checkpoint_path=SVGP_CHECKPOINT,
        X=X,
        groupsX=GROUPS_X,
        Y=Y,
        V=V_PARAM_INIT,
        L=L_FACTORS,
        lengthscale=LENGTHSCALE,
        kernel_kwargs={
            "sigma": 1.0,
            "group_diff_param": 10.0,
        },
        jitter=JITTER,
        num_inducing=target,
        inducing_allocation=INDUCING_ALLOCATION,
        inducing_seed=SEED,
        device=device,
        seed=SEED,
    )
    freeze_kernel_and_inputs(svg_model)
    svg_optimizer = optim.Adam(filter(lambda p: p.requires_grad, svg_model.parameters()), lr=LR)
    svg_path = os.path.join(OUTPUT_DIR, "slideseq_mggp_svgp.pth")
    res = benchmark_and_save(
        label="MGGP_SVGP_RBF",
        model=svg_model,
        optimizer=svg_optimizer,
        train_fn=train_mggp_svgp_with_tracking,
        save_path=svg_path,
    )
    results.append(res)
    svg_model.to("cpu")
    del svg_model
    torch.cuda.empty_cache()

    for K in K_LIST:
        vnngp_model, _ = build_model(
            "slideseq/vnngp_mggp_nsf",
            checkpoint_path=VNNGP_CHECKPOINTS.get(K),
            X=X,
            groupsX=GROUPS_X,
            Y=Y,
            V=V_PARAM_INIT,
            L=L_FACTORS,
            lengthscale=LENGTHSCALE,
            kernel_kwargs={
                "sigma": 1.0,
                "group_diff_param": 10.0,
            },
            jitter=JITTER,
            K=K,
            subset_step=10,
            device=device,
            seed=SEED,
            precompute_knn=True,
        )
        freeze_kernel_and_inputs(vnngp_model)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, vnngp_model.parameters()), lr=LR)
        weights_path = os.path.join(OUTPUT_DIR, f"slideseq_mggp_vnngp_k={K}.pth")
        res = benchmark_and_save(
            label=f"MGGP_VNNGP_RBF_K={K}",
            model=vnngp_model,
            optimizer=optimizer,
            train_fn=train_mggp_vnngp_with_tracking,
            save_path=weights_path,
            e_samples=1,
        )
        results.append(res)
        vnngp_model.to("cpu")
        del vnngp_model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"mggp_slideseq_benchmark_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(OUTPUT_DIR, f"mggp_slideseq_benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved weights and logs to:", OUTPUT_DIR)
    print(df)


if __name__ == "__main__":
    main()
