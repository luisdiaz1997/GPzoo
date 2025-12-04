"""Shared configuration values for liver MERFISH training scripts."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = Path("/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_merfish.h5ad")

OUTPUT_DIR = ROOT / "models" / "liver"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR = ROOT / "models" / "liver_benchmarks"

CELL_TYPE_COLUMN = "Cell_Type"

SEED = 123
STEPS = 100  # From notebook: steps = 100
X_BATCH = 13000  # From notebook: x_batch_size=13000
Y_BATCH = 317  # D from notebook (n_vars)
L_FACTORS = 12
LR = 2e-4  # From notebook: lr=2e-4
LR_LENGTHSCALE = 2e-4
SPATIAL_SCALE = 50.0  # Rescale to [-1, 1] then multiply by 50
LENGTHSCALE = 10.0  # Appropriate for rescaled coordinates
LENGTHSCALE_TRAIN_AFTER = STEPS // 2
IMAGE_LOG_EVERY = 10  # Log every 10 steps
JITTER = 1e-5  # From notebook: jitter=1e-5
SVGP_INDUCING = 4000
INDUCING_ALLOCATION = "equal"

SVGP_MGGP_CHECKPOINT = OUTPUT_DIR / "liver_healthy_mggp_svgp.pth"
VNNGP_MGGP_CHECKPOINT = OUTPUT_DIR / "liver_healthy_mggp_vnngp_k=50.pth"
SVGP_CHECKPOINT = OUTPUT_DIR / "liver_healthy_svgp.pth"
VNNGP_CHECKPOINT = OUTPUT_DIR / "liver_healthy_vnngp_k=50.pth"

VNNGP_K = 50  # From notebook: K=50
VNNGP_E_SAMPLES = 1
GROUP_DIFF_PARAM = 10.0  # From notebook: group_diff_param=10.0

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # From notebook: device cuda:1
