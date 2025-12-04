"""Shared configuration values for 10x Visium training scripts."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path("/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data")

OUTPUT_DIR = ROOT / "models" / "tenxvisium"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR = ROOT / "models" / "mggp_tenxvisium_benchmarks"

H5AD_PATTERN = "*.h5ad"
REGION_COLUMN = "Region"

SEED = 123
STEPS = 8000
X_BATCH = 4221
Y_BATCH = 10000
L_FACTORS = 12
LR = 2e-3
LR_LENGTHSCALE = 2e-4
SPATIAL_SCALE = 50.0
LENGTHSCALE = 10.0
LENGTHSCALE_TRAIN_AFTER = STEPS // 2
IMAGE_LOG_EVERY = 200
JITTER = 1e-5
SVGP_INDUCING = 4000
INDUCING_ALLOCATION = "equal"

SVGP_MGGP_CHECKPOINT = OUTPUT_DIR / "tenxvisium_mggp_svgp.pth"
VNNGP_MGGP_CHECKPOINT = BENCHMARK_DIR / "tenxvisium_mggp_vnngp_k=50.pth"
SVGP_CHECKPOINT = OUTPUT_DIR / "tenxvisium_svgp.pth"
VNNGP_CHECKPOINT = OUTPUT_DIR / "tenxvisium_vnngp_k=50.pth"

VNNGP_K = 50
VNNGP_E_SAMPLES = 1
GROUP_DIFF_PARAM = 10.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
