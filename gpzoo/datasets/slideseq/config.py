"""Shared configuration values for Slideseq training scripts."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "models" / "slideseq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR = ROOT / "models" / "mggp_slideseq_benchmarks"



VNNGP_K = 50
VNNGP_E_SAMPLES = 1

SEED = 67
STEPS = 2000
X_BATCH = 7000
Y_BATCH = 1000
L_FACTORS = 10
LR = 1e-3
LR_LENGTHSCALE = 1e-4
LR_SCALE = 1e-3
LR_MEAN = 1e-1
LR_LOADING = 1e-4


SPATIAL_SCALE = 50.0
LENGTHSCALE = 4.0
LENGTHSCALE_TRAIN_AFTER = STEPS
SCALE_TRAIN_AFTER = STEPS//10
SCALE_MULTIPLIER = 1e-1
JITTER = 1e-5
SVGP_INDUCING = 3000
INDUCING_ALLOCATION = "equal"
GROUP_DIFF_PARAM = 10.0
IMAGE_LOG_EVERY = 100

# Loadings W transformation mode: 'softplus', 'exp', 'exp_sum', or 'projected'
LOADINGS_MODE = 'projected'

# Whether to use scheduler for learning rate adjustment
USE_SCHEDULER = False

SVGP_MGGP_CHECKPOINT = OUTPUT_DIR / "slideseq_mggp_svgp.pth"
VNNGP_MGGP_CHECKPOINT = OUTPUT_DIR / "slideseq_mggp_vnngp_k=50.pth"
SVGP_CHECKPOINT = OUTPUT_DIR / "slideseq_svgp.pth"
VNNGP_CHECKPOINT = OUTPUT_DIR / f"slideseq_vnngp_k={VNNGP_K}.pth"

# LCGP-specific configuration
LCGP_K = VNNGP_K  # Use same K as VNNGP
LCGP_RANK = VNNGP_K + 5  # Default rank
LCGP_DIAG_MODE = "softplus"
LCGP_CHECKPOINT = OUTPUT_DIR / f"slideseq_lcgp_k={VNNGP_K}.pth"
LCGP_MGGP_CHECKPOINT = OUTPUT_DIR / f"slideseq_mggp_lcgp_k={VNNGP_K}.pth"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
