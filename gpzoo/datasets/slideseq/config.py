"""Shared configuration values for Slideseq training scripts."""

from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "models" / "slideseq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR = ROOT / "models" / "mggp_slideseq_benchmarks"

SEED = 123
STEPS = 2000
X_BATCH = 7000
Y_BATCH = 1000
L_FACTORS = 10
LR = 1.0
LR_LENGTHSCALE = 1e-4
LR_SCALE = 1e-2
SPATIAL_SCALE = 50.0
LENGTHSCALE = 4.0
LENGTHSCALE_TRAIN_AFTER = STEPS
SCALE_TRAIN_AFTER = STEPS//2
SCALE_MULTIPLIER = 1e-6
JITTER = 1e-5
SVGP_INDUCING = 3000
INDUCING_ALLOCATION = "equal"
GROUP_DIFF_PARAM = 10.0
IMAGE_LOG_EVERY = 100

# Loadings W transformation mode: 'softplus', 'exp', 'exp_sum', or 'projected'
LOADINGS_MODE = 'projected'

SVGP_MGGP_CHECKPOINT = OUTPUT_DIR / "slideseq_mggp_svgp.pth"
VNNGP_MGGP_CHECKPOINT = OUTPUT_DIR / "slideseq_mggp_vnngp_k=50.pth"
SVGP_CHECKPOINT = OUTPUT_DIR / "slideseq_svgp.pth"
VNNGP_CHECKPOINT = OUTPUT_DIR / "slideseq_vnngp_k=50.pth"

VNNGP_K = 50
VNNGP_E_SAMPLES = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
