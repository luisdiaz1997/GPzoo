# GPzoo Development Notes

## Project Overview

GPzoo is a pip-installable PyTorch library of Gaussian Process building blocks for spatial transcriptomics. It is used as a **dependency by two downstream libraries**:

- **[PNMF](https://github.com/luisdiaz1997/Probabilistic-NMF)** — Probabilistic NMF with GP spatial priors
- **[Spatial-Factorization](https://github.com/luisdiaz1997/Spatial-Factorization)** — End-to-end spatial transcriptomics pipeline

## Installation

```bash
pip install git+https://github.com/luisdiaz1997/GPzoo.git
```

Or in editable mode for development:
```bash
pip install -e .
```

All metadata and dependencies live in `pyproject.toml`. `setup.py` is a minimal stub for backwards compatibility only.

## Package Structure

Only the top-level files in `gpzoo/` are part of the installed package (subfolders are not included):

```
gpzoo/
├── __init__.py          # Public API: SVGP_NSF, VNNGP_NSF, MGGP_SVGP_NSF, MGGP_VNNGP_NSF, build_model
├── gp.py                # Core GP classes: SVGP, VNNGP, LCGP, MGGP_SVGP, MGGP_VNNGP, MGGP_LCGP, GaussianPrior
├── kernels.py           # Kernel functions: batched_Matern32, batched_MGGP_Matern32, batched_MGGP_RBF, etc.
├── likelihoods.py       # Likelihood models: NSF2, GaussianLikelihood
├── model_utilities.py   # Utilities: kmeans_inducing_points(), mggp_kmeans_inducing_points()
├── modules.py           # Parameter classes: PositiveParameter, CholeskyParameter
├── natural_gradient.py  # NaturalGradientDescent optimizer
├── training_utilities.py# Training helpers
└── utilities.py         # Math utilities: add_jitter, svgp_forward, whitened_KL, calculate_knn, etc.
```

## Key Components Consumed by PNMF

| Component | File | Used for |
|-----------|------|----------|
| `PositiveParameter`, `CholeskyParameter` | `modules.py` | Constrained parameters |
| `GaussianPrior` | `gp.py` | Non-spatial variational prior |
| `SVGP`, `MGGP_SVGP` | `gp.py` | Sparse GP prior (N < 10k) |
| `LCGP`, `MGGP_LCGP` | `gp.py` | Locally conditioned GP prior (N > 10k) |
| `batched_Matern32`, `batched_MGGP_Matern32` | `kernels.py` | Kernel functions |
| `kmeans_inducing_points()`, `mggp_kmeans_inducing_points()` | `model_utilities.py` | Inducing point selection |

## GP Model Hierarchy

```
SVGP  (sparse variational GP, M << N, full Cholesky)
  └── MGGP_SVGP  (multi-group variant, batched_MGGP_Matern32 kernel)
  └── LCGP       (M = N, VNNGP-style S = Lu @ Lu.T, locally conditioned KL)
        └── MGGP_LCGP  (multi-group LCGP)
VNNGP  (nearest-neighbor GP, separate from SVGP hierarchy)
  └── MGGP_VNNGP
```

## LCGP Parameterization

LCGP uses the **same parameterization as VNNGP**: `Lu` is a raw `nn.Parameter(L, M, K)` and `S = Lu @ Lu.T`. Key overrides vs SVGP:
- `apply_constraints()` — returns `(mu, Lu)` directly (no Cholesky constraint)
- `reshape_parameters()` — indexes Lu by KNN, computes `S = Lu_knn @ Lu_knn.T`, Cholesky-factors them
- `kl_divergence_full()` — locally conditioned KL using VNNGP's whitened formulation
- KNN convention: training uses `knn_idz = calculate_knn(Z)[:, 1:]`, inference uses `knn_idx = calculate_knn(X)[:, :-1]`

## Dependencies

- `torch>=1.9.0`
- `numpy>=1.19.0`
- `scikit-learn>=1.0.0` (KMeans, NMF, KNeighbors)
- `tqdm>=4.0.0`
- `faiss-cpu>=1.7.0` (KNN computation via `calculate_knn`)
- `matplotlib>=3.3.0`

## Development Workflow

Every new plan or significant feature gets its own git branch. `PLAN.md` lives on the feature branch, not on main.
