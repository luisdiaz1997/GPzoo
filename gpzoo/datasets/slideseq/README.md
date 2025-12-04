# SlideseqV2 Dataset Training

This directory contains training scripts for four Gaussian Process models on the SlideseqV2 spatial transcriptomics dataset.

## Available Models

1. **SVGP (Sparse Variational GP)** - `svgp_nsf.py`
   - Standard sparse variational GP with inducing points
   - Uses Matern32 kernel
   - Suitable for general spatial modeling

2. **SVGP-MGGP (Multi-Group GP)** - `svgp_mggp_nsf.py`
   - SVGP with multi-group kernel
   - Incorporates cell type/cluster information via `group_diff_param`
   - Better for datasets with known cell type structure

3. **VNNGP (Variational Nearest Neighbor GP)** - `vnngp_nsf.py`
   - Uses K-nearest neighbors for scalability
   - Matern32 kernel with neighbor-based approximation
   - Good for very large datasets

4. **VNNGP-MGGP** - `vnngp_mggp_nsf.py`
   - Combines VNNGP with multi-group kernel
   - Most sophisticated model for large, structured datasets

## Quick Start

### Single Model Training

```bash
# SVGP (standard)
python -m gpzoo.datasets.slideseq.svgp_nsf

# SVGP-MGGP (with cell type groups)
python -m gpzoo.datasets.slideseq.svgp_mggp_nsf

# VNNGP (nearest neighbor)
python -m gpzoo.datasets.slideseq.vnngp_nsf

# VNNGP-MGGP (nearest neighbor with groups)
python -m gpzoo.datasets.slideseq.vnngp_mggp_nsf
```

### Parallel Training on Multiple GPUs

```bash
# GPU 0: Run both VNNGP models
CUDA_VISIBLE_DEVICES=0 python -m gpzoo.datasets.slideseq.vnngp_nsf &
CUDA_VISIBLE_DEVICES=0 python -m gpzoo.datasets.slideseq.vnngp_mggp_nsf &

# GPU 1: Run both SVGP models
CUDA_VISIBLE_DEVICES=1 python -m gpzoo.datasets.slideseq.svgp_nsf &
CUDA_VISIBLE_DEVICES=1 python -m gpzoo.datasets.slideseq.svgp_mggp_nsf &
```

## Configuration Settings

All models share common configuration in `config.py`:

### Training Parameters
- `STEPS = 34000` - Total training iterations
- `X_BATCH = 7000` - Spatial locations per batch
- `Y_BATCH = 1000` - Genes per batch
- `L_FACTORS = 10` - Number of latent factors
- `SEED = 123` - Random seed for reproducibility

### Learning Rates
- `LR = 1e-2` - Base learning rate for most parameters
- `LR_SCALE = 1e-4` - Learning rate for scale (Lu) parameters
- `LR_LENGTHSCALE = 1e-4` - Learning rate for lengthscale (unfrozen later)

### Kernel Parameters
- `LENGTHSCALE = 8.0` - Initial kernel lengthscale
- `LENGTHSCALE_TRAIN_AFTER = STEPS` - When to unfreeze lengthscale (default: never)
- `SPATIAL_SCALE = 50.0` - Spatial coordinate scaling factor
- `GROUP_DIFF_PARAM = 10.0` - MGGP group difference parameter
- `JITTER = 1e-5` - Numerical stability jitter

### Model-Specific Settings
- `SVGP_INDUCING = 3500` - Number of inducing points for SVGP
- `INDUCING_ALLOCATION = "equal"` - How to allocate inducing points across groups
- `VNNGP_K = 50` - Number of nearest neighbors for VNNGP
- `VNNGP_E_SAMPLES = 1` - Number of expectation samples for VNNGP

### Output Paths
- `OUTPUT_DIR` - Where models and logs are saved (`models/slideseq/`)
- `BENCHMARK_DIR` - Benchmark results directory
- Checkpoint paths: `slideseq_svgp.pth`, `slideseq_mggp_svgp.pth`, etc.

## Data Loading

The dataset is loaded via `load_slideseq_with_groups()` in `common.py`:

1. Loads SlideseqV2 dataset from squidpy
2. Filters: mitochondrial genes < 20%, min 100 counts/cell, min 10 cells/gene
3. Scales spatial coordinates by `SPATIAL_SCALE`
4. For MGGP models: extracts cell type clusters from `adata.obs['cluster']`
5. Returns: spatial coordinates (X), gene counts (Y), groups, size factors (V)

## Model Initialization

### SVGP Models
- Inducing points initialized via k-means (or from checkpoint)
- Lu (scale) matrix initialized via Cholesky of K(Z,Z) + jitter
- Diagonal stored in log-space: `gp.Lu = log_diagonals(Lu_base)`

### VNNGP Models
- K-nearest neighbors precomputed for all spatial locations
- Lu initialized via SVD of subset covariance

### Checkpoint Resumption
All models support checkpoint resumption:
- If checkpoint exists at expected path, training resumes
- Inducing points/groups loaded from checkpoint state dict
- Step counter continues from saved state

## Training Process

### Parameter Freezing
Initially frozen:
- Kernel parameters (lengthscale, sigma, group_diff_param)
- Inducing points (Z) and groups (groupsZ)
- Size factors (V)

Initially trainable:
- Mean parameters (mu)
- Scale parameters (Lu) - with separate `LR_SCALE`
- Weights (W)

### Key Implementation Details

**Lu Parameter Storage:**
- Diagonal stored in log-space: `gp.Lu = log_diagonals(Lu_base)`
- During forward pass: diagonal is exponentiated via `apply_constraints`
- This ensures positive definiteness while allowing unconstrained optimization

**Cholesky Initialization (SVGP):**
- When `use_cholesky=True`, Lu initialized via `chol(K(Z,Z) + jitter*I)`
- More stable than SVD for well-conditioned covariance matrices
- Default: Cholesky for SVGP, SVD for VNNGP

**Separate Learning Rates:**
- `LR = 1e-2` for most parameters (mu, W)
- `LR_SCALE = 1e-4` for Lu parameters (slower learning)
- `LR_LENGTHSCALE = 1e-4` for lengthscale (when unfrozen)

**TensorBoard Logging:**
- Lu diagonal logged after exp transformation: `exp(diag(Lu))`
- Factor visualizations use 5th/95th percentiles for color scaling
- 10 factors displayed in 2×5 grid layout

### Lengthscale Training
By default, lengthscale remains frozen (`LENGTHSCALE_TRAIN_AFTER = STEPS`).
To enable lengthscale training, set `LENGTHSCALE_TRAIN_AFTER` to a step number < `STEPS`.

### TensorBoard Logging
All models log to TensorBoard:
- Loss curves
- Kernel parameters (lengthscale, group_diff_param)
- Factor means and scales
- Lu diagonal statistics
- Factor visualizations every `IMAGE_LOG_EVERY` steps

View logs:
```bash
tensorboard --logdir models/slideseq/tb/
```

## Output Files

Each training run produces:
- `*.pth` - Model checkpoint (weights)
- `*_losses.csv` - Loss history
- `*_losses.npy` - Loss history (numpy)
- `*.json` - Training metadata (time, memory, device)
- TensorBoard logs in `tb/` subdirectory

## Model Selection Guide

| Model | Use Case | Key Features |
|-------|----------|--------------|
| **SVGP** | Standard spatial modeling | Inducing points, Matern32 kernel |
| **SVGP-MGGP** | Cell type-aware modeling | Group-specific kernels, `group_diff_param` |
| **VNNGP** | Large datasets | K-nearest neighbors, scalable |
| **VNNGP-MGGP** | Large, structured datasets | Combines VNNGP with group structure |

## Troubleshooting

### Memory Issues
- Reduce `X_BATCH` and `Y_BATCH` if OOM occurs
- VNNGP models use more memory due to neighbor storage
- Check GPU memory with `nvidia-smi`

### Training Divergence
- Reduce `LR` if loss becomes NaN
- Check `JITTER` value (increase if numerical issues)
- Verify `LENGTHSCALE` is appropriate for spatial scale

### Checkpoint Issues
- Ensure checkpoint paths match in `config.py`
- Delete corrupt checkpoints to start fresh
- Check TensorBoard logs for training curves

## Module Structure

```
gpzoo/datasets/slideseq/
├── README.md                 # This file
├── __init__.py              # Package exports
├── config.py                # Shared configuration
├── common.py                # Data loading and training utilities
├── svgp_nsf.py              # SVGP training script
├── svgp_mggp_nsf.py         # SVGP-MGGP training script
├── vnngp_nsf.py             # VNNGP training script
└── vnngp_mggp_nsf.py        # VNNGP-MGGP training script
```

## References

- SlideseqV2 Dataset: [10x Genomics](https://www.10xgenomics.com/)
- SVGP: Titsias (2009), "Variational Learning of Inducing Variables in Sparse Gaussian Processes"
- VNNGP: Wu et al. (2022), "Variational Nearest Neighbor Gaussian Process"
- MGGP: Custom multi-group kernel extension