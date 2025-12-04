# GPzoo: Gaussian Process Models for Spatial Transcriptomics

GPzoo is a Python library for scalable Gaussian Process models tailored for spatial transcriptomics data analysis. It provides implementations of several state-of-the-art GP approximations with multi-group extensions for cell type-aware modeling.

## Features

- **Multiple GP Approximations**:
  - SVGP (Sparse Variational GP) with inducing points
  - VNNGP (Variational Nearest Neighbor GP) for large datasets
  - Multi-Group extensions (MGGP) for cell type-aware modeling

- **Spatial Transcriptomics Integration**:
  - Built-in support for SlideseqV2, 10x Visium, and liver datasets
  - Automatic data loading and preprocessing
  - Integration with Scanpy and Squidpy ecosystems

- **Scalable Training**:
  - Batched training for memory efficiency
  - GPU acceleration support
  - TensorBoard logging for monitoring
  - Checkpoint resumption

- **Model Variants**:
  - SVGP-NSF (Neural Spectral Factorization)
  - VNNGP-NSF
  - Multi-Group versions of all models

## Installation

### From Source
```bash
git clone https://github.com/luisdiaz1997/GPzoo.git
cd GPzoo
pip install -e .
```

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
# Visit https://pytorch.org/get-started/locally/ for the appropriate command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Training a Model

```python
from gpzoo.models import build_model

# Load your spatial data (X: coordinates, Y: gene counts)
X, Y = load_your_data()

# Build and train an SVGP model
model, _ = build_model(
    "slideseq/svgp_nsf",
    X=X,
    Y=Y,
    L=10,  # number of latent factors
    lengthscale=8.0,
    device="cuda"
)

# Or use the provided training scripts
python -m gpzoo.datasets.slideseq.svgp_nsf
```

### Available Models

| Model | Command | Description |
|-------|---------|-------------|
| SVGP | `python -m gpzoo.datasets.slideseq.svgp_nsf` | Standard sparse variational GP |
| SVGP-MGGP | `python -m gpzoo.datasets.slideseq.svgp_mggp_nsf` | Multi-group SVGP |
| VNNGP | `python -m gpzoo.datasets.slideseq.vnngp_nsf` | Nearest neighbor GP |
| VNNGP-MGGP | `python -m gpzoo.datasets.slideseq.vnngp_mggp_nsf` | Multi-group VNNGP |

### Parallel Training
```bash
# Run multiple models on different GPUs
CUDA_VISIBLE_DEVICES=0 python -m gpzoo.datasets.slideseq.vnngp_nsf &
CUDA_VISIBLE_DEVICES=1 python -m gpzoo.datasets.slideseq.svgp_nsf &
```

## Project Structure

```
GPzoo/
├── gpzoo/                    # Main library
│   ├── kernels.py           # GP kernel implementations
│   ├── likelihoods.py       # Likelihood functions
│   ├── models.py            # Model registry and builder
│   ├── model_utilities.py   # Model construction utilities
│   ├── training_utilities.py # Training loops and logging
│   ├── utilities.py         # General utilities
│   └── datasets/            # Dataset-specific code
│       ├── slideseq/        # SlideseqV2 dataset
│       ├── liver/           # Liver spatial transcriptomics
│       └── tenxvisium/      # 10x Visium dataset
├── notebooks/               # Example notebooks and analysis
├── models/                  # Saved model checkpoints
└── requirements.txt         # Python dependencies
```

## Configuration

Models are configured via shared configuration files in each dataset directory. Key parameters:

- `STEPS = 34000` - Training iterations
- `X_BATCH = 7000` - Spatial locations per batch
- `Y_BATCH = 1000` - Genes per batch
- `L_FACTORS = 10` - Number of latent factors
- `LR = 1e-2` - Base learning rate
- `LR_SCALE = 1e-4` - Learning rate for scale parameters
- `LENGTHSCALE = 8.0` - Kernel lengthscale

## Model Details

### SVGP (Sparse Variational GP)
- Uses inducing points for scalability
- Matern32 kernel for spatial smoothness
- Neural Spectral Factorization (NSF) for covariance modeling
- Suitable for medium-sized datasets

### VNNGP (Variational Nearest Neighbor GP)
- Uses K-nearest neighbors for approximation
- More scalable for very large datasets
- Default: K=50 nearest neighbors
- 1 expectation sample during training

### Multi-Group GP (MGGP)
- Extends kernels with group-specific parameters
- Incorporates cell type/cluster information
- `group_diff_param` controls between-group similarity
- Requires `groupsX` tensor indicating cell types

### Lu Parameter Storage
- Diagonal stored in log-space for positive definiteness
- During forward pass: diagonal exponentiated via `apply_constraints`
- Separate learning rate (`LR_SCALE`) for scale parameters
- Cholesky initialization for SVGP, SVD for VNNGP

## Outputs

Each training run produces:
- `*.pth` - Model checkpoint
- `*_losses.csv` - Loss history
- `*_losses.npy` - Loss history (numpy)
- `*.json` - Training metadata (time, memory, device)
- TensorBoard logs in `models/*/tb/`

View TensorBoard logs:
```bash
tensorboard --logdir models/slideseq/tb/
```

## Datasets

### SlideseqV2
- Mouse brain spatial transcriptomics
- ~50,000 spots, ~20,000 genes (filtered)
- Cell type clusters from original publication
- Automatic loading via `load_slideseq_with_groups()`

### Liver Spatial Transcriptomics
- Human liver disease vs healthy
- Spatial coordinates and gene counts
- Disease state labels for MGGP

### 10x Visium
- Standard spatial transcriptomics format
- Compatible with Scanpy/Squidpy workflows

## Advanced Usage

### Custom Datasets
```python
from gpzoo.models import build_model

# Your data: X (coordinates), Y (gene counts), groups (optional)
X = torch.tensor(coordinates, dtype=torch.float32)
Y = torch.tensor(gene_counts, dtype=torch.float32)
groups = torch.tensor(cell_types, dtype=torch.long)  # for MGGP

model, _ = build_model(
    "slideseq/svgp_mggp_nsf",
    X=X,
    Y=Y,
    groupsX=groups,
    L=12,
    lengthscale=10.0,
    device="cuda"
)
```

### Checkpoint Resumption
```python
# Training resumes automatically if checkpoint exists
model, _ = build_model(
    "slideseq/svgp_nsf",
    checkpoint_path="models/slideseq/slideseq_svgp.pth",
    X=X,
    Y=Y,
    # ... other parameters
)
```

### Hyperparameter Tuning
Modify configuration in dataset-specific `config.py`:
- Adjust `LENGTHSCALE` for spatial scale
- Change `X_BATCH`, `Y_BATCH` for memory usage
- Set `LENGTHSCALE_TRAIN_AFTER` to unfreeze lengthscale
- Modify `GROUP_DIFF_PARAM` for MGGP group similarity

## Troubleshooting

### Memory Issues
- Reduce `X_BATCH` and `Y_BATCH` sizes
- VNNGP uses more memory due to neighbor storage
- Monitor GPU memory with `nvidia-smi`

### Training Divergence
- Reduce learning rate (`LR`, `LR_SCALE`)
- Increase `JITTER` for numerical stability
- Check `LENGTHSCALE` is appropriate for spatial scale

### Checkpoint Issues
- Ensure checkpoint paths exist in `config.py`
- Delete corrupt checkpoints to start fresh
- Check TensorBoard logs for training curves

## Citation

If you use GPzoo in your research, please cite:

```bibtex
@software{gpzoo2024,
  title = {GPzoo: Gaussian Process Models for Spatial Transcriptomics},
  author = {Diaz, Luis},
  year = {2024},
  url = {https://github.com/luisdiaz1997/GPzoo}
}
```

## References

- SVGP: Titsias (2009), "Variational Learning of Inducing Variables in Sparse Gaussian Processes"
- VNNGP: Wu et al. (2022), "Variational Nearest Neighbor Gaussian Process"
- SlideseqV2: [10x Genomics](https://www.10xgenomics.com/)
- Scanpy: Wolf et al. (2018), "SCANPY: large-scale single-cell gene expression data analysis"
- Squidpy: Palla et al. (2022), "Squidpy: a scalable framework for spatial omics analysis"

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
