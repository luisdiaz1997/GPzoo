# 10X Visium Benchmarking Pipeline

This directory contains scripts for benchmarking PCA, NMF, VNNGP, and MGGP_VNNGP models on 10X Visium spatial transcriptomics datasets.

## Overview

The benchmarking pipeline consists of two main steps:

1. **Annotation Generation** (uses GPzoo conda environment): Load trained models, extract factors, apply K-means clustering, and save annotations
2. **Benchmarking** (uses SDMBench conda environment): Compute metrics using SDMBench and compare model performance

## Files

- `generate_annotations.py` - Python script to generate annotations for all datasets (uses GPzoo environment)
- `benchmark.py` - Python script to benchmark all annotations (uses SDMBench environment)
- `vnngp_mggp_nsf.py` - Model training script for MGGP_VNNGP
- `vnngp_nsf.py` - Model training script for VNNGP
- `common.py` - Shared utilities
- `config.py` - Configuration settings

## Directory Structure

```
/gladstone/engelhardt/home/lchumpitaz/gitclones/
├── SDMBench/
│   ├── Data/
│   │   ├── 151507.h5ad
│   │   ├── 151508.h5ad
│   │   └── ... (all 10X Visium datasets)
│   └── Benchmarks/
│       ├── 151507_pca.txt
│       ├── 151507_nmf.txt
│       ├── 151507_vnngp.txt
│       ├── 151507_mggp_vnngp.txt
│       ├── 151507_benchmark_results.csv
│       └── ... (annotations and results for all datasets)
└── GPzoo/
    ├── models/tenxvisium/
    │   ├── 151507_vnngp_k=50.pth
    │   ├── 151507_mggp_vnngp_k=50.pth
    │   └── ... (trained models)
    └── gpzoo/
        └── datasets/
            └── tenxvisium/
                ├── generate_annotations.py
                ├── benchmark.py
                ├── vnngp_mggp_nsf.py
                ├── vnngp_nsf.py
                ├── common.py
                └── config.py
```

## Usage

### Step 1: Generate Annotations (GPzoo Environment)

**IMPORTANT**: This step requires the GPzoo conda environment.

#### Command to run in tmux:

```bash
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.generate_annotations
```

Or from the GPzoo root directory:
```bash
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.generate_annotations
```

This will:
- Load all .h5ad files from SDMBench/Data (datasets starting with '151')
- For each dataset:
  - Generate PCA baseline annotations
  - Generate NMF baseline annotations
  - Load VNNGP model (if available) and generate annotations
  - Load MGGP_VNNGP model (if available) and generate annotations
- Save all annotations as .txt files in SDMBench/Benchmarks/

**Expected output**: Annotation .txt files in `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/`

---

### Step 2: Benchmark Annotations (SDMBench Environment)

**IMPORTANT**: This step requires the SDMBench conda environment (different from GPzoo).

#### Command to run in tmux:

```bash
~/miniconda3/envs/sdmbench/bin/python -m gpzoo.datasets.tenxvisium.benchmark
```

Or with full path:
```bash
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo
~/miniconda3/envs/sdmbench/bin/python -m gpzoo.datasets.tenxvisium.benchmark
```

**Note**: If your SDMBench environment has a different name, replace `sdmbench` with your environment name.

This will:
- Load all .h5ad files and corresponding .txt annotation files
- Compute metrics using SDMBench:
  - **Accuracy**: ARI, NMI, Homogeneity, Completeness
  - **Continuity**: CHAOS, PAS, ASW
  - **Marker score**: Moran's I, Geary's C
- Save results to `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/results/`:
  - Individual dataset results: `{dataset_name}_benchmark_results.csv`
  - Summary across all datasets: `summary_all_datasets.csv`
  - Mean metrics only: `summary_means_only.csv`

---

### Complete Workflow (Copy-Paste for tmux)

```bash
# Step 1: Generate annotations (GPzoo environment)
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.generate_annotations

# Step 2: Benchmark annotations (SDMBench environment)
~/miniconda3/envs/sdmbench/bin/python -m gpzoo.datasets.tenxvisium.benchmark
```

### Process Individual Datasets

You can modify the scripts to process specific datasets:

```python
# In generate_annotations.py
dataset_names = ['151507', '151508']  # Specify which datasets to process
```

## Models

The pipeline expects the following model types:

1. **PCA**: Generated from data (no pre-trained model needed)
2. **NMF**: Generated from data (no pre-trained model needed)
3. **VNNGP**: Expects `{dataset_name}_vnngp_k=50.pth` in models/tenxvisium/
4. **MGGP_VNNGP**: Expects `{dataset_name}_mggp_vnngp_k=50.pth` in models/tenxvisium/

## Data Requirements

Each .h5ad file should contain:
- `adata.obsm['X_spatial']` or `adata.obsm['spatial']`: Spatial coordinates
- `adata.obs['ground_truth']`: Ground truth labels for benchmarking
- `adata.raw.X` or `adata.X`: Gene expression matrix

## Model Architecture

All models use:
- **L = 12**: Number of latent factors
- **K = 50**: Number of nearest neighbors for VNNGP/MGGP_VNNGP

## Annotation Format

Annotation files are saved as .txt files with the format:
```
{dataset_name}_{method}
1
2
0
...
```

Where each line is a cluster label (0 to K-1) for each cell.

## Benchmarking Metrics

### Accuracy Metrics
- **ARI** (Adjusted Rand Index): Measures agreement between predicted and ground truth labels
- **NMI** (Normalized Mutual Information): Information-theoretic measure of clustering quality
- **Homogeneity**: Whether clusters contain only members of a single class
- **Completeness**: Whether all members of a class are in the same cluster

### Continuity Metrics
- **CHAOS**: Spatial continuity measure
- **PAS** (Probabilistic Assessment of Spatial expression): Spatial coherence
- **ASW** (Average Silhouette Width): Cluster separation in spatial coordinates

### Marker Score Metrics
- **Moran's I**: Spatial autocorrelation of marker genes
- **Geary's C**: Local spatial autocorrelation

## Output Files

### Individual Dataset Results
`{dataset_name}_benchmark_results.csv`:
```
method,Accuracy_ARI,Accuracy_NMI,...
151507_pca,0.45,0.52,...
151507_nmf,0.48,0.55,...
151507_vnngp,0.62,0.68,...
151507_mggp_vnngp,0.65,0.71,...
```

### Summary Results
`summary_all_datasets.csv`: Mean and standard deviation of each metric across all datasets

`summary_means_only.csv`: Just the mean values for easier comparison

## Troubleshooting

### Missing Models
If a model file is not found, the script will skip it and continue with other methods:
```
Warning: VNNGP model not found: /path/to/151507_vnngp_k=50.pth
```

### Memory Issues
If you encounter OOM errors:
- Process datasets one at a time
- Reduce the number of cells used for some metrics (e.g., use `[::20]` instead of `[::10]`)
- Close other applications

### CUDA Errors
The scripts use CPU by default. If you want to use GPU:
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

## Customization

### Change Number of Clusters
By default, the number of clusters is determined from ground truth labels. To override:
```python
n_clusters = 10  # Fixed number
```

### Change Model Parameters
```python
L = 15  # Number of factors
K = 100  # Number of nearest neighbors
```

### Add New Metrics
Add custom metrics in the `compute_metrics` function:
```python
def compute_metrics(adata, pred_labels, method_name):
    # ... existing metrics ...

    # Add your custom metric
    custom_metric = your_metric_function(adata, pred_labels)

    return {
        # ... existing metrics ...
        ('Custom', 'MyMetric'): custom_metric,
    }
```

## References

- Original liver analysis: `liver_mggp_healthy_exploratory.ipynb`
- SDMBench benchmarking example: `SDMBench/Benchmarks/liver.ipynb`
- GPzoo models: `gpzoo/models.py`
- Training scripts: `gpzoo/datasets/tenxvisium/`
