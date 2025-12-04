# Quick Reference: Run Commands for Benchmarking

## Copy-Paste Commands for tmux

### Step 1: Generate Annotations (GPzoo environment)
```bash
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.generate_annotations
```

### Step 2: Run Benchmarks (SDMBench environment)
```bash
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo
~/miniconda3/envs/SDMBench/bin/python -m gpzoo.datasets.tenxvisium.benchmark
```

---

## What Each Command Does

### 1. Generate Annotations
**Environment**: `gpzoo`
**Input**:
- Data: `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/*.h5ad`
- Models: `/gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo/models/tenxvisium/*.pth`

**Output**: `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/`
- `{dataset}_pca.txt`
- `{dataset}_nmf.txt`
- `{dataset}_vnngp.txt`
- `{dataset}_mggp_vnngp.txt`
- `{dataset}_mggp_vnngp_{region}.txt` (group-wise)

**What it does**:
- Loads each .h5ad dataset
- Generates PCA baseline (12 components)
- Generates NMF baseline (12 factors)
- Loads VNNGP model, extracts factors in **log space**, applies K-means
- Loads MGGP_VNNGP model, extracts factors in **log space**, applies K-means
- K = number of unique regions from `adata.obs['Region']`
- Saves cluster labels to .txt files

### 2. Run Benchmarks
**Environment**: `sdmbench`
**Input**:
- Data: `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/*.h5ad`
- Annotations: `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/*.txt`

**Output**: `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/results/`
- `{dataset}_benchmark_results.csv` (per dataset)
- `summary_all_datasets.csv` (aggregated)
- `summary_means_only.csv` (mean metrics only)

**Metrics computed**:
- **Accuracy**: ARI, NMI, Homogeneity, Completeness
- **Continuity**: CHAOS, PAS, ASW
- **Marker score**: Moran's I, Geary's C

**What it does**:
- Loads each .h5ad dataset
- Loads corresponding annotation .txt files
- Compares predictions to `adata.obs['Region']` ground truth
- Computes all SDMBench metrics
- Creates summary across all datasets

---


## Comparison to Model Training Commands

**Training VNNGP**:
```bash
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.vnngp_nsf
```

**Training MGGP_VNNGP**:
```bash
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.vnngp_mggp_nsf
```

**Generating Annotations**:
```bash
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.generate_annotations
```

**Benchmarking** (different environment!):
```bash
~/miniconda3/envs/sdmbench/bin/python -m gpzoo.datasets.tenxvisium.benchmark
```

---

## Full Pipeline (All Steps)

```bash
# Navigate to project
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo

# 1. Train VNNGP models (if not done)
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.vnngp_nsf

# 2. Train MGGP_VNNGP models (if not done)
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.vnngp_mggp_nsf

# 3. Generate annotations from trained models
~/miniconda3/envs/gpzoo/bin/python -m gpzoo.datasets.tenxvisium.generate_annotations

# 4. Benchmark all methods (switch to SDMBench environment)
~/miniconda3/envs/sdmbench/bin/python -m gpzoo.datasets.tenxvisium.benchmark
```

---

## Checking Results

### Check annotation files:
```bash
ls -lh /gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/*.txt
```

### Check benchmark results:
```bash
ls -lh /gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/results/
```

### View summary:
```bash
cat /gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks/results/summary_means_only.csv
```
