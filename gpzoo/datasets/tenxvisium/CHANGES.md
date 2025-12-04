# Changes Made to Benchmarking Scripts

## Summary of Updates

Based on your requirements, I've updated the annotation generation and benchmarking scripts with the following key changes:

### 1. Use `Region` Column from config.py ✓

**What changed:**
- Both `generate_annotations.py` and `benchmark.py` now use `REGION_COLUMN = "Region"` from config.py
- Replaced all references to `'ground_truth'` with the `REGION_COLUMN` constant
- Number of K-means clusters (K) is now equal to `len(adata.obs[REGION_COLUMN].unique())`

**Files affected:**
- `generate_annotations.py`: Lines 28, 31, 42, 48, 57-59, 263, 346, 350
- `benchmark.py`: Lines 30, 33, 75, 100, 284
- `config.py`: Line 19 (already defined)

### 2. K-means Clusters = Number of Regions ✓

**What changed:**
- The number of K-means clusters is now dynamically set based on the unique values in `adata.obs[REGION_COLUMN]`
- Previously used a hardcoded default of 7; now uses actual region count
- Code: `n_clusters = len(adata.obs[REGION_COLUMN].astype('category').cat.categories)`

**Files affected:**
- `generate_annotations.py`: Line 264

### 3. Use qF.mean (Log Space) for K-means ✓

**What changed:**
- **Previously**: Used `np.log(torch.exp(qF.mean))` - exponentiating then taking log (redundant!)
- **Now**: Use `qF.mean` directly (already in log space)
- **Rationale**: exp() can mislead K-means clustering by amplifying differences

**Specific changes:**

#### In `extract_factors()` function:
```python
# OLD:
mean = torch.exp(qF.mean).detach().numpy()
return mean, scale

# NEW:
mean_log = qF.mean.detach().numpy()  # For K-means
mean_exp = torch.exp(qF.mean).detach().numpy()  # For visualization
return mean_log, mean_exp, scale
```

#### In `get_groupwise_factors()` function:
```python
# OLD:
factors_test = torch.exp(qF_test.mean).cpu().numpy()
results.append({'factors': factors_test, ...})

# NEW:
factors_log = qF_test.mean.cpu().numpy()  # For K-means
factors_exp = torch.exp(qF_test.mean).cpu().numpy()  # For visualization
results.append({'factors_log': factors_log, 'factors_exp': factors_exp, ...})
```

#### In K-means calls:
```python
# OLD (VNNGP):
vnngp_labels = KMeans(...).fit_predict(np.log(mean_ordered.T))  # Taking log of exp!

# NEW (VNNGP):
vnngp_labels = KMeans(...).fit_predict(mean_log_ordered.T)  # Already in log space

# OLD (MGGP groupwise):
group_labels = KMeans(...).fit_predict(np.log(result['factors']).T)

# NEW (MGGP groupwise):
group_labels = KMeans(...).fit_predict(result['factors_log'].T)  # Use log factors
```

**Files affected:**
- `generate_annotations.py`: Lines 137-165, 168-206, 296-311, 328-364

## Why These Changes Matter

### 1. Region Column Consistency
Using the centralized `REGION_COLUMN` constant ensures:
- Consistency across training and benchmarking
- Easy updates if column name changes
- Clear documentation of expected data format

### 2. Correct Number of Clusters
Setting K = number of regions ensures:
- K-means matches the actual spatial structure
- Fair comparison across datasets with different region counts
- No arbitrary cluster numbers

### 3. Log Space for K-means
Using `qF.mean` directly (log space) instead of `exp(qF.mean)`:
- **Mathematical correctness**: NSF models output log-normal distributions
- **Better clustering**: exp() amplifies small differences, misleading K-means
- **Consistent with your liver notebook**: You used `np.log(mean.T)` there too
- **Example**:
  - Log space: [0.1, 0.2, 0.3] → differences of ~0.1
  - Exp space: [1.1, 1.2, 1.3] → differences of ~0.1 but amplified relative differences
  - Taking log of exp brings us back but with numerical precision issues

## Testing Recommendations

Before running on all datasets, test on one dataset (e.g., 151507):

```bash
# 1. Generate annotations (GPzoo environment)
conda activate gpzoo
cd /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo/gpzoo/datasets/tenxvisium

# Test with a single dataset by modifying generate_annotations.py:
# In main(), change dataset_names to: dataset_names = ['151507']
python generate_annotations.py

# 2. Benchmark (SDMBench environment)
conda activate sdmbench
python benchmark.py
```

## Verification Checklist

- [ ] REGION_COLUMN = "Region" used throughout
- [ ] n_clusters = number of unique regions
- [ ] K-means uses qF.mean (log space) not exp(qF.mean)
- [ ] Moran's I still uses exp(qF.mean) for visualization
- [ ] Both environments (gpzoo and sdmbench) work correctly

## Files Modified

1. `generate_annotations.py` - Main annotation generation script
2. `benchmark.py` - Main benchmarking script
3. `README_BENCHMARKING.md` - Updated documentation
4. `CHANGES.md` - This file

## Original References

- Original liver workflow: `notebooks/liver_mggp_healthy_exploratory.ipynb`
- SDMBench example: `SDMBench/Benchmarks/liver.ipynb`
