"""
Parallel benchmark script for 10X Visium datasets using SDMBench.

This script parallelizes benchmarking across datasets using multiprocessing.
Each dataset is processed in a separate worker process.

Usage:
    python -m gpzoo.datasets.tenxvisium.benchmark_parallel --workers 4

NOTE: This script should be run in the SDMBench conda environment.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# Import SDMBench
try:
    from SDMBench import sdmbench
except ImportError:
    print("ERROR: SDMBench not found. Make sure you're in the SDMBench conda environment.")
    sys.exit(1)

# Note: Can't import from .config here because we're in a different conda env
# We'll just hard-code the REGION_COLUMN value
REGION_COLUMN = "Region"


def _set_single_threaded():
    """
    Force single-threaded execution to avoid nested parallelism issues.
    This prevents 'daemonic processes cannot have children' errors.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def compute_metrics_for_prediction(adata, pred_path: Path, label_column: str = REGION_COLUMN,
                                  skip_marker_scores: bool = False) -> Dict:
    """
    Compute benchmark metrics for a single prediction file.

    Args:
        adata: AnnData object with ground truth labels
        pred_path: Path to prediction file
        label_column: Column name in adata.obs containing ground truth labels
        skip_marker_scores: If True, skip marker scores (use when in worker process)

    Returns:
        Dictionary with computed metrics
    """
    pred = pd.read_csv(pred_path)
    method_name = pred.columns[0]

    if pred.shape[0] != adata.shape[0]:
        raise ValueError(f'{pred_path.name}: expected {adata.shape[0]} predictions, got {pred.shape[0]}')

    adata_with_pred = adata.copy()
    adata_with_pred.obs['pred'] = pred[method_name].values

    # Compute metrics
    if skip_marker_scores:
        # Skip marker scores in worker processes to avoid multiprocessing issues
        moranI, gearyC = np.nan, np.nan
    else:
        # Compute marker scores (only in main process)
        try:
            moranI, gearyC = sdmbench.marker_score(adata_with_pred, label_column)
        except Exception as e:
            print(f"Warning: Could not compute marker scores for {method_name}: {e}")
            moranI, gearyC = np.nan, np.nan

    return {
        'method': method_name,
        ('Accuracy', 'ARI'): sdmbench.compute_ARI(adata_with_pred, label_column, 'pred'),
        ('Accuracy', 'NMI'): sdmbench.compute_NMI(adata_with_pred, label_column, 'pred'),
        ('Accuracy', 'HOM'): sdmbench.compute_HOM(adata_with_pred, label_column, 'pred'),
        ('Accuracy', 'COM'): sdmbench.compute_COM(adata_with_pred, label_column, 'pred'),
        ('Continuity', 'CHAOS'): sdmbench.compute_CHAOS(adata_with_pred, 'pred'),
        ('Continuity', 'PAS'): sdmbench.compute_PAS(adata_with_pred, 'pred', spatial_key='spatial'),
        ('Continuity', 'ASW'): sdmbench.compute_ASW(adata_with_pred, 'pred', spatial_key='spatial'),
        ('Marker score', 'Moran\'I'): moranI,
        ('Marker score', "Geary's C"): gearyC,
    }


def load_adata(h5ad_path: Path, label_column: str = REGION_COLUMN) -> sc.AnnData:
    """Load and prepare AnnData object for benchmarking."""
    adata = sc.read_h5ad(h5ad_path)

    # Ensure spatial coordinates are in the right place
    if 'X_spatial' in adata.obsm and 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['X_spatial']

    # Remove NaN labels if present
    if label_column in adata.obs.columns:
        adata_valid = adata[np.logical_not(adata.obs[label_column].isna())]
        return adata_valid
    else:
        return adata


def benchmark_single_dataset(
    dataset_name: str,
    data_dir: Path,
    benchmark_dir: Path,
    output_dir: Path,
    label_column: str = REGION_COLUMN
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Benchmark a single dataset (designed to be called in parallel).

    Args:
        dataset_name: Name of dataset (e.g., '151507')
        data_dir: Directory containing .h5ad files
        benchmark_dir: Directory containing prediction .txt files
        output_dir: Directory to save results
        label_column: Column name in adata.obs containing ground truth labels

    Returns:
        Tuple of (dataset_name, results_dataframe or None)
    """
    # Force single-threaded execution in worker processes
    _set_single_threaded()

    print(f"\n[{dataset_name}] Starting benchmark...")

    try:
        # Load data
        h5ad_path = data_dir / f"{dataset_name}.h5ad"
        if not h5ad_path.exists():
            print(f"[{dataset_name}] ERROR: Data file not found: {h5ad_path}")
            return (dataset_name, None)

        adata = load_adata(h5ad_path, label_column=label_column)
        print(f"[{dataset_name}] Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

        # Find all prediction files for this dataset
        prediction_files = sorted(benchmark_dir.glob(f'{dataset_name}_*.txt'))

        if len(prediction_files) == 0:
            print(f"[{dataset_name}] WARNING: No prediction files found")
            return (dataset_name, None)

        print(f"[{dataset_name}] Found {len(prediction_files)} prediction files")

        # Compute metrics for each prediction (skip marker scores in worker process)
        results = []
        for pred_path in prediction_files:
            try:
                metrics = compute_metrics_for_prediction(adata, pred_path, label_column=label_column,
                                                        skip_marker_scores=True)
                results.append(metrics)
                print(f"[{dataset_name}] ✓ {pred_path.name}")
            except Exception as e:
                print(f"[{dataset_name}] ✗ {pred_path.name}: {e}")
                continue

        if len(results) == 0:
            print(f"[{dataset_name}] ERROR: No results computed")
            return (dataset_name, None)

        # Create DataFrame
        results_df = pd.DataFrame(results).set_index('method')
        results_df.columns = pd.MultiIndex.from_tuples(results_df.columns)

        # Save results
        output_path = output_dir / f"{dataset_name}_benchmark_results.csv"
        results_df.to_csv(output_path)
        print(f"[{dataset_name}] ✓ Saved results to: {output_path}")

        return (dataset_name, results_df)

    except Exception as e:
        print(f"[{dataset_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return (dataset_name, None)


def create_summary_report(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Create a summary report across all datasets.

    Args:
        all_results: Dictionary mapping dataset names to their benchmark DataFrames
        output_dir: Directory to save summary report
    """
    print(f"\n{'='*80}")
    print("Creating summary report")
    print(f"{'='*80}\n")

    # Collect all methods across all datasets
    all_methods = set()
    for df in all_results.values():
        if df is not None:
            all_methods.update(df.index)

    print(f"Methods found: {sorted(all_methods)}")

    # For each metric, compute average across datasets for each method
    if len(all_results) == 0:
        print("No results to summarize")
        return

    # Get metric names from first non-None result
    sample_df = next((df for df in all_results.values() if df is not None), None)
    if sample_df is None:
        print("No valid results found")
        return

    metrics = sample_df.columns

    # Create summary dataframe
    summary_data = []

    for method in sorted(all_methods):
        method_data = {'method': method}

        for metric in metrics:
            values = []
            for dataset_name, df in all_results.items():
                if df is not None and method in df.index:
                    val = df.loc[method, metric]
                    if not pd.isna(val):
                        values.append(val)

            if len(values) > 0:
                method_data[('Mean', metric[0], metric[1])] = np.mean(values)
                method_data[('Std', metric[0], metric[1])] = np.std(values)
            else:
                method_data[('Mean', metric[0], metric[1])] = np.nan
                method_data[('Std', metric[0], metric[1])] = np.nan

        summary_data.append(method_data)

    summary_df = pd.DataFrame(summary_data).set_index('method')
    summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)

    # Save summary
    summary_path = output_dir / "summary_all_datasets.csv"
    summary_df.to_csv(summary_path)
    print(f"Saved summary to: {summary_path}")

    # Display summary
    print("\nSummary across all datasets:")
    print(summary_df)

    # Create a simplified view with just means
    mean_cols = [col for col in summary_df.columns if col[0] == 'Mean']
    mean_df = summary_df[mean_cols]
    mean_df.columns = pd.MultiIndex.from_tuples([(col[1], col[2]) for col in mean_cols])

    print("\nMean metrics across all datasets:")
    print(mean_df)

    mean_path = output_dir / "summary_means_only.csv"
    mean_df.to_csv(mean_path)
    print(f"Saved mean metrics to: {mean_path}")


def main():
    """Main function to benchmark all datasets in parallel."""
    parser = argparse.ArgumentParser(description='Parallel benchmarking of 10X Visium datasets')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to benchmark (default: all datasets starting with 151)')
    args = parser.parse_args()

    # Determine number of workers
    n_workers = args.workers if args.workers is not None else cpu_count()
    print(f"Using {n_workers} parallel workers")

    # Setup paths
    data_dir = Path('/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data')
    benchmark_dir = Path('/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks')
    output_dir = benchmark_dir / 'results'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset names
    if args.datasets is not None:
        dataset_names = args.datasets
    else:
        # Get all h5ad files
        h5ad_files = sorted(data_dir.glob('*.h5ad'))

        # Filter to get dataset names (10X Visium datasets start with 151)
        dataset_names = []
        for h5ad_file in h5ad_files:
            name = h5ad_file.stem
            if name.startswith('151'):
                dataset_names.append(name)

    print(f"\nFound {len(dataset_names)} datasets to benchmark:")
    for name in dataset_names:
        print(f"  - {name}")

    # Create partial function with fixed arguments
    benchmark_func = partial(
        benchmark_single_dataset,
        data_dir=data_dir,
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
        label_column=REGION_COLUMN
    )

    # Benchmark datasets in parallel
    print(f"\n{'='*80}")
    print(f"Starting parallel benchmarking with {n_workers} workers")
    print(f"{'='*80}")

    with Pool(processes=n_workers) as pool:
        results = pool.map(benchmark_func, dataset_names)

    # Convert results to dictionary
    all_results = {name: df for name, df in results}

    # Count successful benchmarks
    n_success = sum(1 for df in all_results.values() if df is not None)
    n_failed = len(all_results) - n_success

    print(f"\n{'='*80}")
    print(f"Parallel benchmarking complete: {n_success} succeeded, {n_failed} failed")
    print(f"{'='*80}")

    # Post-process: Compute marker scores sequentially in main process
    print(f"\n{'='*80}")
    print("Computing marker scores (Moran's I, Geary's C) sequentially...")

    
    print(f"{'='*80}\n")

    for dataset_name in dataset_names:
        if all_results[dataset_name] is None:
            continue

        print(f"\n[{dataset_name}] Computing marker scores...")

        try:
            # Load data
            h5ad_path = data_dir / f"{dataset_name}.h5ad"
            adata = load_adata(h5ad_path, label_column=REGION_COLUMN)

            # Find prediction files
            prediction_files = sorted(benchmark_dir.glob(f'{dataset_name}_*.txt'))

            # Load existing results
            results_path = output_dir / f"{dataset_name}_benchmark_results.csv"
            results_df = pd.read_csv(results_path, header=[0, 1], index_col=0)

            # Compute marker scores for each method
            for pred_path in prediction_files:
                pred = pd.read_csv(pred_path)
                method_name = pred.columns[0]

                if method_name not in results_df.index:
                    continue

                # Compute marker scores
                adata_with_pred = adata.copy()
                adata_with_pred.obs['pred'] = pred[method_name].values

                try:
                    moranI, gearyC = sdmbench.marker_score(adata_with_pred, REGION_COLUMN)

                    # Update the results
                    results_df.loc[method_name, ('Marker score', "Moran'I")] = moranI
                    results_df.loc[method_name, ('Marker score', "Geary's C")] = gearyC

                    print(f"  ✓ {method_name}: Moran's I={moranI:.3f}, Geary's C={gearyC:.3f}")
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    continue

            # Save updated results
            results_df.to_csv(results_path)
            all_results[dataset_name] = results_df
            print(f"[{dataset_name}] ✓ Updated results with marker scores")

        except Exception as e:
            print(f"[{dataset_name}] ERROR computing marker scores: {e}")
            continue

    # Create summary report with updated marker scores
    print(f"\n{'='*80}")
    print("Creating summary report...")
    print(f"{'='*80}")
    create_summary_report(all_results, output_dir)

    print(f"\n{'='*80}")
    print("All benchmarking complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
