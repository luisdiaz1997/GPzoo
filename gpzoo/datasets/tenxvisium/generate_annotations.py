"""
Generate annotations for all 10X Visium datasets using trained models.

This script:
1. Loads .h5ad files from SDMBench/Data
2. Loads trained VNNGP and MGGP_VNNGP models from gpzoo/models/tenxvisium
3. Generates factors and applies K-means clustering
4. Saves annotations to SDMBench/Benchmarks/
"""

import torch
import numpy as np
import scanpy as sc
import anndata as ad
from torch import nn
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA

# Import GPzoo modules
from gpzoo.kernels import batched_MGGP_RBF
from gpzoo.gp import MGGP_VNNGP
from gpzoo.likelihoods import NSF2
from gpzoo.utilities import scanpy_sizefactors, dims_autocorr, init_softplus, rescale_spatial_coords

# Import config
from .config import REGION_COLUMN, SPATIAL_SCALE


def load_adata(h5ad_path, region_column=REGION_COLUMN):
    """Load and prepare AnnData object."""
    adata = ad.read_h5ad(h5ad_path)

    # Ensure spatial coordinates are available
    if 'X_spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['X_spatial']
    elif 'spatial' not in adata.obsm:
        raise ValueError(f"No spatial coordinates found in {h5ad_path}")

    # Remove NaN labels if present
    if region_column in adata.obs.columns:
        adata = adata[np.logical_not(adata.obs[region_column].isna())]

    return adata


def get_model_architecture(adata, L=12, K=50, region_column=REGION_COLUMN):
    """
    Determine model architecture from data.
    Returns D (genes), N (cells), n_groups, and group codes.

    IMPORTANT: Rescales spatial coordinates to match training:
        X = rescale_spatial_coords(X) * SPATIAL_SCALE
    """
    import scipy.sparse as sp

    # Rescale spatial coordinates (same as in training scripts)
    X_np = np.asarray(adata.obsm['spatial'], dtype=np.float32)
    X_np = rescale_spatial_coords(X_np) * SPATIAL_SCALE
    X = torch.tensor(X_np).type(torch.float)

    # Get expression matrix and convert sparse to dense if needed
    if adata.raw:
        Y_matrix = adata.raw.X.T
    else:
        Y_matrix = adata.X.T

    if sp.issparse(Y_matrix):
        Y_matrix = Y_matrix.toarray()

    Y = torch.tensor(Y_matrix).type(torch.float)

    # Determine groups from region labels
    if region_column in adata.obs.columns:
        groupsX = torch.tensor(adata.obs[region_column].astype('category').cat.codes.values).type(torch.LongTensor)
        n_groups = len(adata.obs[region_column].astype('category').cat.categories)
    else:
        # No region labels - use single group
        groupsX = torch.zeros(X.shape[0], dtype=torch.long)
        n_groups = 1

    D, N = Y.shape

    return X, Y, groupsX, n_groups, D, N, L, K


def create_vnngp_model(adata, L=12, K=50):
    """Create VNNGP model architecture (non-MGGP)."""
    X, Y, groupsX, n_groups, D, N, L, K = get_model_architecture(adata, L, K)

    # For VNNGP without MGGP, use a simple RBF kernel
    from gpzoo.kernels import batched_RBF
    from gpzoo.gp import VNNGP

    M = N
    V = scanpy_sizefactors(Y.T.numpy())

    Z = nn.Parameter(X.clone(), requires_grad=False)
    kernel = batched_RBF(sigma=1.0, lengthscale=100.0)
    gp = VNNGP(kernel, M=M, jitter=1e-5, K=K)

    mu = torch.randn(L, M)
    gp.mu = nn.Parameter(mu)
    Lu = 1e-2 * torch.randn(L, M, K)
    gp.Lu = nn.Parameter(Lu.clone().detach())
    gp.Z = Z

    model = NSF2(gp, Y, L=L)
    W = torch.randn(D, L)
    model.W = nn.Parameter(W)
    model.V = nn.Parameter(torch.squeeze(torch.tensor(init_softplus(V)).type(torch.float)))

    return model, X, Y, groupsX, n_groups


def create_mggp_vnngp_model(adata, L=12, K=50):
    """Create MGGP_VNNGP model architecture."""
    X, Y, groupsX, n_groups, D, N, L, K = get_model_architecture(adata, L, K)

    M = N
    V = scanpy_sizefactors(Y.T.numpy())

    Z = nn.Parameter(X.clone(), requires_grad=False)
    groupsZ = nn.Parameter(groupsX.clone(), requires_grad=False)
    kernel = batched_MGGP_RBF(sigma=1.0, lengthscale=100.0, group_diff_param=10.0, n_groups=n_groups)
    gp = MGGP_VNNGP(kernel, M=M, n_groups=n_groups, jitter=1e-5, K=K)

    mu = torch.randn(L, M)
    gp.mu = nn.Parameter(mu)
    Lu = 1e-2 * torch.randn(L, M, K)
    gp.Lu = nn.Parameter(Lu.clone().detach())
    gp.Z = Z
    gp.groupsZ = groupsZ

    model = NSF2(gp, Y, L=L)
    W = torch.randn(D, L)
    model.W = nn.Parameter(W)
    model.V = nn.Parameter(torch.squeeze(torch.tensor(init_softplus(V)).type(torch.float)))

    return model, X, Y, groupsX, n_groups


def reorder_nsf_model(model, exp_mean_pred, X):
    """Reorder model factors by Moran's I autocorrelation."""
    moran_idx, moranI = dims_autocorr((exp_mean_pred).T, X.detach().numpy())
    idx = torch.as_tensor(moran_idx, device=model.W.device, dtype=torch.long)
    with torch.no_grad():
        model.prior.mu.copy_(model.prior.mu.index_select(0, idx))
        model.prior.Lu.copy_(model.prior.Lu.index_select(0, idx))
        model.W.copy_(model.W.index_select(1, idx))
    return moranI


def extract_factors(model, X, groupsX=None, n_groups=1):
    """
    Extract factors from trained model.

    Returns factors in LOG SPACE (qF.mean) for better K-means clustering.
    """
    model.cpu()
    model.eval()
    model.prior.K = 50

    knn_idx = model.prior.calculate_knn(X)[:, :-1]
    model.prior.knn_idx = knn_idx

    if hasattr(model.prior, 'Z'):
        knn_idz = model.prior.calculate_knn(model.prior.Z)[:, 1:]
        model.prior.knn_idz = knn_idz

    with torch.no_grad():
        if groupsX is not None:
            qF, _, _ = model.prior.forward(X=X, groupsX=groupsX, verbose=False)
        else:
            qF, _, _ = model.prior.forward(X=X, verbose=False)

        # Return log-space mean for K-means (exp can mislead clustering)
        mean_log = qF.mean.detach().numpy()
        mean_exp = torch.exp(qF.mean).detach().numpy()
        scale = qF.scale.detach().numpy()

    return mean_log, mean_exp, scale


def get_groupwise_factors(model, X, groupsX, n_groups, L=12):
    """
    Get factors for each group separately (for MGGP models).

    Returns factors in LOG SPACE for K-means clustering.
    """
    model.cpu()
    model.eval()

    results = []

    with torch.no_grad():
        knn_idx = model.prior.calculate_knn(X)
        model.prior.knn_idx = knn_idx[:, :-1]

        for group_index in tqdm(range(n_groups), desc="Computing group-wise factors"):
            group_mask = (groupsX == group_index)
            groupsX_test = torch.full_like(groupsX, fill_value=group_index)

            qF_test, _, _ = model.prior(X, groupsX=groupsX_test)

            # Store both log and exp for visualization (but use log for clustering)
            factors_log = qF_test.mean.cpu().numpy()
            factors_exp = torch.exp(qF_test.mean).cpu().numpy()

            # Compute Moran's I on exp factors for visualization ordering
            moran_idx, moranI = dims_autocorr((factors_exp).T, X.detach().numpy())

            del qF_test

            results.append({
                'group_index': group_index,
                'group_mask': group_mask.cpu().numpy(),
                'factors_log': factors_log,  # For K-means
                'factors_exp': factors_exp,  # For visualization
                'moranI': moranI[np.argsort(moran_idx)]
            })

    return results


def generate_nmf_baseline(adata, L=12):
    """Generate NMF baseline factors."""
    import scipy.sparse as sp

    # Get expression matrix (cells x genes)
    if adata.raw:
        Y_cells_genes = adata.raw.X
    else:
        Y_cells_genes = adata.X

    # Convert to dense if sparse
    if sp.issparse(Y_cells_genes):
        Y_cells_genes = Y_cells_genes.toarray()

    # Compute size factors (per cell)
    V = scanpy_sizefactors(Y_cells_genes)

    # Normalize and fit NMF
    nmf_model = NMF(n_components=L, max_iter=300, init='random', random_state=0, alpha_H=0.0, alpha_W=1e-2)
    nmf_model.fit((Y_cells_genes / V))

    exp_factors = nmf_model.transform(Y_cells_genes / V) / 5
    nmf_factors = np.log(exp_factors + 1e-2)

    moran_idx, moranI = dims_autocorr(exp_factors, adata.obsm['spatial'])
    factors_ordered = exp_factors[:, moran_idx]

    return factors_ordered


def generate_pca_baseline(adata, L=12):
    """Generate PCA baseline factors."""
    import scipy.sparse as sp

    # Get expression matrix (cells x genes)
    Y = adata.raw.X if adata.raw else adata.X

    # Convert to dense if sparse (PCA needs dense for this sklearn version)
    if sp.issparse(Y):
        Y = Y.toarray()

    pca_model = PCA(n_components=L, random_state=0)
    pca_factors = pca_model.fit_transform(Y)

    return pca_factors


def process_dataset(dataset_name, data_dir, models_dir, output_dir, L=12, K=50, region_column=REGION_COLUMN):
    """
    Process a single dataset: load data, load models, generate annotations.

    Args:
        dataset_name: Name of dataset (e.g., '151507')
        data_dir: Path to directory containing .h5ad files
        models_dir: Path to directory containing trained models
        output_dir: Path to directory where annotations will be saved
        L: Number of factors
        K: Number of nearest neighbors
        region_column: Column name for region labels (from config.py)
    """
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}\n")

    # Load data
    h5ad_path = data_dir / f"{dataset_name}.h5ad"
    if not h5ad_path.exists():
        print(f"Warning: Data file not found: {h5ad_path}")
        return

    adata = load_adata(h5ad_path, region_column=region_column)
    print(f"Loaded data: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Determine number of clusters from region labels (K = number of unique regions)
    if region_column in adata.obs.columns:
        n_clusters = len(adata.obs[region_column].astype('category').cat.categories)
        print(f"Number of regions (K for K-means): {n_clusters}")
    else:
        n_clusters = 7  # Default
        print(f"No {region_column} column found, using default n_clusters={n_clusters}")

    # Generate PCA baseline
    print("\n--- Generating PCA baseline ---")
    pca_factors = generate_pca_baseline(adata, L=L)
    print(f"  PCA factors shape: {pca_factors.shape} (should be N x L = {adata.shape[0]} x {L})")
    pca_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(pca_factors)

    pca_output = output_dir / f"{dataset_name}_pca.txt"
    np.savetxt(pca_output, pca_labels, fmt='%d', header=f"{dataset_name}_pca", comments='')
    print(f"Saved PCA annotations: {pca_output}")

    # Generate NMF baseline
    print("\n--- Generating NMF baseline ---")
    nmf_factors = generate_nmf_baseline(adata, L=L)
    print(f"  NMF factors shape: {nmf_factors.shape} (should be N x L = {adata.shape[0]} x {L})")
    nmf_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(nmf_factors)

    nmf_output = output_dir / f"{dataset_name}_nmf.txt"
    np.savetxt(nmf_output, nmf_labels, fmt='%d', header=f"{dataset_name}_nmf", comments='')
    print(f"Saved NMF annotations: {nmf_output}")

    # Process VNNGP model
    vnngp_model_path = models_dir / f"{dataset_name}_vnngp_k={K}.pth"
    if vnngp_model_path.exists():
        print("\n--- Processing VNNGP model ---")
        try:
            model, X, Y, groupsX, n_groups = create_vnngp_model(adata, L=L, K=K)
            model.load_state_dict(torch.load(vnngp_model_path))

            # Extract factors (returns log, exp, scale)
            mean_log, mean_exp, scale = extract_factors(model, X, groupsX, n_groups)

            # Use LOG SPACE for K-means (exp can mislead clustering)
            # mean_log is (L, N), we need (N, L) for K-means
            factors_for_kmeans = mean_log.T  # (N, L)
            print(f"  VNNGP factors shape: {factors_for_kmeans.shape} (should be N x L = {adata.shape[0]} x {L})")
            vnngp_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(factors_for_kmeans)

            vnngp_output = output_dir / f"{dataset_name}_vnngp.txt"
            np.savetxt(vnngp_output, vnngp_labels, fmt='%d', header=f"{dataset_name}_vnngp", comments='')
            print(f"Saved VNNGP annotations: {vnngp_output}")

            del model, X, Y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing VNNGP model: {e}")
    else:
        print(f"Warning: VNNGP model not found: {vnngp_model_path}")

    # Process MGGP_VNNGP model
    mggp_model_path = models_dir / f"{dataset_name}_mggp_vnngp_k={K}.pth"
    if mggp_model_path.exists():
        print("\n--- Processing MGGP_VNNGP model ---")
        try:
            model, X, Y, groupsX, n_groups = create_mggp_vnngp_model(adata, L=L, K=K)
            model.load_state_dict(torch.load(mggp_model_path))

            # Overall factors (returns log, exp, scale)
            mean_log, mean_exp, scale = extract_factors(model, X, groupsX, n_groups)

            # Use LOG SPACE for K-means (exp can mislead clustering)
            # mean_log is (L, N), we need (N, L) for K-means
            factors_for_kmeans = mean_log.T  # (N, L)
            print(f"  MGGP_VNNGP factors shape: {factors_for_kmeans.shape} (should be N x L = {adata.shape[0]} x {L})")
            mggp_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(factors_for_kmeans)

            mggp_output = output_dir / f"{dataset_name}_mggp_vnngp.txt"
            np.savetxt(mggp_output, mggp_labels, fmt='%d', header=f"{dataset_name}_mggp_vnngp", comments='')
            print(f"Saved MGGP_VNNGP annotations: {mggp_output}")

            # Group-wise factors (if we have region groups)
            if n_groups > 1 and region_column in adata.obs.columns:
                print(f"Generating group-wise annotations for {n_groups} groups...")
                results = get_groupwise_factors(model, X, groupsX, n_groups, L=L)

                region_labels = adata.obs[region_column].astype('category').cat.categories.to_list()
                for result in results:
                    region_name = region_labels[result['group_index']]

                    # Use LOG SPACE factors for K-means
                    # factors_log is (L, N), we need (N, L) for K-means
                    factors_for_kmeans = result['factors_log'].T  # (N, L)
                    print(f"    MGGP group '{region_name}' factors shape: {factors_for_kmeans.shape} (should be N x L = {adata.shape[0]} x {L})")
                    group_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(factors_for_kmeans)

                    group_output = output_dir / f"{dataset_name}_mggp_vnngp_{region_name}.txt"
                    np.savetxt(
                        group_output, group_labels, fmt='%d',
                        header=f"{dataset_name}_mggp_vnngp_{region_name}", comments=''
                    )
                    print(f"  Saved group '{region_name}': {group_output}")

            del model, X, Y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing MGGP_VNNGP model: {e}")
    else:
        print(f"Warning: MGGP_VNNGP model not found: {mggp_model_path}")


def main():
    """Main function to process all datasets."""
    # Setup paths
    data_dir = Path('/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data')
    models_dir = Path('/gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo/models/tenxvisium')
    output_dir = Path('/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Benchmarks')

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all h5ad files
    h5ad_files = sorted(data_dir.glob('*.h5ad'))

    # Filter to get dataset names (exclude osmfish which is different format)
    dataset_names = []
    for h5ad_file in h5ad_files:
        name = h5ad_file.stem
        if name.startswith('151'):  # 10X Visium datasets start with 151
            dataset_names.append(name)

    print(f"Found {len(dataset_names)} datasets to process:")
    for name in dataset_names:
        print(f"  - {name}")

    # Process each dataset
    for dataset_name in dataset_names:
        try:
            process_dataset(
                dataset_name=dataset_name,
                data_dir=data_dir,
                models_dir=models_dir,
                output_dir=output_dir,
                L=12,
                K=50
            )
        except Exception as e:
            print(f"\nERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("All datasets processed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
