import numpy as np
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import torch
from gpzoo.utilities import train, deviancePoisson, rescale_spatial_coords, anndata_to_train_val, regularized_nmf
import gpzoo.pri_experiment_utilities as putil
from gpzoo.gp import SVGP, VNNGP, GaussianPrior
from gpzoo.likelihoods import PNMF
import scanpy as sc
from os import path
import random
from copy import deepcopy
from scipy import sparse
from contextlib import suppress
from scanpy import read_h5ad
from tensorflow import constant
from tensorflow.data import Dataset
from torch import optim, distributions, nn
from sklearn.metrics import root_mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import ast

import gseapy as gps
from gseapy.plot import barplot, dotplot

import warnings
warnings.filterwarnings("ignore") 


def remake_original(model, X, **kwargs):
    '''
    parameters:
    - models: trained model instance
    '''
    kwargs = kwargs['kwargs']
    with torch.no_grad():
        model.cpu()
        #try: 
        qF, _, _ = model.prior(X, kwargs=kwargs)
        means = torch.exp(qF.mean).detach().numpy() # means = factors
        W = (model.W).cpu()
        #except: # hybrid models
           # qF, _, _ = model.sf.prior(X, kwargs=kwargs)
           # means = torch.exp(qF.mean).detach().numpy() # means = factors
           # W = (model.sf.W).cpu()
        W_transformed = nn.functional.softplus(W.T)
        W_transformed = W_transformed.detach().numpy()
        y_nmf = ((means.T)).dot(W_transformed)
    return y_nmf


def evaluate_model(model, X, Y, device, evaluation_metric=root_mean_squared_error, **kwargs):
    model.eval()  # Set the model to evaluation mode
    kwargs = kwargs['kwargs']
    y_nmf = remake_original(model, X, kwargs=kwargs)
    
    # Check for invalid values (NaNs or Infs) in y_nmf
    y_nmf = np.nan_to_num(y_nmf, nan=0.0, posinf=0.0, neginf=0.0)
    if Y.shape != y_nmf.shape:
        y_nmf = y_nmf.T

    try:
        Y = putil.check_array(Y, ensure_2d=False, force_all_finite='allow-nan')
        y_nmf = putil.check_array(y_nmf, ensure_2d=False, force_all_finite='allow-nan')
    except ValueError:
        # If invalid values are found, apply standard scaling
        scaler = StandardScaler()
        Y = scaler.fit_transform(Y.reshape(-1, 1)).flatten()
        y_nmf = scaler.fit_transform(y_nmf.reshape(-1, 1)).flatten()
    
    # Calculate the reconstruction error using the evaluation metric
    try:
        reconstruction_error = evaluation_metric(Y, y_nmf)
    except:
        reconstruction_error = evaluation_metric(Y.flatten(), y_nmf.flatten())

    return reconstruction_error

def calculate_eval_metric(X, Y, param_dict, datapath, eval_function=root_mean_squared_error, 
                          train=False, hybrid=False, model_type='VNNGP'):
    """
    General function to evaluate models for both VNNGP and SVGP model types based on param_dict.
    
    Parameters:
    - X: Input features.
    - Y: Target values.
    - param_dict: Dictionary containing hyperparameters (L, M, K, sigma, lengthscale, etc.).
    - datapath: Directory path to load/save model files.
    - eval_function: Evaluation function for model performance.
    - train: If True, evaluate on training and validation sets.
    - hybrid: If True, use a hybrid model.
    - model_type: Model type to specify ('VNNGP' or 'SVGP').
    
    Returns:
    - data_dict: Dictionary containing evaluation metrics (RMSE) for different combinations of L, M, and K.
    """
    
    if train:
        X_train, X_val, y_train, y_val = train_test_split(X, Y.T, test_size=0.05, random_state=256)
    
    L = param_dict['L']
    M = param_dict['M']
    K = param_dict.get('K', [None])  # K may not be used for SVGP
    data_dict = {str(l): [] for l in L}
    
    for l in L:
        inducing_point_rmse = []
        for m in M:
            neighbor_rmse = []
            for k in (K if model_type == 'VNNGP' else [None]):
                
                # Model parameter dictionary
                dicts = {
                    'L': l, 
                    'M': m, 
                    'K': k,  # VNNGP specific, not used for SVGP
                    'sigma': param_dict['sigma'], 
                    'lengthscale': param_dict['lengthscale'], 
                    'jtr': param_dict['jtr'],
                    'batch_size': 128,
                    'lr': param_dict['lr'],
                    'rs': 256,
                    'lkzz_build': 1,
                    'model': param_dict['model'],
                    'L1_jitter': param_dict['L1_jitter']
                }

                # Load NMF factors and loadings
                nmf_save_path = path.join(datapath, 'nmf')
                factors_file = f"nmf_factors_iter=1000_rs=256_L={l}.npy"
                loadings_file = f"nmf_loadings_iter=1000_rs=256_L={l}.npy"

                if train:
                    factors_file = f"train_{factors_file}"
                    loadings_file = f"train_{loadings_file}"
                
                factors = np.load(path.join(nmf_save_path, factors_file))
                loadings = np.load(path.join(nmf_save_path, loadings_file))
                
                # Build model based on the type and whether hybrid or not
                if not hybrid:
                    if train:
                        model = putil.build_model(X_train, y_train.T, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                    else:
                        model = putil.build_model(X, Y, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                else:
                    model = putil.build_model_hybrid(X, Y, loadings=loadings, factors=factors, model_type=dicts['model'], kwargs=dicts)
                
                # Define model file path based on the type of model
                if model_type == 'VNNGP':
                    modelpath = path.join(datapath, 'nnnsf')
                    filepath = f"VNNGP_K={k}_lkzz=1_M={m}_L={l}_lr={dicts['lr']}_jtr={dicts['jtr']}_ls={dicts['lengthscale']}_sigma={dicts['sigma']}_bs=128_NMFinit_state_dict.pth"
                else:  # model_type == 'SVGP'
                    modelpath = path.join(datapath, 'nsf')
                    filepath = f"SVGP_M={m}_L={l}_lr={dicts['lr']}_jtr={dicts['jtr']}_ls={dicts['lengthscale']}_sigma={dicts['sigma']}_bs={dicts['batch_size']}_NMFinit_state_dict.pth"
                
                dictpath = path.join(modelpath, filepath)
                print(dictpath)

                if not path.exists(dictpath):
                    print("Does not exist.")
                    neighbor_rmse.append(np.nan)
                    continue

                # Load model state
                model.load_state_dict(torch.load(dictpath))

                # Evaluate model
                if train:
                    X_train = torch.tensor(X_train).type(torch.float)
                    X_val = torch.tensor(X_val).type(torch.float)
                    train_loss = putil.evaluate_model(model, X_train.cpu(), y_train.T, device, evaluation_metric=eval_function,
                                  kwargs=dicts)
                
                    val_loss = putil.evaluate_model(model, X_val.cpu(), y_val.T, device, evaluation_metric=eval_function,
                                  kwargs=dicts)
                    loss = (train_loss, val_loss)
                else:
                    X = torch.tensor(X).type(torch.float)
                    X = X.clone().detach().type(torch.float)
                    loss = evaluate_model(model, X.cpu(), Y.T, device, evaluation_metric=eval_function,
                                  kwargs=dicts)

                print(f"With {l} factors and {m} neighbors/IPs: ", loss)
                neighbor_rmse.append(loss)

            inducing_point_rmse.append(neighbor_rmse)
        data_dict[str(l)] = inducing_point_rmse
        print(f"DONE with factor {l}")
    
    return data_dict


adata = sq.datasets.slideseqv2()
adata = adata.raw.to_adata()
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs.pct_counts_mt < 20] #from 53K to 45K
sc.pp.filter_cells(adata, min_counts=100)
sc.pp.filter_genes(adata, min_cells=10)

Dtr, Dval = anndata_to_train_val(adata, sz="scanpy")
y_train = Dtr['Y'].T
y_train = y_train[~adata.var.MT]
X_train = Dtr['X']*50
V_train = Dtr['sz']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(256)
random_seed = 256

ROOT_PATH = '/engelhardt/home/pshrestha/vnngp/'
RPATH = path.join(ROOT_PATH, "results/")
SPATH = path.join(RPATH, "slideseq/")
FPATH = path.join(SPATH, "plots/")
nnnsf_path = path.join(SPATH, "nnnsf/")
nsf_path = path.join(SPATH, "nsf/")

inducing_points = [50, 1000, 2000, 3000, 4000, 5000]
K = [2, 4, 8]

params = {
        'L': [10], 
        'K': [2, 4, 8],
        'M': inducing_points, 
        'sigma': 1.0, 
        'lengthscale': 1.2, 
        'jtr': 0.01,
        'batch_size': 128,
        'lr': 1e-3,
        'rs': 256,
        'lkzz_build': 1,
        'model': 'SVGP',
        'L1_jitter': 1e-1}
eval_function = mean_poisson_deviance
torch.cuda.empty_cache()
X = torch.tensor(X_train)
Y = np.array(y_train)
nsf_k_experiments_poi = calculate_eval_metric(X, Y, params, SPATH, eval_function=eval_function, train=False, hybrid=False, model_type='SVGP')

print(nsf_k_experiments_poi)