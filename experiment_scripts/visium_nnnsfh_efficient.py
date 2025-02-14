import csv
import gpzoo.pri_experiment_utilities as putil
import matplotlib.pyplot as plt
import numpy as np
import random
import scanpy as sc
import squidpy as sq
import time
import torch

from gpzoo.gp import SVGP, VNNGP
from gpzoo.kernels import NSF_RBF, RBF
from gpzoo.likelihoods import GaussianLikelihood, NSF2
from gpzoo.utilities import rescale_spatial_coords, dims_autocorr, regularized_nmf, add_jitter, scanpy_sizefactors, deviancePoisson, anndata_to_train_val
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from os import path
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from torch import nn, distributions, optim
from torch.distributions import Normal
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(256)
random_seed = 256

ROOT_PATH = '/engelhardt/home/pshrestha/vnngp/'
RPATH = path.join(ROOT_PATH, "results/")
SPATH = path.join(RPATH, "visium/")
FPATH = path.join(SPATH, "figs/")

def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)
    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(size*4, size), tight_layout=True)
        
    for i in range(L):
        plt.subplot(1, 4, i+1)
        curr_ax = ax[i]
        max_val = np.percentile(factors[i], 90)
        min_val = np.percentile(factors[i], 10)
        curr_ax.scatter(X[:, 0], X[:,1], c=factors[i], vmin=min_val, vmax=max_val, alpha=alpha, cmap='Blues', s=s)
        curr_ax.invert_yaxis()
        
        if names is not None:
            curr_ax.set_title(names[i], x=0.03, y=.88, fontsize="small", c="white",
                     ha="left", va="top")
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_facecolor('xkcd:gray')
    return fig

def run_experiment(X, Y, save_path, NMF=True, batched=True):
    # Define the parameter grid
    param_grid = {
        'sigma': [1.0],
        'lengthscale': [1.0],
        'learning_rate': [1e-3],
        'iterations': [1000, 3000],  # Reduced iterations for memory optimization
        'L': [5, 15, 10],
        'M': [50, 1000, 2000, len(X)],
        'K': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 500]
    }
    
    nsf_path = path.join(SPATH, 'nnnsfh/')
    csv_path = path.join(nsf_path, 'synthetic_nsf_csv_results.csv')

    # Check if the CSV file exists and create it if not
    file_exists = path.isfile(csv_path)
    if not file_exists:
        headers = ['iterations', 'learning_rate', 'sigma', 'lengthscale', 'K', 'mean_cv_loss']
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
                
    scaler = GradScaler()

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        # Iterate over each parameter combination
        for params in ParameterGrid(param_grid):
            dicts = {
                'L': params['L'], 
                'M': params['M'],  # Use total number of inducing points
                'sigma': params['sigma'], 
                'lengthscale': params['lengthscale'], 
                'jtr': 0.1,  # No lower
                'batch_size': 64,  # Reduced batch size
                'lr': params['learning_rate'],
                'rs': 256,
                'lkzz_build': 1,
                'model': 'VNNGP',
                'L1_jitter': 1e-1,  # No lower
                'iterations': params['iterations'],
                'K': params['K'],
            }
        
            kwargs = dicts
        
            file_path = kwargs['model']
            if kwargs['model'] == 'VNNGP':
                file_path += f"_K={kwargs['K']}"
                if kwargs['lkzz_build']:
                    file_path += f"_lkzz={kwargs['lkzz_build']}"
            file_path += f"_M={kwargs['M']}_L={kwargs['L']}_lr={kwargs['lr']}_jtr={kwargs['jtr']}_ls={kwargs['lengthscale']}_sigma={kwargs['sigma']}"
    
            if batched:
                file_path += f"_bs={kwargs['batch_size']}"
                
            if NMF:
                file_path += f"_NMFinit"
    
            # Check if model and its state dictionary already exist
            model_state_path = f'{save_path}/{file_path}_state_dict.pth'
            model_info_path = f'{save_path}/{file_path}_model.pt'
    
            if path.exists(model_state_path) and path.exists(model_info_path):
                print(f"Hallelujah {model_state_path} already exists.")
                continue
            else:
                nmf_save_path = path.join(SPATH, 'nmf')
                factors_path = path.join(nmf_save_path, f"train_nmf_factors_iter=1000_rs=256_L={kwargs['L']}.npy")
                loadings_path = path.join(nmf_save_path, f"train_nmf_loadings_iter=1000_rs=256_L={kwargs['L']}.npy")
                factors = np.load(factors_path)
                loadings = np.load(loadings_path)
                moran_idx, moranI = putil.dims_autocorr(factors, np.array(X))
                factors = factors[:, moran_idx]
                loadings = loadings[:, moran_idx]
                print(f"Training a new model and saving to {model_state_path}.")
                model = putil.build_model_hybrid(np.array(X), np.array(Y), loadings=loadings, factors=factors, kwargs=kwargs)
                model.to(device)
        
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            X_torch = torch.tensor(X).type(torch.float).to(device)
            Y_torch = torch.tensor(Y).type(torch.float).to(device)
        
            start_time = time.time()
            with autocast():
                more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(
                    model, optimizer, X_torch, Y_torch, device,
                    steps=kwargs['iterations'], E=3, batch_size=kwargs['batch_size'], kwargs=kwargs
                )
            scaler.scale(more_losses).backward()
            scaler.step(optimizer)
            scaler.update()

            end_time = time.time()
            final_time = (end_time - start_time)

            # Combine new and previous losses, means, scales
            losses, means, scales = [], [], []
            losses.extend(more_losses)
            means.extend(more_means)
            scales.extend(more_scales)
            
            # Train non-spatial components
            model.sf.prior.kernel.lengthscale.requires_grad = False
            model.sf.prior.Z.requires_grad = False
            model.sf.prior.mu.requires_grad = False
            model.sf.prior.Lu.requires_grad = False
            model.sf.W.requires_grad = False
            model.cf.W.requires_grad = True
            model.cf.prior.mean.requires_grad = True

            start_time = time.time()
            with autocast():
                more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(
                    model, optimizer, X_torch, Y_torch, device,
                    steps=kwargs['iterations'], E=3, batch_size=kwargs['batch_size'], kwargs=kwargs
                )
            scaler.scale(more_losses).backward()
            scaler.step(optimizer)
            scaler.update()
            end_time = time.time()
            final_time += (end_time - start_time)

            # Combine new and previous losses, means, scales
            losses.extend(more_losses)
            means.extend(more_means)
            scales.extend(more_scales)
    
            torch.save(model.state_dict(), model_state_path)
            torch.save({
                    'losses': losses,
                    'means': means,
                    'scales': scales,
                    'time': final_time
            }, model_info_path)
        
            with torch.no_grad():
                X_torch = torch.tensor(X).type(torch.float)
                Y_torch = torch.tensor(Y).type(torch.float)
                val_loss = putil.evaluate_model_hybrid(model, X_torch.cpu(), Y, device, kwargs=kwargs)
                
            mean_cv_loss = np.mean(val_loss)
            print(f'Params: {params}, Mean CV Loss: {mean_cv_loss}')
        
            # Write the results to the CSV file
            writer.writerow([params['iterations'], params['learning_rate'], params['sigma'], params['lengthscale'], params['K'], mean_cv_loss])

            # Clear GPU memory
            del X_torch, Y_torch
            torch.cuda.empty_cache()


def main():
    X, Y = putil.load_visium()
    print(X.shape)
    print(Y.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, Y.T, test_size=0.05, random_state=256)  # 95% training, 5% validation
    save_path = path.join(SPATH, 'nnnsfh/validation_experiments')
    run_experiment(X_train, y_train.T, save_path)
    print("done")


if __name__ == '__main__':
    main()
    



    
    
        

