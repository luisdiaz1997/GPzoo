from anndata import AnnData
import scanpy as sc
from os import path
import os
import numpy as np
from gpzoo.kernels import NSF_RBF, RBF
from gpzoo.likelihoods import NSF2
from tqdm.auto import tqdm
from gpzoo.gp import SVGP, VNNGP
from gpzoo.utilities import rescale_spatial_coords, dims_autocorr, regularized_nmf, add_jitter, scanpy_sizefactors, deviancePoisson, anndata_to_train_val
import random
import squidpy as sq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import gpzoo.pri_experiment_utilities as putil
from sklearn.metrics import root_mean_squared_error, mean_poisson_deviance
import torch
from torch import optim, distributions, nn
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import ParameterGrid, KFold, train_test_split
import csv
import argparse

from numba import cuda 
device = cuda.get_current_device()
device.reset()

random.seed(100)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
print("GPU ITEMS")
print("Is CUDA available?", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current device: ",torch.cuda.current_device())
print("Device: ", torch.cuda.device(0))
print("Name: ", torch.cuda.get_device_name(0))

root_path = "/engelhardt/home/pshrestha/vnngp/"
#save_path = /engelhardt/home/pshrestha/vnngp/nnnsf/3000_new_processing'
dpth = path.join(root_path, "data/")
SPATH = path.join(root_path, "results/merfish")

def plot_umap_factors(loadings, ad):
    L = loadings.shape[1]
    size = 2
    fig, axes = plt.subplots(2, 6, figsize=(size * 6, size * 2), tight_layout=True)

    # Find the global vmin and vmax across all factors
    vmin = loadings.min().min()
    vmax = loadings.max().max()

    for i in range(L):
        curr_ax = axes[i // 6, i % 6]
        sc.pl.embedding(
            ad,
            basis='umap_har',
            color=f'Factor_{i+1}',
            ax=curr_ax,
            show=False,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax  # Set the global color limits
        )

    plt.tight_layout()
    plt.show()
    return fig, axes


def run_experiment(X, Y, save_path, ad, NMF=True, batched=True):
    # Define the parameter grid
    param_grid = {
        'sigma': [0.3],
        'lengthscale': [0.02],
        'learning_rate': [0.01],
        'iterations': [5000],
        'M': [3000],
        'K': [2, 4, 8],
        'L': [6, 12, 18],
    }

    # Specify the file path
    csv_file_path = path.join(save_path, 'nnnsf_validation.csv')
    i = 0
    
    # Check if the file exists
    file_exists = os.path.isfile(csv_file_path)

    for params in ParameterGrid(param_grid):
        kwargs = {
                        'L': params['L'], 
                        'M': params['M'], # use total number of inducing points
                        'sigma': params['sigma'], 
                        'lengthscale': params['lengthscale'], 
                        'jtr': 1e-1, # no lower
                        'batch_size': 128,
                        'lr': params['learning_rate'],
                        'rs': 256,
                        'lkzz_build': 1,
                        'model': 'VNNGP',
                        'L1_jitter': 1e-1, # no lower
                        'iterations': params['iterations'],
                        'K': params['K'],
        }
        
        
        file_path = kwargs['model']
        if kwargs['model'] == 'VNNGP':
            file_path += f"_K={kwargs['K']}"
            if kwargs['lkzz_build']:
                file_path += f"_lkzz={kwargs['lkzz_build']}"
        file_path += f"_M={kwargs['M']}_L={kwargs['L']}_lr={kwargs['lr']}_jtr={kwargs['jtr']}_ls={kwargs['lengthscale']}_sigma={kwargs['sigma']}"
    
        X_array = None
        Y_array = None
                
        if batched:
            file_path += f"_bs={kwargs['batch_size']}"
                
        factors = None
        loadings = None
        if NMF:
            file_path += f"_NMFinit"
            nmf_save_path = path.join(SPATH, 'nmf')
            factors_path = path.join(nmf_save_path, f"3000/nmf_factors_iter=1000_rs=256_L={kwargs['L']}_rm(20.8, 22.6, 3.33)_new.npy")
            loadings_path = path.join(nmf_save_path, f"3000/nmf_loadings_iter=1000_rs=256_L={kwargs['L']}_rm(20.8, 22.6, 3.33)_new.npy")
            factors = np.load(factors_path)
            loadings = np.load(loadings_path)
            print("factors shape: ", factors.shape)
            print("loadings shape: ", loadings.shape)
    
    
        model_state_path = f'{save_path}/{file_path}_state_dict.pth'
        model_info_path = f'{save_path}/{file_path}_model.pt'
    
        if path.exists(model_state_path) and path.exists(model_info_path):
            print(f"Loading existing model from {model_state_path} and continuing training.")  
            continue
            saved_info = torch.load(model_info_path)
            losses = saved_info.get('losses', [])
            means = saved_info.get('means', [])
            scales = saved_info.get('scales', [])
            final_time = saved_info.get('time', 0)
        else:
            print(f"Training a new model and saving to {model_state_path}.")
            model = putil.build_model(np.array(X), np.array(Y), loadings=loadings, factors=factors, kwargs=kwargs)
            losses, means, scales, final_time = [], [], [], 0
        
        putil.model_grads(model)
        model.prior.jitter = kwargs['jtr']
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
        model.to(device)
        X_torch = torch.tensor(X).type(torch.float).to(device)
        Y_torch = torch.tensor(Y).type(torch.float).to(device)
        
        start_time = time.time()
        more_losses, more_means, more_scales, idxs = putil.train_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=kwargs['iterations'], E=3,kwargs=kwargs)
        end_time = time.time()
        final_time += (end_time - start_time)
    
        # combine new and previous losses, means, scales
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
            val_loss = putil.evaluate_model(model, X_torch.cpu(), Y, device, kwargs=kwargs)
            
        fig, ax = plt.subplots()
        ax.plot(losses)
        fig.suptitle(f"{kwargs['model']} Loss | sigma: {kwargs['sigma']}, lengthscale: {kwargs['lengthscale']}")
        plt.show()
        fig.savefig(f'{save_path}/{file_path}_loss.png')
    
        #model.cpu()
        #qF, _, _ = model.prior(X.cpu(), kwargs=kwargs)
        #mean = torch.exp(qF.mean).detach().numpy()
    
        #ad.obsm['X_nnnsf'] = mean.T
        #sc.pp.neighbors(ad, use_rep='X_nnnsf')
        #sc.tl.umap(ad)
    
        #for i in range((mean.T).shape[1]):
            #ad.obs[f'Factor_{i+1}'] = (mean.T)[:, i]
        #fig, ax = plot_umap_factors(loadings, ad)
        #fig.suptitle(f'Factors | sigma: {kwargs["sigma"]}, lengthscale: {kwargs["lengthscale"]}')
        #fig.savefig(f'{save_path}/{file_path}_plot.png')
    
        with torch.no_grad():                
            X_torch = torch.tensor(X).type(torch.float)
            Y_torch = torch.tensor(Y).type(torch.float)
            rmse = putil.evaluate_model(model, X_torch.cpu(), Y, device, evaluation_metric=root_mean_squared_error, kwargs=kwargs)
            poi = putil.evaluate_model(model, X_torch.cpu(), Y, device, evaluation_metric=mean_poisson_deviance, kwargs=kwargs)
        
        print(f'Params: {params}, RMSE: {rmse}, POI: {poi}')
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            #if not file_exists:
                #writer.writerow(["L", "M", "K", "sigma", "ls", "rmse", "poi"]) 
                #print("Header added to CSV file")
            writer.writerow([kwargs['L'], kwargs['M'], kwargs['K'], kwargs['sigma'], kwargs['lengthscale'], rmse, poi])
        print("Row written to CSV")
            


def main(save_path):
    merpath = path.join(dpth, "merfish.h5ad")
    ad = sc.read_h5ad(merpath)
    ad = ad[(ad.obs['percent.mt'] < 5), :]
    min_genes_per_cell = 500
    ad = ad[ad.obs['nFeature_RNA'] >= min_genes_per_cell, :]
    min_value = np.min(ad.X)
    if min_value < 0:
        ad.X += np.abs(min_value)
    ad.var_names_make_unique()
    labels = ad.obs.Age.unique()
    ad.obs['Age_cat'] = ad.obs['Age'].astype('category')
    ad = ad[~ad.obs['Age'].isin([20.8, 22.6, 3.33])]
    Y_sums = np.array(np.sum(ad.raw.X > 0, axis=0))[0]
    Y = np.array(ad.X, dtype=int).T
    X = ad.obsm['X_harmony']
    X = X.astype('float32')
    Y = Y.astype('float32')
    #X = rescale_spatial_coords(X)
    ad.uns['counts'] = ad.X.copy()
    
    ad.var['deviance_poisson'] = deviancePoisson(ad.uns["counts"])
    o = np.argsort(-ad.var['deviance_poisson'])
    idx = list(range(ad.shape[0]))
    random.shuffle(idx)
    ad = ad[idx,o]
    
    X = np.array(ad.obs['Age'])[:, None]
    X = torch.tensor(X, dtype=torch.float)
    
    Y = np.log(np.exp(Y) + 1e-2)  # correction
    #Y = Y.T

    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)
    
    #save_path = path.join(SPATH, 'nnnsf/3000_new_processing')
    run_experiment(X, Y, save_path, ad)
    print("DONE!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run experiment with specified save path.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the experiment results")
    args = parser.parse_args()
    main(args.save_path)

