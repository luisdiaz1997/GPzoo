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
from tqdm.auto import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(256)
random_seed = 256

ROOT_PATH = '/engelhardt/home/pshrestha/vnngp/'
RPATH = path.join(ROOT_PATH, "results/")
SPATH = path.join(RPATH, "slideseq/")
FPATH = path.join(SPATH, "figs/")

def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1, names=None):

    max_val = np.percentile(factors, 99)
    min_val = np.percentile(factors, 1)

    
    if moran_idx is not None:
        factors = factors[moran_idx]
        if names is not None:
            names = names[moran_idx]

    L = len(factors)

    if ax is None:
        fig, ax = plt.subplots(L//6, 6, figsize=(size*6, size*(L//6)), tight_layout=True)
        
    for i in range(L):
        
        
        curr_ax = ax[i//6, i%6]
        
        
        
        curr_ax.scatter(X[:, 0], X[:,1], c=factors[i], vmin=min_val, vmax=max_val, alpha=alpha, cmap='turbo', s=s)

        curr_ax.invert_yaxis()
        if names is not None:
            curr_ax.set_title(names[i], x=0.03, y=.88, fontsize="small", c="white",
                     ha="left", va="top")
            
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_facecolor('xkcd:gray')
    return fig, ax

def run_experiment(X, Y, save_path, NMF=True, batched=True):
    # Define the parameter grid
    param_grid = {
        'sigma': [1.0],
        'lengthscale': [1.7],
        'learning_rate': [0.0001],
        'L': [12],
        'M': [4000],
        'K': [8]
    }
    
   
    nsf_path = path.join(SPATH, 'nnnsfh/')
    csv_path = path.join(nsf_path, 'nnnsfh_csv_results.csv')

    file_exists = path.isfile(csv_path)
    if not file_exists:
        headers = ['learning_rate', 'sigma', 'lengthscale', 'K', 'ips', 'mean_cv_loss']
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
                
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
    
        # Iterate over each parameter combination
        for params in ParameterGrid(param_grid):
            kwargs = {
                'L': params['L'], 
                'M': params['M'], # use total number of inducing points
                'sigma': params['sigma'], 
                'lengthscale': params['lengthscale'], 
                'jtr': 0.01, # no lower
                'batch_size': 128,
                'lr': params['learning_rate'],
                'rs': 256,
                'lkzz_build': 1,
                'model': 'VNNGP',
                'L1_jitter': 1e-1, # no lower
                'K': params['K'],
            }
        
            file_path = kwargs['model']
            if kwargs['model'] == 'VNNGP':
                file_path += f"_K={kwargs['K']}"
            file_path += f"_M={kwargs['M']}_L={kwargs['L']}_lr={kwargs['lr']}_jtr={kwargs['jtr']}_ls={kwargs['lengthscale']}_sigma={kwargs['sigma']}"
 
                
            if batched:
                file_path += f"_bs={kwargs['batch_size']}_2"
    
            # check of model and its state dictionary already exist
            model_state_path = f'{save_path}/{file_path}_state_dict.pth'
            model_info_path = f'{save_path}/{file_path}_model.pt'
            if path.exists(model_state_path) and path.exists(model_info_path):
                print("Model already trained. Skipping.")
                continue
            

            nmf_save_path = path.join(SPATH, 'nmf')
            factors_path = path.join(nmf_save_path, f"nmf_factors_iter=1000_rs=256_L=23.npy")
            loadings_path = path.join(nmf_save_path, f"nmf_loadings_iter=1000_rs=256_L=23.npy")
            factors = np.load(factors_path)
            loadings = np.load(loadings_path)
            moran_idx, moranI = putil.dims_autocorr(factors, np.array(X))
            factors = factors[:, moran_idx]
            loadings = loadings[:, moran_idx]
            #factors = factors[:, :kwargs['L']]
            #loadings = loadings[:, :kwargs['L']]
            print(f"Training a new model and saving to {model_state_path}.")
            print(X.shape)
            print(Y.shape)
            print(loadings.shape)
            print(factors.shape)
            model = putil.build_model_hybrid(np.array(X), np.array(Y), loadings=loadings, factors=factors, kwargs=kwargs)
            losses, means, scales, final_time = [], [], [], 0
        
            model.sf.prior.jitter = kwargs['jtr']
            model.cf.prior.scale_pf = 1e-1
            model.to(device)
            X_torch = torch.tensor(X).type(torch.float).to(device)
            Y_torch = torch.tensor(Y).type(torch.float).to(device)

            # 0 (9hrs)
            print("Training spatial and nonspatial variances.")
            putil.model_grads_hybrid0(model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            start_time = time.time()
            more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=100*360*3, E=3, batch_size=kwargs['batch_size'],
                                                                             kwargs=kwargs) #100*360*3
            end_time = time.time()
            final_time += (end_time - start_time)
            losses.extend(more_losses)
            
            # 1 (6hrs)
            print("Training spatial and nonspatial means.")
            putil.model_grads_hybrid1(model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            start_time = time.time()
            more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=100*360*2, E=3, batch_size=kwargs['batch_size'],
                                                                             kwargs=kwargs)#100*360*2
            end_time = time.time()
            final_time += (end_time - start_time)
            losses.extend(more_losses)

            # 2 (6hrs)
            print("Training spatial mean and variance and nonspatial variance.")
            putil.model_grads_hybrid2(model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            start_time = time.time()
            more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=100*360, E=3, batch_size=kwargs['batch_size'],
                                                                             kwargs=kwargs)#100*360
            end_time = time.time()
            final_time += (end_time - start_time)
            losses.extend(more_losses)

            # 3 (6hrs)
            print("Training spatial mean, variance, loading and nonspatial variance")
            putil.model_grads_hybrid3(model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            start_time = time.time()
            more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=100*360*2, E=3, batch_size=kwargs['batch_size'],
                                                                             kwargs=kwargs) #100*360*2
            end_time = time.time()
            final_time += (end_time - start_time)
            losses.extend(more_losses)

            # 4 (6hrs)
            print("Training nonspatial mean, variance, loadings.")
            putil.model_grads_hybrid4(model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            start_time = time.time()
            more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=100*360*2, E=3, batch_size=kwargs['batch_size'],
                                                                             kwargs=kwargs)#100*360*2
            end_time = time.time()
            final_time += (end_time - start_time)
            losses.extend(more_losses)

            # 5 (6hrs)
            print("Training everything.")
            putil.model_grads_hybrid5(model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kwargs['lr'])
            start_time = time.time()
            more_losses, more_means, more_scales, idxs = putil.train_hybrid_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                             steps=100*360*2, E=3, batch_size=kwargs['batch_size'],
                                                                             kwargs=kwargs)#100*360*2
            end_time = time.time()
            final_time += (end_time - start_time)
            losses.extend(more_losses)

            fig1, ax1 = plt.subplots()
            ax1.plot(losses)
            fig1.suptitle(f"{kwargs['model']} Loss | sigma: {params['sigma']}, lengthscale: {params['lengthscale']}")
            fig1.savefig(f'{save_path}/{file_path}_loss.png')
            plt.close()
        
            model.cpu()
            qF, _, _ = model.sf.prior(X_torch.cpu(), kwargs=kwargs)
            mean = torch.exp(qF.mean).detach().numpy()

            torch.save(model.state_dict(), model_state_path)
            torch.save({
                    'losses': losses,
                    'means': means,
                    'scales': scales,
                    'time': final_time
            }, model_info_path)
        
            with torch.no_grad():
                val_loss = putil.evaluate_model_hybrid(model, X_torch.cpu(), Y, device, kwargs=kwargs)
                
            mean_cv_loss = np.mean(val_loss)
            print(f'Params: {params}, Mean CV Loss: {mean_cv_loss}')
            
            if kwargs['L'] == 5:    
                fig2, ax = putil.plot_factors_five(mean[:, :kwargs['L']], X_torch.cpu().detach().numpy(), moran_idx=moran_idx, size=2, s=1, alpha=1, ax=None)
            else:
                print(f"mean: {mean[:, :kwargs['L']].shape}")
                print(X_torch.shape)
                fig2, ax = plot_factors(mean[:kwargs['L'], :], X_torch.cpu().detach().numpy(), moran_idx=moran_idx, size=2, s=1, alpha=1, ax=None)
            
            #fig2.suptitle("NNNSFH", size=15)
            #fig2.set_figheight(2.25)
            fig2.tight_layout()
            fig2.savefig(f'{save_path}/{file_path}_factors.png')
            plt.close()
            model.to(device)

def main():
    adata = sq.datasets.slideseqv2()
    adata = adata.raw.to_adata()
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 20] #from 53K to 45K
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=10)

    Dtr, Dval = anndata_to_train_val(adata, sz="scanpy")
    Y_train = Dtr['Y'].T
    Y_train = Y_train[~adata.var.MT]
    X_train = Dtr['X']*50
    V_train = Dtr['sz']
    
    X_train = torch.tensor(X_train).type(torch.float)
    Y_train = torch.tensor(Y_train).type(torch.float)
    print(f"X_train.shape {X_train.shape}")
    save_path = path.join(SPATH, 'nnnsfh/validation_experiments')
    run_experiment(X_train, Y_train, save_path)
    print("DONE!!")


if __name__ == '__main__':
    main()
    



    
    
        

