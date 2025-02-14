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

def evaluate_model(model, X, Y, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        model.cpu()
        qF, _, _ = model.prior(X)
        means = torch.exp(qF.mean).detach().numpy() # means = factors
        W = (model.W).cpu()
        W_transformed = nn.functional.softplus(W.T)
        W_transformed = W_transformed.detach().numpy()
        y_nmf = ((means.T)).dot(W_transformed)
        reconstruction_error = root_mean_squared_error(Y, y_nmf.T)
        
    return reconstruction_error

def run_experiment(X, Y, save_path, NMF=True, batched=True):
    # Define the parameter grid
    param_grid = {
        'sigma': [1.0],
        'lengthscale': [1.2],
        'learning_rate': [1e-3],
        'iterations': [5000],
        'L': [5, 15],
        'M': [50, 1000, 2000, 3000, 4000, 5000],
    }
   
    #csv_path = path.join(nsf_path, 'synthetic_nsf_csv_results.csv')

    #file_exists = path.isfile(csv_path)
    #if not file_exists:
        #headers = ['iterations', 'learning_rate', 'sigma', 'lengthscale', 'mean_cv_loss']
        #with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            #writer = csv.writer(file)
            #writer.writerow(headers)
                
    #with open(csv_path, mode='a', newline='') as file:
        #writer = csv.writer(file)
    
        # Iterate over each parameter combination
    for params in ParameterGrid(param_grid):
        dicts = {
                    'L': params['L'], 
                    'M': params['M'], # use total number of inducing points
                    'sigma': params['sigma'], 
                    'lengthscale': params['lengthscale'], 
                    'jtr': 0.01, # no lower
                    'batch_size': 128,
                    'lr': params['learning_rate'],
                    'rs': 256,
                    'lkzz_build': 1,
                    'model': 'SVGP',
                    'L1_jitter': 1e-1, # no lower
                    'iterations': params['iterations'],
        }
    
        kwargs = dicts
    
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
            #X_array = np.array(X)
            #Y_array = np.array(Y)
            nmf_save_path = path.join(SPATH, 'nmf')
            factors_path = path.join(nmf_save_path, f"nmf_factors_iter=1000_rs=256_L={dicts['L']}.npy")
            loadings_path = path.join(nmf_save_path, f"nmf_loadings_iter=1000_rs=256_L={dicts['L']}.npy")
            factors = np.load(factors_path)
            loadings = np.load(loadings_path)
            moran_idx, moranI = putil.dims_autocorr(factors, np.array(X))
            factors = factors[:, moran_idx]
            loadings = loadings[:, moran_idx]

        # check of model and its state dictionary already exist
        model_state_path = f'{save_path}/{file_path}_state_dict.pth'
        model_info_path = f'{save_path}/{file_path}_model.pt'

        if path.exists(model_state_path) and path.exists(model_info_path):
            print(f"Loading existing model from {model_state_path} and continuing training.")
            continue
            #model = putil.build_model(np.array(X), np.array(Y), loadings=loadings, factors=factors, kwargs=kwargs)
            #model.load_state_dict(torch.load(model_state_path))
                
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
                                                                         steps=dicts['iterations'], E=3,kwargs=kwargs)
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
            
        mean_cv_loss = np.mean(val_loss)
        print(f'Params: {params}, Mean CV Loss: {mean_cv_loss}')
    
        #write the results to the CSV file
        #writer.writerow([params['iterations'], params['learning_rate'], params['sigma'], params['lengthscale'], mean_cv_loss])

        fig1, ax1 = plt.subplots()
        ax1.plot(losses)
        fig1.suptitle(f"{kwargs['model']} Loss | sigma: {params['sigma']}, lengthscale: {params['lengthscale']}")
        fig1.savefig(f'{save_path}/{file_path}_loss.png')
        plt.close()
    
        #model.cpu()
        #qF, _, _ = model.prior(X.cpu(), kwargs=kwargs)
        #mean = torch.exp(qF.mean).detach().numpy()
            
        #fig2, ax = putil.plot_factors(mean, X.cpu().detach().numpy(), moran_idx=moran_idx, size=2, s=5, alpha=1, ax=None)
        #fig2.suptitle("Slideseq NSF", size=15)
        #fig2.set_figheight(2.25)
        #fig2.tight_layout()
        #fig2.savefig(f'{save_path}/{file_path}_plot.png')
        #plt.close()
        #model.to(device)



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
    save_path = path.join(SPATH, 'nsf')
    run_experiment(X_train, Y_train, save_path)
    print("DONE!!")


if __name__ == '__main__':
    main()
    



    
    
        

