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
from sklearn.metrics import mean_poisson_deviance
from torch import nn, distributions, optim
from torch.distributions import Normal
from tqdm.auto import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(256)
random_seed = 256

ROOT_PATH = '/engelhardt/home/pshrestha/vnngp/'
RPATH = path.join(ROOT_PATH, "results/")
SPATH = path.join(RPATH, "visium/")
FPATH = path.join(SPATH, "figs/")

def average_rmse(A, B):
    A = np.array(A)
    B = np.array(B)
    return (np.sqrt((A.flatten() - (B).flatten())**2).sum()) / (A.shape[0]*A.shape[1])

def run_experiment(X, Y, save_path, param_grid, batched=True):
    #regions = [ (-1.6327526351276007, 0.06724736487239924, -2.196548419047762, -0.49654841904776226)]
    regions = [(-0.6590241627691157, 0.04097583723088427, -1.952626897827451, -1.252626897827451), 
              #(-0.6590241627691157, 0.04097583723088427, -1.952626897827451, -1.252626897827451), 
              #(-0.6590241627691157, 0.04097583723088427, -1.952626897827451, -1.252626897827451)]

    experiment_indices = []
    
    for i, (x_min, x_max, y_min, y_max) in enumerate(regions):
        indices = np.where((X[:, 0] >= x_min) & (X[:, 0] <= x_max) & (X[:, 1] >= y_min) & (X[:, 1] <= y_max))[0]
        X_square = X[indices]
        Y_square = Y[:, indices]
        experiment_indices.append(indices)

    poi_eval = mean_poisson_deviance
    rmse_eval = root_mean_squared_error
    
    for l in param_grid['L']:
        mean_ks_poi = []
        mean_ks_rmse = []
        for k in param_grid['K']:
            dicts = {
                            'L': l, 
                            'M': None, # fill this in
                            'sigma': param_grid['sigma'], 
                            'lengthscale': param_grid['lengthscale'], 
                            'jtr': 1e-1, # no lower
                            'lr': param_grid['learning_rate'],
                            'rs': 256,
                            'lkzz_build': 1,
                            'model': 'VNNGP',
                            'L1_jitter': 1e-1, # no lower
                            'iterations': param_grid['iterations'],
                            'K': k,
                            'batch_size': None,
                    }
            all_index_rmse = []
            all_index_poi = []
            
            for index in experiment_indices:
                X_square = X[index]
                Y_square = Y.T[:, index]
                Y_square = Y_square.T
                dicts['M'] = len(X_square) # fill this in
                dicts['batch_size'] = len(X_square) # fill this in
                #print(Y_square.shape)
        
                nmf_model = NMF(n_components=dicts['L'], max_iter=100, init='random', random_state=256, alpha_H=2e-1, alpha_W=1e-7)
                nmf_model.fit(Y_square)
                exp_factors = nmf_model.transform(Y_square)
                factors = np.log(exp_factors + 1e-2)
                loadings = nmf_model.components_.T
                #print(factors.shape)
                #print(X_square.shape)
                moran_idx, moranI = putil.dims_autocorr(factors, np.array(X_square))
                factors = factors[:, moran_idx]
                loadings = loadings[:, moran_idx]
                Y_square = Y_square.T
                model = putil.build_model(np.array(X_square), np.array(Y_square), loadings=loadings, factors=factors, kwargs=dicts)
                losses, means, scales, final_time = [], [], [], 0
                
        
                putil.model_grads(model)
                model.prior.jitter = dicts['jtr']
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=dicts['lr'])
                model.to(device)
                X_torch = torch.tensor(X_square).type(torch.float).to(device)
                Y_torch = torch.tensor(Y_square).type(torch.float).to(device)
            
                start_time = time.time()
                more_losses, more_means, more_scales, idxs = putil.train_batched(model, optimizer, X_torch, Y_torch, device, 
                                                                                 steps=dicts['iterations'], E=3, kwargs=dicts)
                end_time = time.time()
                final_time += (end_time - start_time)
        
                # combine new and previous losses, means, scales
                losses.extend(more_losses)
                means.extend(more_means)
                scales.extend(more_scales)
        
                #torch.save(model.state_dict(), model_state_path)
                #torch.save({
                        #'losses': losses,
                        #'means': means,
                        #'scales': scales,
                        #'time': final_time
                #}, model_info_path)
            
                with torch.no_grad():
                    X_torch = torch.tensor(X_square).type(torch.float)
                    Y_torch = torch.tensor(Y_square).type(torch.float)
                    poi = putil.evaluate_model(model, X_torch.cpu(), Y_torch, device, evaluation_metric=poi_eval, kwargs=dicts)
                    rmse = putil.evaluate_model(model, X_torch.cpu(), Y_torch, device, evaluation_metric=rmse_eval, kwargs=dicts)
                all_index_poi.append(poi)
                all_index_rmse.append(rmse)
                print(f'Params: {dicts}, Poi: {poi}, RMSE: {rmse}')
            mean_ks_poi.append(np.array(all_index_poi).mean())
            mean_ks_rmse.append(np.array(all_index_rmse).mean())
        mean_ks_poi = np.array(mean_ks_poi)
        mean_ks_rmse = np.array(mean_ks_rmse)
        np.save(path.join(save_path, f'k_exp_L={l}_poi_fov=104_cortex.npy'), mean_ks_poi)
        np.save(path.join(save_path, f'k_exp_L={l}_rmse_fov=104_cortex.npy'), mean_ks_rmse)
        print(f"Done with L = {l}")
        #return mean_ks_poi, mean_ks_rmse

def main():
    X, Y = putil.load_visium()
    #X_train, X_val, y_train, y_val = train_test_split(X, Y.T, test_size=0.05, random_state=256) # split validation into 95%
                                                                                                 # training and 5% validation
    save_path = path.join(SPATH, 'nnnsf')
    param_grid = {
        'sigma': 1.0,
        'lengthscale': 1.0,
        'learning_rate': 1e-3,
        'iterations': 5000,
        'L': [5, 10, 15],
        'K': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 70, 100, 104]}
    run_experiment(X, Y, save_path, param_grid)
    print("Done with experiment.")



if __name__ == '__main__':
    main()
    



    
    
        

