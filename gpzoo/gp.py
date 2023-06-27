import torch
import numpy as np
from torch import distributions
import torch.nn as nn
import tqdm
import kernels
from utilities import Utilities

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
        
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
    self.Lu = nn.Parameter(torch.randn((M, M)))
    self.mu = nn.Parameter(torch.zeros((M,)))

  def forward(self, X, verbose=False):
    if verbose:
      print('calculating Kxx')
    Kxx = self.kernel(X, X, diag=True)

    if verbose:
      print('calculating Kzx')
    Kzx = self.kernel(self.Z, X)

    if verbose:
      print('calculating kzz')
    Kzz = self.kernel(self.Z, self.Z)

    if verbose:
      print('calculating cholesky')
    L = torch.cholesky(Utilities.add_jitter(Kzz, self.jitter))
   
    if verbose:
        print('calculating W')
   
    W = torch.cholesky_solve(Kzx, L) #(Kzz)-1 @ Kzx
    # Kzz_inv = torch.cholesky_inverse(L)
    # W = Kzz_inv @ Kzx
    W = torch.transpose(W, -2, -1)# Kxz@(Kzz)-1
    S = torch.transpose(self.Lu, -2, -1) @ self.Lu
    
    if verbose:
        print('calculating predictive mean')

    mean = W@ (self.mu.unsqueeze(-1))
    mean = torch.squeeze(mean)

    if verbose:
        print('calculating predictive covariance')

    cov_diag = Kxx + torch.diagonal( W@ (S-Kzz)@ torch.transpose(W, -2, -1), dim1=-2, dim2=-1)
    qF = distributions.Normal(mean, cov_diag ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=torch.cholesky(Utilities.add_jitter(S, self.jitter)))
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU
  
  def fit(self, X, y, optimizer, lr=0.005, epochs=1000, E=20):
    losses = []
    for it in tqdm(range(epochs)):
        optimizer.zero_grad()
        pY, qF, qU, pU = self.forward(X, E=E)
        ELBO = (pY.log_prob(y)).mean(axis=0).sum()
        ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
        loss = -ELBO
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print("finished Training")

    return losses

class NSF(nn.Module):
    def __init__(self, X, y, kernel, M=50, L=10, jitter=1e-4):
      super().__init__()
      D, N = y.shape
      self.kernel = kernel
      self.svgp = SVGP(self.kernel, dim=2, M=M, jitter=jitter)
      self.svgp.Lu = nn.Parameter(5e-2*torch.rand((L, M, M)))
      self.svgp.mu = nn.Parameter(torch.zeros((L, M)))

      self.W = nn.Parameter(torch.rand((D, L)))

      self.V = nn.Parameter(torch.ones((N,)))


    def forward(self, X, E=10, verbose=False):
      qF, qU, pU = self.svgp(X, verbose)
        
      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      Z = torch.matmul(torch.abs(self.W), F) #shape ExDxN
      pY = distributions.Poisson(torch.abs(self.V)*Z)
      return pY, qF, qU, pU
  
    def fit(self, X, y, optimizer, lr=0.005, epochs=1000, E=20):
      losses = []
      for it in tqdm.tqdm(range(epochs)):
          optimizer.zero_grad()
          pY, qF, qU, pU = self.forward(X, E=E)
          ELBO = (pY.log_prob(y)).mean(axis=0).sum()
          ELBO -= torch.sum(distributions.kl_divergence(qU, pU))
          loss = -ELBO
          loss.backward()
          optimizer.step()
          losses.append(loss.item())
      
      print("finished Training")

      return losses