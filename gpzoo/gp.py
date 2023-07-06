import torch
import numpy as np
from torch import distributions
from torch.distributions import constraints, transform_to
import torch.nn as nn
import tqdm
from .utilities import add_jitter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
        
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
    self.Lu = nn.Parameter(torch.randn((M, M)))
    self.mu = nn.Parameter(torch.zeros((M,)))
    self.constraint = constraints.lower_cholesky

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
    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter))
   
    if verbose:
        print('calculating W')
   
    W = torch.cholesky_solve(Kzx, L) #(Kzz)-1 @ Kzx
    # Kzz_inv = torch.cholesky_inverse(L)
    # W = Kzz_inv @ Kzx
    W = torch.transpose(W, -2, -1)# Kxz@(Kzz)-1
    Lu = transform_to(self.constraint)(self.Lu)
    S = Lu @ torch.transpose(Lu, -2, -1)
    
    if verbose:
        print('calculating predictive mean')

    mean = W@ (self.mu.unsqueeze(-1))
    mean = torch.squeeze(mean)

    if verbose:
        print('calculating predictive covariance')


    diff = S-Kzz
    cov_diag = Kxx + torch.sum((W @ diff)* W, dim=-1)
    qF = distributions.Normal(mean, cov_diag ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
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


class MGGP_SVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4, n_groups=2):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
        
    self.Z = nn.Parameter(torch.randn((n_groups*M, dim))) #choose inducing points
    self.groupsZ = torch.concatenate([i*torch.ones(M) for i in range(n_groups)]).type(torch.LongTensor)
    self.Lu = nn.Parameter(torch.randn((n_groups*M, n_groups*M)))
    self.mu = nn.Parameter(torch.zeros((n_groups*M,)))
    self.constraint = constraints.lower_cholesky

  def forward(self, X, groupsX, verbose=False):
    if verbose:
      print('calculating Kxx')
    Kxx = self.kernel(X, X, groupsX, groupsX, diag=True)

    if verbose:
      print('calculating Kzx')
    Kzx = self.kernel(self.Z, X, self.groupsZ, groupsX)

    if verbose:
      print('calculating kzz')
    Kzz = self.kernel(self.Z, self.Z, self.groupsZ, self.groupsZ)

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter))
   
    if verbose:
        print('calculating W')
   
    W = torch.cholesky_solve(Kzx, L) #(Kzz)-1 @ Kzx
    # Kzz_inv = torch.cholesky_inverse(L)
    # W = Kzz_inv @ Kzx
    W = torch.transpose(W, -2, -1)# Kxz@(Kzz)-1
    Lu = transform_to(self.constraint)(self.Lu)
    S = Lu @ torch.transpose(Lu, -2, -1)
    
    if verbose:
        print('calculating predictive mean')

    mean = W@ (self.mu.unsqueeze(-1))
    mean = torch.squeeze(mean)

    if verbose:
        print('calculating predictive covariance')


    diff = S-Kzz
    cov_diag = Kxx + torch.sum((W @ diff)* W, dim=-1)
    qF = distributions.Normal(mean, cov_diag ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
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

    
    #experimental batched forward
    def batched_forward(self, X, idx, E=10, verbose=False):
      qF, qU, pU = self.svgp(X, verbose)
      F = qF.rsample((E,)) #shape ExLxN
      F = torch.exp(F)

      W = self.W[idx]

      Z = torch.matmul(torch.abs(W), F) #shape ExDxN
      pY = distributions.Poisson(torch.abs(self.V)*Z)
      return pY, qF, qU, pU

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