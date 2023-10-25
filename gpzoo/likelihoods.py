from torch import distributions
import torch.nn as nn
import torch

from .gp import MGGP_SVGP


class GaussianLikelihood(nn.Module):
  def __init__(self, gp):
    super().__init__()
    self.gp = gp

    self.noise = nn.Parameter(torch.tensor(0.1))

  def forward(self, X, E=1, verbose=False):
    qF, qU, pU = self.gp(X, verbose=verbose)
    F = qF.rsample((E, ))
    pY = distributions.Normal(F, torch.abs(self.noise))

    return pY, qF, qU, pU
  

class NSF(nn.Module):
    def __init__(self, gp, y, M=50, L=10, jitter=1e-4):
      super().__init__()
      D, N = y.shape
      self.gp = gp
      self.gp.Lu = nn.Parameter(5e-2*torch.rand((L, M, M)))
      self.gp.mu = nn.Parameter(torch.zeros((L, M)))

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
    

class MGGP_NSF(nn.Module):
    def __init__(self, X, y, kernel, M=50, L=10, jitter=1e-4, n_groups=2, non_spatial_factors=0):
      super().__init__()
      D, N = y.shape
      self.non_spatial_factors = non_spatial_factors
      self.svgp = MGGP_SVGP(kernel, dim=2, M=M, jitter=jitter, n_groups=n_groups)
      self.svgp.Lu = nn.Parameter(5e-1*torch.rand((L, n_groups*M, n_groups*M)))
      self.svgp.mu = nn.Parameter(torch.randn((L, n_groups*M)))

      
      self.non_spatial_mean = nn.Parameter(torch.randn((L, N)))
      self.non_spatial_scales = nn.Parameter(torch.rand((L, N)))

      

      self.W = nn.Parameter(torch.rand((D, L+self.non_spatial_factors)))

      self.V = nn.Parameter(torch.ones((N,)))

    
    def forward_batched(self, X, groupsX, idx, E=10, verbose=False):
      qF, qU, pU = self.svgp(X[idx], groupsX[idx], verbose)
        
      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      Z = torch.matmul(torch.abs(self.W), F) #shape ExDxN
      pY = distributions.Poisson(torch.abs(self.V[idx])*Z)
      return pY, qF, qU, pU

    def forward(self, X, groupsX, E=10, verbose=False):
      qF, qU, pU = self.svgp(X, groupsX, verbose)
        
      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      Z = torch.matmul(torch.abs(self.W), F) #shape ExDxN
      pY = distributions.Poisson(torch.abs(self.V)*Z)
      return pY, qF, qU, pU