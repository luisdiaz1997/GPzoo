from torch import distributions
import torch.nn as nn
import torch

from torch.distributions import constraints, transform_to

from .gp import MGGP_SVGP


class GaussianLikelihood(nn.Module):
  def __init__(self, gp, noise=0.1):
    super().__init__()
    self.gp = gp

    self.noise = nn.Parameter(torch.tensor(noise))

  def forward(self, X, E=1, verbose=False, **kwargs):
    qF, qU, pU = self.gp(X, verbose=verbose, **kwargs)
    F = qF.rsample((E, ))
    noise = torch.nn.functional.softplus(self.noise) #ensure positive
    pY = distributions.Normal(F, noise)

    return pY, qF, qU, pU
  

class NSF(nn.Module):
    def __init__(self, gp, y, L=10):
      super().__init__()
      D, N = y.shape
      self.gp = gp

      self.W = nn.Parameter(torch.rand((D, L)))

      self.V = nn.Parameter(torch.ones((N,)))

    def forward(self, X, E=10, verbose=False, **kwargs):
      qF, qU, pU = self.gp(X=X, verbose=verbose,  **kwargs)
        
      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      W = torch.nn.functional.softplus(self.W)
      V = torch.nn.functional.softplus(self.V)

      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson(V*Z)
      return pY, qF, qU, pU
    
    def forward_batched(self, X, idx, E=10, verbose=False, **kwargs):
      qF, qU, pU = self.gp(X=X[idx], verbose=verbose, **kwargs)
        
      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)

      W = torch.nn.functional.softplus(self.W)
      V = torch.nn.functional.softplus(self.V)

      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson((V[idx])*Z)
      return pY, qF, qU, pU
    

class Hybrid_NSF(NSF):
    def __init__(self, gp, y, L=10, non_spatial_factors=10):
      super().__init__(gp=gp, y=y, L=L)

      D, N = y.shape

      self.W2 = nn.Parameter(torch.rand((D, non_spatial_factors)))
      self.mF = nn.Parameter(torch.zeros((non_spatial_factors, N)))
      self.scaleF = nn.Parameter(1e-1*torch.rand((non_spatial_factors, N)))

    def forward(self, X, E=10, verbose=False, **kwargs):
      qF, qU, pU = self.gp(X=X, verbose=verbose,  **kwargs)
      scaleF = torch.nn.functional.softplus(self.scaleF) #keep it positive

      qF2 = distributions.Normal(self.mF, scaleF)


      
      F = qF.rsample((E,)) #shape ExLxN
      F2 = qF2.rsample((E,))

      F = torch.cat((F, F2), dim=1)

      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      W = torch.nn.functional.softplus(self.W)
      W2 = torch.nn.functional.softplus(self.W2) 

      W = torch.cat((W, W2), dim=1)

      V = torch.nn.functional.softplus(self.V)

      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson(V*Z)

      pF2 = distributions.Normal(torch.zeros_like(self.mF), torch.ones_like(scaleF))

      return pY, qF, qU, pU, qF2, pF2
    
    def forward_batched(self, X, idx, E=10, verbose=False, **kwargs):
      qF, qU, pU = self.gp(X=X[idx], verbose=verbose,  **kwargs)
      scaleF = torch.nn.functional.softplus(self.scaleF) #keep it positive

      qF2 = distributions.Normal(self.mF[:, idx], scaleF[:, idx])


      
      F = qF.rsample((E,)) #shape ExLxN
      F2 = qF2.rsample((E,))

      F = torch.cat((F, F2), dim=1)

      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      W = torch.nn.functional.softplus(self.W)
      W2 = torch.nn.functional.softplus(self.W2) 

      W = torch.cat((W, W2), dim=1)

      V = torch.nn.functional.softplus(self.V)

      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson((V[idx])*Z)

      pF2 = distributions.Normal(torch.zeros_like(self.mF[:, idx]), torch.ones_like(scaleF[:, idx]))

      return pY, qF, qU, pU, qF2, pF2



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