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
  
class PNMF(nn.Module):
  def __init__(self, prior, y, L=10):
    super().__init__()
    D, N = y.shape
    self.prior = prior
    self.W = nn.Parameter(torch.rand((D, L)))
    self.V = nn.Parameter(torch.ones((N,)))

  def forward(self, E=10, **kwargs):
    qF, pF = self.prior()
    F = qF.rsample((E,))

    F = torch.exp(F)
    W = torch.nn.functional.softplus(self.W)
    V = torch.nn.functional.softplus(self.V)

    Z = torch.matmul(W, F) #shape ExDxN
    pY = distributions.Poisson(V*Z)

    return pY, qF, pF

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
      self.scale_qF = nn.Parameter(1e-1*torch.rand((non_spatial_factors, N)))

      # self.scale_pF = nn.Parameter(torch.ones(non_spatial_factors)[:, None])
      # self.mean_pF =  nn.Parameter(torch.zeros(non_spatial_factors))

    def forward(self, X, E=10, verbose=False, **kwargs):
      qF, qU, pU = self.gp(X=X, verbose=verbose,  **kwargs)
      scale_qF = torch.nn.functional.softplus(self.scale_qF) #keep it positive
      # scale_pF = torch.nn.functional.softplus(self.scale_pF)

      qF2 = distributions.Normal(self.mF, scale_qF)


      
      F = qF.rsample((E,)) #shape ExLxN
      F2 = qF2.rsample((E,))

      F = torch.cat((F, F2), dim=1)

      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      # W = torch.nn.functional.softplus(self.W)
      # W2 = torch.nn.functional.softplus(self.W2)
      # W = torch.clamp(W, min=0.0)
      # W2 = torch.clamp(W2, min=0.0)
      

      W = torch.cat((self.W, self.W2), dim=1)

      V = torch.nn.functional.softplus(self.V)

      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson(V*Z)

      pF2 = distributions.Normal(torch.zeros_like(self.mF), torch.ones_like(scale_qF))

      return pY, qF, qU, pU, qF2, pF2
    
    def forward_batched(self, X, idx, E=10, verbose=False, **kwargs):
      qF, qU, pU = self.gp(X=X[idx], verbose=verbose,  **kwargs)
      scale_qF = torch.nn.functional.softplus(self.scale_qF) #keep it positive
      # scale_pF = torch.nn.functional.softplus(self.scale_pF)

      qF2 = distributions.Normal(self.mF[:, idx], scale_qF[:, idx])


      
      F = qF.rsample((E,)) #shape ExLxN
      F2 = qF2.rsample((E,))

      F = torch.cat((F, F2), dim=1)


      F = torch.exp(F)


      W = torch.cat((self.W, self.W2), dim=1)

      V = torch.nn.functional.softplus(self.V)

      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson((V[idx])*Z)
      pF2 = distributions.Normal(torch.zeros_like(self.mF[:, idx]), torch.ones_like(scale_qF[:, idx]))

      return pY, qF, qU, pU, qF2, pF2



class MGGP_NSF(nn.Module):
    def __init__(self, gp, y, L=10):
      super().__init__()
      D, N = y.shape
      self.gp = gp #needs to be a MGGP
    
      self.W = nn.Parameter(torch.rand((D, L)))
      self.V = nn.Parameter(torch.ones((N,)))

    
    def forward_batched(self, X, groupsX, idx, E=10, verbose=False):

      if verbose:
        print('X:shape:',X[idx].shape)
        print('groupsX.shape:',groupsX[idx].shape)

      qF, qU, pU = self.gp(X[idx], groupsX[idx], verbose)
      
      W = torch.nn.functional.softplus(self.W)
      V = torch.nn.functional.softplus(self.V)

      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson((V[idx])*Z)
      return pY, qF, qU, pU

    def forward(self, X, groupsX, E=10, verbose=False):
      qF, qU, pU = self.gp(X, groupsX, verbose)
      
      W = torch.nn.functional.softplus(self.W)
      V = torch.nn.functional.softplus(self.V)
      F = qF.rsample((E,)) #shape ExLxN
      # F = 255*torch.softmax(F, dim=2)
      F = torch.exp(F)
      #F = torch.transpose(F, -2, -1)
      Z = torch.matmul(W, F) #shape ExDxN
      pY = distributions.Poisson(V*Z)
      return pY, qF, qU, pU