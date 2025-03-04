from torch import distributions
import torch.nn as nn
import torch

from torch.distributions import constraints, transform_to

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


class ExactLikelihood(nn.Module):
  def __init__(self, gp, noise=0.1):
    super().__init__()
    self.gp = gp

    self.noise = nn.Parameter(torch.tensor(noise))

  def forward(self, X, E=1, verbose=False, **kwargs):
    qF, qU, pU = self.gp(X, verbose=verbose, **kwargs)

    noise = torch.nn.functional.softplus(self.noise) #ensure positive
    pY = distributions.Normal(qF.mean, noise)

    return pY, qF, qU, pU
  

class PoissonFactorization(nn.Module):
  '''
  Poisson Factorization base model for both PNMF, and NSF
  '''
  def __init__(self, prior, y, L=10):
    super().__init__()
    D, N = y.shape
    self.prior = prior
    self.W = nn.Parameter(torch.rand((D, L)))

  def get_rate(self, prior_samples):
    F = torch.exp(prior_samples)
    W = torch.nn.functional.softplus(self.W)
    Z = torch.matmul(W, F) #shape ExDxN
    return Z


class PNMF(PoissonFactorization):
  def __init__(self, prior, y, L=10):
    super().__init__(prior=prior, y=y, L=L)
    D, N = y.shape
    self.V = nn.Parameter(torch.ones((N,)))
    self.X = nn.Parameter(torch.zeros((N, 2)), requires_grad=False) #to keep track of inputs, for plotting purposes


  def forward(self, E=10, **kwargs):
    qF, pF = self.prior()
    F = qF.rsample((E,))

    Z = self.get_rate(F)
    V = torch.nn.functional.softplus(self.V)
    pY = distributions.Poisson(V*Z)

    return pY, qF, pF
  
class NSF2(PoissonFactorization):
  def __init__(self, gp, y, L=10):
    super().__init__(prior=gp, y=y, L=L)
    D, N = y.shape
    self.V = nn.Parameter(torch.ones((N,)))

  def forward(self, X, E=10, verbose=False, **kwargs):
    qF, qU, pU = self.prior(X=X, verbose=verbose, **kwargs)
    F = qF.rsample((E,))
    Z = self.get_rate(F)
    V = torch.nn.functional.softplus(self.V)
    pY = distributions.Poisson(V*Z)

    return pY, qF, qU, pU
  

  def forward_batched(self, X, idx, E=10, verbose=False, **kwargs):
    qF, qU, pU = self.prior(X=X[idx], verbose=verbose, **kwargs)
    F = qF.rsample((E,))
    Z = self.get_rate(F)
    V = torch.nn.functional.softplus(self.V[idx])
    pY = distributions.Poisson(V*Z)

    return pY, qF, qU, pU


class Hybrid_NSF2(nn.Module):
  def __init__(self, gp, prior, y, L=10, T=10):
    super().__init__()

    D, N = y.shape
    self.sf = PoissonFactorization(prior = gp, y=y, L=L)
    self.cf = PoissonFactorization(prior = prior, y=y, L=T)
    self.V = nn.Parameter(torch.ones((N,)))


  def forward(self, X, E=10, verbose=False, **kwargs):
    qF1, qU, pU = self.sf.prior(X=X, verbose=verbose, **kwargs)
    qF2, pF2 = self.cf.prior()

    F1 = qF1.rsample((E,))
    F2 = qF2.rsample((E,))


    Z1 = self.sf.get_rate(F1)
    Z2 = self.cf.get_rate(F2)
    Z = Z1+Z2

    V = torch.nn.functional.softplus(self.V)
    pY = distributions.Poisson(V*Z)

    return pY, qF1, qU, pU, qF2, pF2

  
  def forward_batched(self, X, idx, E=10, verbose=False, **kwargs):

    qF1, qU, pU = self.sf.prior(X=X[idx], verbose=verbose, **kwargs)
    qF2, pF2 = self.cf.prior.forward_batched(idx)

    F1 = qF1.rsample((E,))
    F2 = qF2.rsample((E,))

    Z1 = self.sf.get_rate(F1)
    Z2 = self.cf.get_rate(F2)

    Z = Z1+Z2

    V = torch.nn.functional.softplus(self.V[idx])
    
    pY = distributions.Poisson(V*Z)

    return pY, qF1, qU, pU, qF2, pF2

  def forward_precomputed(self, W, idx, E=10, verbose=False, **kwargs):

    qF1, qU, pU = self.sf.prior.forward_precomputed(W, verbose=verbose, **kwargs)
    qF2, pF2 = self.cf.prior.forward_batched(idx)

    F1 = qF1.rsample((E,))
    F2 = qF2.rsample((E,))

    Z1 = self.sf.get_rate(F1)
    Z2 = self.cf.get_rate(F2)

    Z = Z1+Z2

    V = torch.nn.functional.softplus(self.V[idx])
    
    pY = distributions.Poisson(V*Z)

    return pY, qF1, qU, pU, qF2, pF2



class Hybrid_NSF_Exact(nn.Module):
  def __init__(self, gp, prior, y, L=10, T=10):
    super().__init__()

    D, N = y.shape
    self.sf = PoissonFactorization(prior = gp, y=y, L=L)
    self.cf = PoissonFactorization(prior = prior, y=y, L=T)
    self.V = nn.Parameter(torch.ones((N,)))


  def forward(self, X, E=10, verbose=False, **kwargs):
    qF1, qU, pU = self.sf.prior(X=X, verbose=verbose, **kwargs)
    qF2, pF2 = self.cf.prior()

    F1 = qF1.mean+ 0.5*(qF1.scale**2)
    F2 = qF2.mean+ 0.5*(qF2.scale**2)


    Z1 = self.sf.get_rate(F1)
    Z2 = self.cf.get_rate(F2)
    Z = Z1+Z2

    V = torch.nn.functional.softplus(self.V)
    pY = distributions.Poisson(V*Z)

    return pY, qF1, qU, pU, qF2, pF2

  
  def forward_batched(self, X, idx, E=10, verbose=False, **kwargs):

    qF1, qU, pU = self.sf.prior(X=X[idx], verbose=verbose, **kwargs)
    qF2, pF2 = self.cf.prior.forward_batched(idx)

    F1 = qF1.mean + 0.5*(qF1.scale**2)
    F2 = qF2.mean + 0.5*(qF2.scale**2)

    Z1 = self.sf.get_rate(F1) 
    Z2 = self.cf.get_rate(F2)

    Z = Z1+Z2

    V = torch.nn.functional.softplus(self.V[idx])
    
    pY = distributions.Poisson(V*Z)

    return pY, qF1, qU, pU, qF2, pF2


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