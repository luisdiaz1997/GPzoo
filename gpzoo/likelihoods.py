from torch import distributions
import torch.nn as nn
import torch

from torch.distributions import constraints, transform_to
from .modules import PositiveParameter

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
  

class MEFISTO(nn.Module):
  def __init__(self, gp, y, L=10, noise=0.1):
    super().__init__()
    D, _ = y.shape
    self.prior = gp
    self.W = nn.Parameter(torch.rand((D, L)))
    noise_init = torch.full((D,), float(noise))
    self.noise = nn.Parameter(noise_init)

  def _project_latents(self, F, idy=None):
    if idy is not None:
      W = self.W[idy]
    else:
      W = self.W
    return torch.matmul(W, F)

  def _noise_scale(self, idy=None):
    noise = torch.nn.functional.softplus(self.noise)
    if idy is not None:
      noise = noise[idy]
    return noise.unsqueeze(0).unsqueeze(-1)

  def _likelihood(self, qF, E, idy=None):
    F = qF.rsample((E,))
    mean = self._project_latents(F, idy=idy)
    scale = self._noise_scale(idy=idy)
    return distributions.Normal(mean, scale)

  def forward(self, X, E=10, verbose=False, **kwargs):
    qF, qU, pU = self.prior(X=X, verbose=verbose, **kwargs)
    pY = self._likelihood(qF, E, idy=None)
    return pY, qF, qU, pU

  def forward_batched(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
    qF, qU, pU = self.prior(X=X, verbose=verbose, **kwargs)
    pY = self._likelihood(qF, E, idy=idy)
    return pY, qF, qU, pU

  def forward_batched_train(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
    if hasattr(self.prior, 'forward_train'):
      qF, qU, pU = self.prior.forward_train(X=X, idx=idx, verbose=verbose, **kwargs)
    else:
      qF, qU, pU = self.prior(X=X, verbose=verbose, **kwargs)
    pY = self._likelihood(qF, E, idy=idy)
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
  def __init__(self, prior, y, L=10, loadings_mode='softplus'):
    super().__init__()
    D, N = y.shape
    self.prior = prior
    self.loadings_mode = loadings_mode
    # Use PositiveParameter for loadings W
    self.W = PositiveParameter((D, L), mode=loadings_mode)

  def get_rate(self, prior_samples, idy=None):
    F = torch.exp(prior_samples)  # shape ExLxN

    if idy is not None:
      W = self.W[idy]
    else:
      W = self.W

    # W is already positive from PositiveParameter
    Z = torch.matmul(W, F)  # shape ExDxN

    return Z

  def project_parameters(self):
    """Apply projection to parameters (clamp negative values to zero)."""
    # PositiveParameter handles projection internally
    self.W.project()


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
  def __init__(self, gp, y, L=10, loadings_mode='softplus'):
    super().__init__(prior=gp, y=y, L=L, loadings_mode=loadings_mode)
    D, N = y.shape
    self.V = nn.Parameter(torch.ones((N,)))

  def forward(self, X, E=10, verbose=False, **kwargs):
    qF, qU, pU = self.prior(X=X, verbose=verbose, **kwargs)
    F = qF.rsample((E,))
    Z = self.get_rate(F)
    V = torch.nn.functional.softplus(self.V)
    pY = distributions.Poisson(V*Z)

    return pY, qF, qU, pU
  

  def forward_batched(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
    '''X should be batched already, this is just for V'''
    qF, qU, pU = self.prior(X=X, verbose=verbose, **kwargs)
    F = qF.rsample((E,))
    Z = self.get_rate(F, idy=idy)
    V = torch.nn.functional.softplus(self.V[idx])
    pY = distributions.Poisson(V*Z)

    return pY, qF, qU, pU
  
  def forward_batched_train(self, X, idx, idy=None, E=10, verbose=False, **kwargs):
    '''X should be batched already, this is just for V'''
    qF, qU, pU = self.prior.forward_train(X=X, idx=idx, verbose=verbose, **kwargs)
    F = qF.rsample((E,))
    Z = self.get_rate(F, idy=idy)
    V = torch.nn.functional.softplus(self.V[idx])
    pY = distributions.Poisson(V*Z)

    return pY, qF, qU, pU
  
  def forward_batched_train_separate(self, X, y, idx, idy=None, E=10, verbose=False, mode='exp_sum', **kwargs):
    '''X should be batched already, this is just for V'''
    qF, qU, pU = self.prior.forward_train(X=X, idx=idx, verbose=verbose, **kwargs)
    F_samples = qF.rsample((E,)) 
    F_exact= qF.mean + 0.5*(qF.scale**2)

    rate_exact = self.get_rate(F_exact, idy=idy, mode=mode)
    log_rate_samples = self.get_rate(F_samples, idy=idy, mode=mode, return_log=True)

    
    V = torch.nn.functional.softplus(self.V[idx])

    rate_exact = V*rate_exact


    expected_log = y[idy]*log_rate_samples
    average_expected_log = torch.mean(expected_log, dim=0)
  

    return average_expected_log-rate_exact, qF, qU, pU


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
