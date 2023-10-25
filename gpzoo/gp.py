import torch
from torch import distributions
from torch.distributions import constraints, transform_to
import torch.nn as nn
import tqdm
from .utilities import add_jitter, svgp_forward

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

class VNNGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, K=3, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
    
    self.K = K
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
    self.Lu = nn.Parameter(torch.randn((M, M)))
    self.mu = nn.Parameter(torch.zeros((M,)))
    self.constraint = constraints.lower_cholesky

  def forward(self, X, verbose=False):


    Kxx = self.kernel(X, X, diag=True)
    Kxx_shape = Kxx.shape
    Kxx = Kxx.view(-1, 1) # (... x N) x 1

    if verbose:
      print('calculating Kxx')
      print('Kxx.shape', Kxx.shape)
    

    Kxz, distances = self.kernel(X, self.Z, return_distance=True)

    Kxz_shape = Kxz.shape
    Kxz = Kxz.view(-1, Kxz_shape[-1]) # (... x N) x M

    if verbose:
      print('calculating Kxz')
      print('Kxz.shape', Kxz.shape)

    Kzz = self.kernel(self.Z, self.Z)

    Kzz_shape = Kzz.shape
    Kzz = Kzz.view(-1, Kzz_shape[-2], Kzz_shape[-1]) # ... x M x M

    if verbose:
      print('calculating kzz')
      print('Kzz.shape', Kzz.shape)

    Lu = transform_to(self.constraint)(self.Lu)
    Lu_shape = Lu.shape
    Lu = Lu.view(-1, Lu_shape[-2], Lu_shape[-1]) # ... x M x M


    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter)) # ... x M x M
    L_shape = L.shape
    L = L.view(-1, L_shape[-2], L_shape[-1]) # ... x M x M

    if verbose:
      print('calculating L')
      print('L.shape', L.shape)


    indexes = torch.argsort(distances, dim=1)[:, :self.K]

    little_L = L[:, indexes] # ... x N x K x M

    if verbose:
      print('Little_L.shape:', little_L.shape)



    little_Kzz = little_L @ torch.transpose(little_L, -2, -1) # ... x N x K x K
    little_Kzz_shape = little_Kzz.shape
    little_Kzz = little_Kzz.view(-1, little_Kzz_shape[-2], little_Kzz_shape[-1]) # ( ... x N) x K x K


    kzz_inv = torch.inverse(add_jitter(little_Kzz, self.jitter)) # (... x N) x KxK

    little_Kxz = torch.gather(Kxz, 1, indexes.expand(Kxz.shape[0], -1))[:, None, :] #(... x N)x1xK

    W = little_Kxz  @ kzz_inv # (... x N) x 1 x K

    if verbose:
      print('W_shape:', W.shape)

    mu_shape = self.mu.shape

    mu = self.mu.view(-1, mu_shape[-1]) # ... x M

    little_mu = mu[:, indexes]# ... x  N x K
    little_mu = little_mu.view(-1, little_mu.shape[-1]) # (... x  N) x K



    little_Lu = Lu[:, indexes] # ... x N x K x M
    little_S = little_Lu @ torch.transpose(little_Lu, -2, -1) # ... x N x K x K
    little_S = little_S.view(-1, little_S.shape[-2], little_S.shape[-1]) # (... x N) x K x K


    if verbose:
      print(Kxx.shape, little_Kzz.shape, W.shape, little_mu.shape, little_S.shape)
    mean, cov = svgp_forward(Kxx, little_Kzz, W, little_mu, little_S)

    if verbose:
      print('mean.shape:', mean.shape)
      print('cov.shape:', cov.shape)

    mean = torch.squeeze(mean)
    cov = torch.squeeze(cov)

    mean = mean.view(*Kxx_shape)
    cov = cov.view(*Kxx_shape)


    qF = distributions.Normal(mean, torch.clamp(cov, min=5e-2) ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU


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
    Kxx = self.kernel(X, X, diag=True) #shape L x N

    if verbose:
      print('calculating Kzx')
    Kzx = self.kernel(self.Z, X) #shape L x M x N

    if verbose:
      print('calculating kzz')
    Kzz = self.kernel(self.Z, self.Z) #shape L x M x M

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter)) #shape L x M x M
   
    if verbose:
        print('calculating W')
   
    W = torch.cholesky_solve(Kzx, L) #(Kzz)-1 @ Kzx
    W = torch.transpose(W, -2, -1) # Kxz@(Kzz)-1, shape # L x N x M
    Lu = transform_to(self.constraint)(self.Lu) #shape L x M x M
    S = Lu @ torch.transpose(Lu, -2, -1) # shape L x M x M

    mean, cov_diag = svgp_forward(Kxx, Kzz, W, self.mu, S)
    mean = torch.squeeze(mean)
    
    qF = distributions.Normal(mean, torch.clamp(cov_diag, min=5e-2) ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU


class MGGP_SVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4, n_groups=2):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
        
    self.Z = nn.Parameter(torch.randn((n_groups*M, dim))) #choose inducing points
    self.groupsZ = nn.Parameter(torch.concatenate([i*torch.ones(M) for i in range(n_groups)]).type(torch.LongTensor), requires_grad=False)
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
    
    
    mean, cov_diag = svgp_forward(Kxx, Kzz, W, self.mu, S)
    mean = torch.squeeze(mean)


    qF = distributions.Normal(mean, torch.clamp(cov_diag, min=5e-2) ** 0.5) #setting max cov_diag to 100, need to find a way to clip values manually later
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU

class NSF(nn.Module):
    def __init__(self, X, y, kernel, M=50, L=10, jitter=1e-4):
      super().__init__()
      D, N = y.shape
      self.svgp = SVGP(kernel, dim=2, M=M, jitter=jitter)
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