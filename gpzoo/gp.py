import torch
from torch import distributions
from torch.distributions import constraints, transform_to
import torch.nn as nn
from .utilities import add_jitter, svgp_forward, reshape_param

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
    Kxx = Kxx.contiguous().view(-1, 1) # (... x N) x 1

    if verbose:
      print('calculating Kxx')
      print('Kxx.shape', Kxx.shape)
    

    Kxz, distances = self.kernel(X, self.Z, return_distance=True)
    print(f'distances: {distances.shape}')

    Kxz_shape = Kxz.shape
    Kxz = Kxz.contiguous().view(-1, Kxz_shape[-1]) # (... x N) x M

    if verbose:
      print('calculating Kxz')
      print('Kxz.shape', Kxz.shape)

    Kzz = self.kernel(self.Z, self.Z)

    Kzz_shape = Kzz.shape
    Kzz = Kzz.contiguous().view(-1, Kzz_shape[-2], Kzz_shape[-1]) # ... x M x M

    if verbose:
      print('calculating kzz')
      print('Kzz.shape', Kzz.shape)

    Lu = transform_to(self.constraint)(self.Lu)
    Lu_shape = Lu.shape
    Lu = Lu.contiguous().view(-1, Lu_shape[-2], Lu_shape[-1]) # ... x M x M


    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter)) # ... x M x M
    L_shape = L.shape
    L = L.contiguous().view(-1, L_shape[-2], L_shape[-1]) # ... x M x M

    if verbose:
      print('calculating L')
      print('L.shape', L.shape)


    indexes = torch.argsort(distances, dim=1)[:, :self.K]
    print('Indexes shape: ', indexes.shape)

    little_L = L[:, indexes] # ... x N x K x M

    if verbose:
      print('Little_L.shape:', little_L.shape)

    little_Kzz = little_L @ torch.transpose(little_L, -2, -1) # ... x N x K x K
    little_Kzz_shape = little_Kzz.shape
    little_Kzz = little_Kzz.contiguous().view(-1, little_Kzz_shape[-2], little_Kzz_shape[-1]) # ( ... x N) x K x K


    kzz_inv = torch.inverse(add_jitter(little_Kzz, self.jitter)) # (... x N) x KxK
      
    print(f'Kxz: {Kxz.shape}')
    print(f'kzz_inv: {kzz_inv.shape}')
    print(f'indexes: {indexes.shape}')

    expanded = indexes.repeat(Kxx_shape[0], 1)
    print('Expanded shape', expanded.shape)

    little_Kxz = torch.gather(Kxz, 1, expanded)[:, None, :] #(... x N)x1xK
    
    W = little_Kxz  @ kzz_inv # (... x N) x 1 x K # issue is here

    if verbose:
      print('W_shape:', W.shape)

    mu_shape = self.mu.shape

    mu = self.mu.contiguous().view(-1, mu_shape[-1]) # ... x M

    little_mu = mu[:, indexes]# ... x  N x K
    little_mu = little_mu.view(-1, little_mu.shape[-1]) # (... x  N) x K

    little_Lu = Lu[:, indexes] # ... x N x K x M
    little_S = little_Lu @ torch.transpose(little_Lu, -2, -1) # ... x N x K x K
    little_S = little_S.contiguous().view(-1, little_S.shape[-2], little_S.shape[-1]) # (... x N) x K x K

    if verbose:
      print(Kxx.shape, little_Kzz.shape, W.shape, little_mu.shape, little_S.shape)
    mean, cov = svgp_forward(Kxx, little_Kzz, W, little_mu, little_S)

    if verbose:
      print('mean.shape:', mean.shape)
      print('cov.shape:', cov.shape)

    mean = torch.squeeze(mean)
    cov = torch.squeeze(cov)

    mean = mean.contiguous().view(*Kxx_shape)
    cov = cov.contiguous().view(*Kxx_shape)

    qF = distributions.Normal(mean, torch.clamp(cov, min=5e-2) ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU


class GaussianPrior(nn.Module):
  def __init__(self, y, L=10):
    super().__init__()
    D, N = y.shape
    self.mean = nn.Parameter(torch.randn(size=(L, N)))
    self.scale = nn.Parameter(torch.rand(size=(L, N)))
    self.scale_pf = 1.0

  def forward(self):
    scale = torch.nn.functional.softplus(self.scale) #ensure it's positive
    qF = distributions.Normal(self.mean, scale)
    pF = distributions.Normal(torch.zeros_like(qF.mean), self.scale_pf*torch.ones_like(qF.scale))
    
    return qF, pF
  
  def forward_batched(self, idx):

    scale = torch.nn.functional.softplus(self.scale[:, idx]) #ensure it's positive
    qF = distributions.Normal(self.mean[:, idx], scale)
    pF = distributions.Normal(torch.zeros_like(qF.mean), self.scale_pf*torch.ones_like(qF.scale))

    return qF, pF


class SVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
    
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose random inducing points
   
    self.precompute_distance = False
    # self.S = nn.Parameter(torch.eye(M))
    self.Lu = nn.Parameter(torch.randn((M,M)))
    self.mu = nn.Parameter(torch.zeros((M,)))
    self.constraint = constraints.lower_cholesky
    # self.constraint = constraints.positive_definite

  def precompute_distance(self, X, idz):

    self.precompute_distance = True
    self.Z = nn.Parameter(X[idz], requires_grad=False)
    self.distance = nn.Parameter(torch.cdist(X, self.Z)) #shape N x M
    self.idz = idz

  def kernel_forward(self, X, Z, **args):
    
    return self.kernel(X, Z, **args)
  
  def forward_kernels(self, X, Z, **args):

    Kxx = self.kernel(X, X, diag=True)
    Kzx = self.kernel(self.Z, X)
    Kzz = self.kernel(self.Z, self.Z)

    return Kxx, Kzx, Kzz

  def forward(self, X, verbose=False):

    if verbose:
      print('calculating Kxx')

    Kxx = self.kernel(X, X, diag=True) #shape L x N

    if verbose:
      print('calculating Kzx')

    if self.precompute_distance:

      Kzx = self.kernel.forward_distance(distance_squared=(self.distance.T)**2) #shape L x M x N
    else:
      Kzx = self.kernel(self.Z, X) #shape L x M x N

    if verbose:
      print('calculating kzz')

    if self.precompute_distance:

      Kzx_shape = Kzx.shape
      Kzz = (Kzx.view(-1, Kzx_shape[-2], Kzx_shape[-1]))[:, :, self.idz]
      Kzz = torch.squeeze(Kzz)
    else:
      Kzz = self.kernel_forward(self.Z, self.Z).contiguous() #shape L x M x M
      Kzz = add_jitter(Kzz, self.jitter)

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(Kzz) #shape L x M x M
   
    if verbose:
        print('calculating W')
   
    W = torch.cholesky_solve(Kzx, L) #(Kzz)-1 @ Kzx
    W = torch.transpose(W, -2, -1) # Kxz@(Kzz)-1, shape # L x N x M
    Lu = transform_to(self.constraint)(self.Lu) #shape L x M x M
    S = Lu @ torch.transpose(Lu, -2, -1) # shape L x M x M

    # S = transform_to(self.constraint)(self.S)

    mean, cov_diag = svgp_forward(Kxx, Kzz, W, self.mu, S)
    mean = torch.squeeze(mean)
    
    qF = distributions.Normal(mean, torch.clamp(cov_diag, min=1e-6) ** 0.5)
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU
  

class WSVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
    
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose random inducing points
   
    self.Lu = nn.Parameter(torch.randn((M,M)))
    self.mu = nn.Parameter(torch.zeros((M,)))
    self.constraint = constraints.lower_cholesky


  def kernel_forward(self, X, Z, **args):
    
    return self.kernel(X, Z, **args)
  
  def forward_kernels(self, X, **args):

    Kxx = self.kernel(X, X, diag=True)  #shape L x N
    Kzx = self.kernel(self.Z, X) #shape L x M x N
    Kzz = self.kernel(self.Z, self.Z).contiguous() #shape L x M x M

    return Kxx, Kzx, Kzz

  def forward(self, X, verbose=False, **args):

    if verbose:
      print('calculating kernels')

    Kxx, Kzx, Kzz = self.forward_kernels(X, **args)
    Kzz = add_jitter(Kzz, self.jitter)

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(Kzz) #shape L x M x M
   
    if verbose:
        print('calculating W')
   

    Wt = torch.linalg.solve_triangular(L, Kzx, upper=False)  #(Lzz)-1 @ Kzx
    W = torch.transpose(Wt, -2, -1) # Kxz@(Lzz)-T, shape # L x N x M
    Lu = transform_to(self.constraint)(self.Lu) #shape L x M x M
    # S = Lu @ torch.transpose(Lu, -2, -1) # shape L x M x M

    # S = transform_to(self.constraint)(self.S)

    if verbose:
        print('calculating cov_diag')

    cov_diag = Kxx - torch.sum(W**2, dim=-1)
    cov_diag = torch.clamp(cov_diag, min=0.0)
    cov_diag = cov_diag + torch.sum(((W@Lu)**2), dim=-1)

    # cov_diag = torch.clamp(cov_diag, min=1e-4)

    if verbose:
      print('calculating mean')

    mean = W @ (self.mu.unsqueeze(-1))
    mean = torch.squeeze(mean)
    
    if verbose:
      print(torch.min(cov_diag))

    qF = distributions.Normal(mean, cov_diag ** 0.5)
    qZ = distributions.MultivariateNormal(self.mu, scale_tril=Lu)

    # pZ = distributions.MultivariateNormal(torch.zeros_like(self.Z[:,0]), scale_tril=torch.diag(torch.ones_like(self.Z[:,0])))
    pZ = None
    return qF, qZ, pZ

  def forward_precomputed(self, W, **args):

    Lu = transform_to(self.constraint)(self.Lu) #shape L x M x M

    
    cov_diag = (self.kernel.sigma**2)[:, None] - torch.sum(W**2, dim=-1)
    cov_diag = torch.clamp(cov_diag, min=0.0)
    cov_diag = cov_diag + torch.sum(((W@Lu)**2), dim=-1)

    mean = W @ (self.mu.unsqueeze(-1))
    mean = torch.squeeze(mean)
    qF = distributions.Normal(mean, cov_diag ** 0.5)
    qZ = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pZ = None
    return qF, qZ, pZ




  

class MGGP_SVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4, n_groups=2):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
        
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
    self.groupsZ = nn.Parameter((torch.randint(0, n_groups, (M,))).type(torch.LongTensor), requires_grad=False)
    self.Lu = nn.Parameter(torch.randn((M, M)))
    self.mu = nn.Parameter(torch.zeros((M,)))
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
    Kzz = Kzz.contiguous()

    if verbose:
      print(Kzz.shape)

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
    cov_diag = torch.squeeze(cov_diag)


    qF = distributions.Normal(mean, torch.clamp(cov_diag, min=5e-2) ** 0.5) #setting max cov_diag to 100, need to find a way to clip values manually later
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU
  

class MGGP_WSVGP(WSVGP):
  def __init__(self, kernel, dim=1, M=50, n_groups=2, jitter=1e-4):
    super().__init__(kernel, dim, M, jitter)
    
    self.groupsZ = nn.Parameter((torch.randint(0, n_groups, (M,))).type(torch.LongTensor), requires_grad=False)
   
  
  def forward_kernels(self, X, **args):

    groupsX = args['groupsX']
    Kxx = self.kernel(X, X, groupsX, groupsX, diag=True)
    Kzx = self.kernel(self.Z, X, self.groupsZ, groupsX)
    Kzz = self.kernel(self.Z, self.Z, self.groupsZ, self.groupsZ).contiguous()

    return Kxx, Kzx, Kzz
  