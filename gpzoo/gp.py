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
  
    #print("K: ", self.K)
    #print("Z Shape: ", self.Z.shape)
    #print("Lu Shape: ", self.Lu.shape)
    #print("Mu: Shape ", self.mu.shape)


  def forward(self, X, verbose=False, lkzz_build=0):
    Kxx = self.kernel(X, X, diag=True)
    Kxx_shape = Kxx.shape
    Kxx = Kxx.contiguous().view(-1, 1) # (... x N) x 1

    if verbose:
      print("Kxx shape before contiguous().view(-1, 1): ", Kxx_shape)
      print('calculating Kxx')
      print('Kxx.shape', Kxx.shape)
    

    Kxz, distances = self.kernel(X, self.Z, return_distance=True)

    Kxz_shape = Kxz.shape
    Kxz = Kxz.contiguous().view(-1, Kxz_shape[-1]) # (... x N) x M

    if verbose:
      print("Kxz shape before contiguous().view(-1, 1): ", Kxz_shape)
      print("distances shape:: ", distances.shape)
      print('calculating Kxz')
      print('Kxz.shape', Kxz.shape)

    Kzz = self.kernel(self.Z, self.Z)

    Kzz_shape = Kzz.shape
    Kzz = Kzz.contiguous().view(-1, Kzz_shape[-2], Kzz_shape[-1]) # ... x M x M

    if verbose:
      print("Kzz shape before contiguous().view(..): ", Kzz_shape)
      print('calculating kzz')
      print('Kzz.shape', Kzz.shape)

    Lu = transform_to(self.constraint)(self.Lu)
    Lu_shape = Lu.shape
    Lu = Lu.contiguous().view(-1, Lu_shape[-2], Lu_shape[-1]) # ... x M x M
    indexes = torch.argsort(distances, dim=1)[:, :self.K]

    if lkzz_build == 0: # cholesky decomposition
      L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter)) # ... x M x M
      L_shape = L.shape
      L = L.contiguous().view(-1, L_shape[-2], L_shape[-1]) # ... x M x M

      if verbose:
        print("indexes.shape: ", indexes.shape)
        print('calculating L')
        print('L.shape', L.shape)


      indexes = torch.argsort(distances, dim=1)[:, :self.K]
      little_L = L[:, indexes] # ... x N x K x M

      if verbose:
        print('Little_L.shape:', little_L.shape)

      little_Kzz = little_L @ torch.transpose(little_L, -2, -1) # ... x N x K x K
      little_Kzz_shape = little_Kzz.shape
      little_Kzz = little_Kzz.contiguous().view(-1, little_Kzz_shape[-2], little_Kzz_shape[-1]) # ( ... x N) x K x K
    
    elif lkzz_build == 1: # indexing with np.arange into Kzz
      jitter = 1e-6
      Kzz_jittered = Kzz + jitter * torch.eye(Kzz.shape[-1], device=Kzz.device)

      batch_size = Kzz_jittered.shape[:-2]
      N = indexes.shape[-2]
      K = indexes.shape[-1]

      # gather appropriate submatrices
      batch_indices = torch.arange(batch_size[0], device=Kzz.device).view(-1, 1, 1)
      little_L = Kzz_jittered[batch_indices, indexes]

      little_Kzz = little_L @ torch.transpose(little_L, -2, -1)

      little_Kzz_shape = little_Kzz.shape
      little_Kzz = little_Kzz.contiguous().view(-1, little_Kzz_shape[-2], little_Kzz_shape[-1])

      if verbose:
        print("little_Kzz_shape before contiguous(): ", little_Kzz_shape)
        print("little_Kzz: ", little_Kzz.shape)

    
    elif lkzz_build == 2:
      # Directly construct little_Kzz
      def construct_little_Kzz(Z, indexes):
          def kernel_func(i):
              return self.kernel(Z[i], Z[i])
          return torch.vmap(kernel_func)(indexes)
          
      little_Kzz = construct_little_Kzz(self.Z, indexes)

      if verbose:
          print('Little_Kzz.shape:', little_Kzz.shape)

      little_Kzz_shape = little_Kzz.shape
      little_Kzz = little_Kzz.contiguous().view(-1, little_Kzz_shape[-2], little_Kzz_shape[-1])  # ( ... x N) x K x K

    elif lkzz_build == 3:
      if verbose:
          print("Running new lkzz code")
          print("Directly constructing little_Kzz")

      N = X.shape[0]
      K = self.K
      L = Kzz.shape[0]
      lkzz = np.zeros((L, N, K, K))  # L x N x K x K
    
      for l in range(Kzz.shape[0]):
          for n in range(N):
              idx = indexes[n]
              if verbose:
                  print("Kzz[l, idx][:, idx] shape: ", (Kzz[l, idx][:, idx]).shape)
              
              llkzz = Kzz[l, idx][:, idx]  # extracts Kzz[l, idx, idx]
              if verbose:
                  print("lkzz shape: ", lkzz.shape)
              lkzz[l, n, :, :] = llkzz.cpu().numpy()

      little_Kzz = torch.tensor(lkzz, device=Kzz.device).float()
      little_Kzz_shape = little_Kzz.shape
      little_Kzz = little_Kzz.contiguous().view(-1, little_Kzz_shape[-2], little_Kzz_shape[-1])  # ( ... x N) x K x K

    kzz_inv = torch.inverse(add_jitter(little_Kzz, self.jitter)) # (... x N) x KxK

    expanded = indexes.repeat(Kxx_shape[0], 1)

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
    #pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)
    #pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), covariance_matrix=Kzz)
    pU = cov, mean

    return qF, qU, pU


class SVGP2(nn.Module):
  def __init__(self, X, kernel, dim=1, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
    

    M, d = X.shape
    
    self.S = nn.Parameter(torch.eye(M))
    self.mu = nn.Parameter(torch.zeros((M,)))
    self.constraint = constraints.positive_definite

  def forward(self, X, idx, verbose=False):

    Z = X[idx]
    Kzz = self.kernel(Z, Z) #shape ... x M x M

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter)) #shape L x M x M

    S = reshape_param(self.S)
    S = S[:, :, idx]
    S = S[:, idx]
    S = transform_to(self.constraint)(S)
    

    mu = self.mu[None, None, :]
    mu = reshape_param(mu)

    mean = torch.squeeze(mu[:,:, idx])
    S = torch.squeeze(S)
    
    qF = distributions.MultivariateNormal(mean, add_jitter(S, self.jitter))
    pU = distributions.MultivariateNormal(torch.zeros_like(mean), scale_tril=L)

    return qF, qF, pU
  
  def forward_test(self, X, idx, X_test, verbose=False):
    if verbose:
      print('calculating Kxx')
    Kxx = self.kernel(X_test, X_test, diag=True) #shape L x N

    if verbose:
      print('calculating Kzx')

    Z = X[idx]
    Kzx = self.kernel(Z, X_test) #shape L x M x N

    if verbose:
      print('calculating kzz')

    Kzz = self.kernel(Z, Z) #shape L x M x M

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(add_jitter(Kzz, self.jitter)) #shape L x M x M
   
    if verbose:
        print('calculating W')
   
    W = torch.cholesky_solve(Kzx, L) #(Kzz)-1 @ Kzx
    W = torch.transpose(W, -2, -1) # Kxz@(Kzz)-1, shape # L x N x M


    S = reshape_param(self.S)
    S = S[:, :, idx]
    S = S[:, idx]
    S = transform_to(self.constraint)(S)
    

    mu = self.mu[None, None, :]
    mu = reshape_param(mu)
    mu = torch.squeeze(mu[:,:, idx])


    mean, cov_diag = svgp_forward(Kxx, Kzz, W, mu, S)
    mean = torch.squeeze(mean)
    cov_diag = torch.squeeze(cov_diag)
    S = torch.squeeze(S)
    
    qF = distributions.Normal(mean, torch.clamp(cov_diag, min=5e-2) ** 0.5)
    qU = distributions.MultivariateNormal(mu, S)
    pU = distributions.MultivariateNormal(torch.zeros_like(mu), scale_tril=L)

    return qF, qU, pU


class GaussianPrior(nn.Module):
  def __init__(self, y, L=10):
    super().__init__()
    D, N = y.shape
    self.mean = nn.Parameter(torch.randn(size=(L, N)))
    self.scale = nn.Parameter(torch.rand(size=(L, N)))

  def forward(self):
    scale = torch.nn.functional.softplus(self.scale) #ensure it's positive
    qF = distributions.Normal(self.mean, scale)
    pF = distributions.Normal(torch.zeros_like(qF.mean), torch.ones_like(qF.scale))
    
    return qF, pF
  
  def forward_batched(self, idx):

    scale = torch.nn.functional.softplus(self.scale[:, idx]) #ensure it's positive
    qF = distributions.Normal(self.mean[:, idx], scale)
    pF = distributions.Normal(torch.zeros_like(qF.mean), torch.ones_like(qF.scale))

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

    # S = transform_to(self.constraint)(self.S)

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
    cov_diag = torch.squeeze(cov_diag)


    qF = distributions.Normal(mean, torch.clamp(cov_diag, min=5e-2) ** 0.5) #setting max cov_diag to 100, need to find a way to clip values manually later
    qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

    return qF, qU, pU