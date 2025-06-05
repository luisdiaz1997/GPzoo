import torch
from torch import distributions
from torch.distributions import constraints, transform_to
import torch.nn as nn
from .utilities import add_jitter, svgp_forward, reshape_param, whitened_KL



class BaseVGP(nn.Module):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__()
    self.kernel = kernel
    self.jitter = jitter
    
    self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
    self.Lu = nn.Parameter(torch.randn(M))
    self.mu = nn.Parameter(torch.zeros((M,)))

  def forward_kernels(self, X, diag=True, **kwargs):
    Kxx = self.kernel(X, X, diag=diag)  # shape L x N
    if not diag:
        Kxx = Kxx.contiguous()

    Kzx = self.kernel(self.Z, X)  # shape L x M x N
    Kzz = self.kernel(self.Z, self.Z).contiguous()  # shape L x M x M

    return Kxx, Kzx, Kzz
  
  def kl_divergence(self, qZ, pZ=None):

    if pZ is None:
      mean, scale_tril = qZ.mean, qZ.scale_tril
      batch_shape = qZ.mean.shape[:-1]
      M = qZ.mean.shape[-1]
      mean_flat = mean.reshape(-1, M)
      scale_tril_flat = scale_tril.reshape(-1, M, M)
      kl_flat = torch.vmap(whitened_KL)(mean_flat, scale_tril_flat)
      kl_term = kl_flat.reshape(batch_shape)
    else:
      kl_term = distributions.kl_divergence(qZ, pZ)

    return kl_term



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



# class SVGP(BaseVGP):
#   def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
#     super().__init__(kernel, dim, M, jitter)
    
   
#     self.precompute_distance = False
#     self.Lu = nn.Parameter(torch.randn((M,M)))
#     self.constraint = constraints.lower_cholesky


#   def precompute_distance(self, X, idz):

#     self.precompute_distance = True
#     self.Z = nn.Parameter(X[idz], requires_grad=False)
#     self.distance = nn.Parameter(torch.cdist(X, self.Z)) #shape N x M
#     self.idz = idz

#   def forward(self, X, diag=True, verbose=False):
#     if verbose:
#         print('calculating Kxx')

#     Kxx = self.kernel(X, X, diag=diag)  # shape L x N (diag=True) or L x N x N (diag=False)

#     if verbose:
#         print('calculating Kzx')

#     if self.precompute_distance:
#         Kzx = self.kernel.forward_distance(distance_squared=(self.distance.T)**2)  # shape L x M x N
#     else:
#         Kzx = self.kernel(self.Z, X)  # shape L x M x N

#     if verbose:
#         print('calculating Kzz')

#     if self.precompute_distance:
#         Kzx_shape = Kzx.shape
#         Kzz = (Kzx.view(-1, Kzx_shape[-2], Kzx_shape[-1]))[:, :, self.idz]
#         Kzz = torch.squeeze(Kzz)
#     else:
#         Kzz = self.kernel_forward(self.Z, self.Z).contiguous()  # shape L x M x M
#         Kzz = add_jitter(Kzz, self.jitter)

#     if verbose:
#         print('calculating cholesky')
#     L = torch.linalg.cholesky(Kzz)  # shape L x M x M

#     if verbose:
#         print('calculating W')

#     W = torch.cholesky_solve(Kzx, L)  # (Kzz)^-1 @ Kzx
#     W = torch.transpose(W, -2, -1)  # Kxz @ (Kzz)^-1, shape L x N x M
#     Lu = transform_to(self.constraint)(self.Lu)  # shape L x M x M
#     S = Lu @ torch.transpose(Lu, -2, -1)  # shape L x M x M

#     if diag:
#         mean, cov_diag = svgp_forward(Kxx, Kzz, W, self.mu, S)
#         mean = torch.squeeze(mean)
#         cov_diag = torch.clamp(cov_diag, min=1e-6)
#         qF = distributions.Normal(mean, cov_diag.sqrt())
#     else:
#         if verbose:
#             print('calculating full covariance')

#         # Optimized covariance computation
#         A = W  # W = K_{zx}^T K_{zz}^{-1} => shape: L x N x M
#         middle = Kzz - S  # L x M x M
#         cov = Kxx - A @ middle @ A.transpose(-2, -1)  # L x N x N
#         cov = cov.contiguous()
#         cov = add_jitter(cov, self.jitter)

#         # Cholesky decomposition for scale_tril
#         L_cov = torch.linalg.cholesky(cov)  # L x N x N
#         mean = torch.squeeze(W @ self.mu.unsqueeze(-1))  # L x N
#         qF = distributions.MultivariateNormal(mean, scale_tril=L_cov)

#     qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
#     pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

#     return qF, qU, pU
  

class WSVGP(BaseVGP):
  """Whitened Sparse Variational GP"""
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__(kernel, dim, M, jitter)

    self.Lu = nn.Parameter(torch.randn((M,M)))
    self.constraint = constraints.lower_cholesky

  def transform_variables(self, L):
    """Transform the parameters to ensure the correct constraints."""
    Lu = transform_to(self.constraint)(self.Lu)
    mu = self.mu
    return mu, Lu

  
  def forward(self, X, diag=True, verbose=False, **kwargs):
    if verbose:
        print('→ Computing kernels')

    Kxx, Kzx, Kzz = self.forward_kernels(X, diag=diag, **kwargs)
    Kzz = add_jitter(Kzz, self.jitter)

    if verbose:
        print('→ Cholesky(Kzz)')
    L = torch.linalg.cholesky(Kzz) # Shape: L x M x M

    if verbose:
        print('→ Solving for W')
    Wt = torch.linalg.solve_triangular(L, Kzx, upper=False)  # (L)^-1 @ Kzx, Shape: L x M x N
    W = Wt.transpose(-2, -1)  # Kxz @ (L)^-T, Shape: L x N x M
    
    mu, Lu = self.transform_variables(L)

    if verbose:
        print('→ Computing mean')
    mean = W @ mu.unsqueeze(-1)  # Shape: N x 1
    mean = mean.squeeze(-1)           # Shape: N

    if diag:
        if verbose:
            print('→ Diagonal covariance')
        cov_diag = Kxx - torch.sum(W**2, dim=-1)
        cov_diag = torch.clamp(cov_diag, min=0.0)
        cov_diag += torch.sum((W @ Lu) ** 2, dim=-1)
        qF = distributions.Normal(mean, cov_diag.sqrt())
    else:
        if verbose:
            print('→ Full covariance')
        Kxzzx = W @ Wt
        WLu = W @ Lu
        cov = Kxx - Kxzzx + WLu @ WLu.transpose(-2, -1)
        cov = add_jitter(cov, self.jitter)
        L_cov = torch.linalg.cholesky(cov)
        qF = distributions.MultivariateNormal(mean, scale_tril=L_cov)

    qZ = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
    M = self.mu.shape[-1]
    eye = torch.eye(M, device=self.mu.device).expand_as(self.Lu)
    pZ = distributions.MultivariateNormal(
        torch.zeros_like(self.mu), scale_tril=eye
    )
    return qF, qZ, pZ   
     

  
  def init_mean(self, X, mean, verbose=False, **args):

    if verbose:
      print('calculating kernels')

    Kxx, Kzx, Kzz = self.forward_kernels(X, diag=False, **args)
    Kzz = add_jitter(Kzz, self.jitter)

    if verbose:
      print('calculating cholesky')
    L = torch.linalg.cholesky(Kzz) #shape L x M x M
   
    if verbose:
        print('calculating W')
   

    Wt = torch.linalg.solve_triangular(L, Kzx, upper=False)  #(Lzz)-1 @ Kzx
    W = torch.transpose(Wt, -2, -1) # Kxz@(Lzz)-T, shape # L x N x M

    if verbose:
      print('W shape:', W.shape)
      print('mean shape:', mean.shape)

    mu_init = torch.linalg.lstsq(W, mean).solution

    return mu_init


  def forward_precomputed(self, W, **kwargs):

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
  

class SVGP(WSVGP):
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
    super().__init__(kernel, dim, M, jitter)

  def transform_variables(self, L):
    """Transform the parameters to ensure the correct constraints."""

    Lu = transform_to(self.constraint)(self.Lu)
    mu = self.mu

    mu_batch_shape = mu.shape[:-1]
    Lu_batch_shape = Lu.shape[:-2]
    M = mu.shape[-1]

    # Reshape mu and Lu to ensure they are compatible for solving
    mu = mu.reshape(*mu_batch_shape, M, 1)  # Shape: [..., M, 1]
    Lu = Lu.reshape(*Lu_batch_shape, M, M)  # Shape: [..., M, M]

    mu = mu.reshape(-1, M, 1)  # Flatten batch dimensions for solving
    Lu = mu.reshape(-1, M, M)  # Flatten batch dimensions for solving

    # Solve the linear system L @ X = rhs, where rhs = [mu, Lu]
    # Copilot, stack my and Lu so Shape: [..., M, mu_batch_shape + M*Lu_batch_shape] 
    
    



    # rhs = torch.cat([mu.unsqueeze(-1), Lu], dim=-1)
    # X = torch.linalg.solve_triangular(L, rhs, upper=False)  # [..., M, 1 + M]

    
    mu = torch.linalg.solve_triangular(L, mu, upper=False) # L^-1 @ mu, Shape: L x M
    Lu = torch.linalg.solve_triangular(L, Lu, upper=False) # L^-1 @ Lu, Shape: L x M x M
    # mu = X[..., 0]  # Extract the first column as mu
    # Lu = X[..., 1:]  # Extract the rest as Lu

    return mu, Lu
  
     
class MGGP:
  def __init__(self, n_groups, M):
    self.groupsZ = nn.Parameter(
        torch.randint(0, n_groups, (M,)).type(torch.LongTensor), 
        requires_grad=False
    )

  def forward_kernels(self, X, diag, **kwargs):
    groupsX = kwargs['groupsX']
    Kxx = self.kernel(X, X, groupsX, groupsX, diag=diag)
    if not diag:
      Kxx = Kxx.contiguous()
    
    Kzx = self.kernel(self.Z, X, self.groupsZ, groupsX)
    Kzz = self.kernel(self.Z, self.Z, self.groupsZ, self.groupsZ).contiguous()

    return Kxx, Kzx, Kzz


def MGGPWrapper(BaseGPClass):
    class MGGPModel(MGGP, BaseGPClass):
        def __init__(self, kernel, dim=1, M=50, jitter=1e-4, n_groups=2):
            BaseGPClass.__init__(self, kernel, dim, M, jitter)
            MGGP.__init__(self, n_groups, M)
    MGGPModel.__name__ = f"MGGP_{BaseGPClass.__name__}"
    return MGGPModel


MGGP_WSVGP = MGGPWrapper(WSVGP)
MGGP_SVGP = MGGPWrapper(SVGP)

    


  
# class MGGP_WSVGP(WSVGP):
#   def __init__(self, kernel, dim=1, M=50, n_groups=2, jitter=1e-4):
#     super().__init__(kernel, dim, M, jitter)
    
#     self.groupsZ = nn.Parameter((torch.randint(0, n_groups, (M,))).type(torch.LongTensor), requires_grad=False)
   
  
#   def forward_kernels(self, X, diag, **kwargs):

#     groupsX = kwargs['groupsX']
#     Kxx = self.kernel(X, X, groupsX, groupsX, diag=diag)
#     if not diag:
#       Kxx = Kxx.contiguous()
    
#     Kzx = self.kernel(self.Z, X, self.groupsZ, groupsX)
#     Kzz = self.kernel(self.Z, self.Z, self.groupsZ, self.groupsZ).contiguous()

#     return Kxx, Kzx, Kzz
  


