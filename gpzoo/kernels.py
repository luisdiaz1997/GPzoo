import torch
import torch.nn as nn

from .utilities import _squared_dist, _torch_sqrt, _embed_distance_matrix


class BaseKernel(nn.Module):
  def __add__(self, other):
    return SumKernel(self, other)
  

class SumKernel(BaseKernel):
    def __init__(self, *kernels):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)

    def forward(self, X, Z, diag=False):
        return sum(k(X, Z, diag=diag) for k in self.kernels)

    def __add__(self, other):
        # Enables (k1 + k2) + k3 to avoid nesting SumKernels
        if isinstance(other, SumKernel):
            return SumKernel(*(self.kernels + other.kernels))
        return SumKernel(*(self.kernels + [other]))


    

class batched_Matern32(BaseKernel):
  def __init__(self, sigma=1.0, lengthscale=2.0):
    super().__init__()

    self.sigma = nn.Parameter(torch.tensor(sigma))
    self.lengthscale = nn.Parameter(torch.tensor(lengthscale))


  def covariance(self, x1, x2, return_distance=False):

    diff = x1-x2
    dist = _torch_sqrt((diff**2).sum())

    val = (3**0.5)*dist/self.lengthscale
    return (self.sigma**2)*(1+val)*torch.exp(-val)


  def forward(self, X, Z, diag=False, return_distance=False):
    
    if diag:
      # return (self.sigma**2).unsqueeze(-1).expand(-1, X.size(0))
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    

    result = torch.vmap(lambda x: torch.vmap(lambda y: self.covariance(x, y))(X))(Z)
    return torch.transpose(result, -1, 0)
  

class batched_separate_MGGP_Matern32(batched_Matern32):
  def __init__(self, sigma=1.0, lengthscale=1.0, group_diff_param=1.0, n_groups=10):
    super().__init__(sigma, lengthscale)

    self.group_diff_param = nn.Parameter(torch.tensor(group_diff_param))
    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def set_group_distances(self, group_distances):

    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def covariance(self, x1, x2, group_embedding1, group_embedding2, return_distance=False):

    diff = x1-x2
    dist = _torch_sqrt((diff**2).sum())

    dist_scaled = dist/torch.abs(self.lengthscale)

    p = x1.unsqueeze(0).shape[-1]

    group_dist = torch.sum((group_embedding1 - group_embedding2) ** 2)

    val = 1/(torch.abs(self.group_diff_param) * group_dist + 1)

    val2 = (3**0.5)*dist_scaled

    cov = (self.sigma**2)*(1+val2)*torch.exp(-val2)*(val**((3+p)/2))

    if return_distance:
      return cov, dist

    return cov


  def forward(self, X, Z, groupsX, groupsZ, diag=False, return_distance=False):
    
    if diag:
      # return (self.sigma**2).unsqueeze(-1).expand(-1, X.size(0))
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    
    
    group_embeddingsX = self.embedding[groupsX]
    group_embeddingsZ = self.embedding[groupsZ]
    

    if return_distance:
      result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy, return_distance=True))(X, group_embeddingsX))(Z, group_embeddingsZ)
      distances = result[1]
      result = result[0]
      return torch.transpose(result, -1, 0), torch.transpose(distances, -1, 0)

    result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy))(X, group_embeddingsX))(Z, group_embeddingsZ)
    return torch.transpose(result, -1, 0)
  

class batched_MGGP_Matern32(batched_Matern32):
  def __init__(self, sigma=1.0, lengthscale=1.0, group_diff_param=1.0, n_groups=10):
    super().__init__(sigma, lengthscale)

    self.group_diff_param = nn.Parameter(torch.tensor(group_diff_param))
    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def set_group_distances(self, group_distances):

    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def covariance(self, x1, x2, group_embedding1, group_embedding2, return_distance=False):

    diff = x1-x2
    dist = _torch_sqrt((diff**2).sum())

    dist_scaled = dist/torch.abs(self.lengthscale)

    p = x1.unsqueeze(0).shape[-1]

    group_dist = torch.sum((group_embedding1 - group_embedding2) ** 2)

    val = 1/(torch.abs(self.group_diff_param) * group_dist + 1)

    val2 = (3**0.5)*dist_scaled*(val**0.5)

    cov = (self.sigma**2)*(1+val2)*torch.exp(-val2)*(val**((3+p)/2))

    if return_distance:
      return cov, dist

    return cov


  def forward(self, X, Z, groupsX, groupsZ, diag=False, return_distance=False):
    
    if diag:
      # return (self.sigma**2).unsqueeze(-1).expand(-1, X.size(0))
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    
    
    group_embeddingsX = self.embedding[groupsX]
    group_embeddingsZ = self.embedding[groupsZ]
    

    if return_distance:
      result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy, return_distance=True))(X, group_embeddingsX))(Z, group_embeddingsZ)
      distances = result[1]
      result = result[0]
      return torch.transpose(result, -1, 0), torch.transpose(distances, -1, 0)

    result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy))(X, group_embeddingsX))(Z, group_embeddingsZ)
    return torch.transpose(result, -1, 0)
  

class batched_Matern52(BaseKernel):
  def __init__(self, sigma=1.0, lengthscale=2.0):
    super().__init__()
    self.sigma = nn.Parameter(torch.tensor(sigma))
    self.lengthscale = nn.Parameter(torch.tensor(lengthscale))

  def covariance(self, x1, x2, return_distance=False):
    diff = x1 - x2
    dist = _torch_sqrt((diff**2).sum())
    val = (5**0.5) * dist / self.lengthscale
    return (self.sigma**2) * (1 + val + val**2 / 3) * torch.exp(-val)

  def forward(self, X, Z, diag=False, return_distance=False):
    if diag:
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    result = torch.vmap(lambda x: torch.vmap(lambda y: self.covariance(x, y))(X))(Z)
    return torch.transpose(result, -1, 0)


class batched_MGGP_Matern52(batched_Matern52):
  def __init__(self, sigma=1.0, lengthscale=1.0, group_diff_param=1.0, n_groups=10):
    super().__init__(sigma, lengthscale)
    self.group_diff_param = nn.Parameter(torch.tensor(group_diff_param))
    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)

  def set_group_distances(self, group_distances):
    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)

  def covariance(self, x1, x2, group_embedding1, group_embedding2, return_distance=False):
    diff = x1 - x2
    dist = _torch_sqrt((diff**2).sum())
    dist_scaled = dist / torch.abs(self.lengthscale)
    p = x1.unsqueeze(0).shape[-1]
    group_dist = torch.sum((group_embedding1 - group_embedding2) ** 2)
    val = 1 / (torch.abs(self.group_diff_param) * group_dist + 1)
    val2 = (5**0.5) * dist_scaled * (val**0.5)
    cov = (self.sigma**2) * (1 + val2 + val2**2 / 3) * torch.exp(-val2) * (val**((5 + p) / 2))
    if return_distance:
      return cov, dist
    return cov

  def forward(self, X, Z, groupsX, groupsZ, diag=False, return_distance=False):
    if diag:
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    group_embeddingsX = self.embedding[groupsX]
    group_embeddingsZ = self.embedding[groupsZ]
    if return_distance:
      result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy, return_distance=True))(X, group_embeddingsX))(Z, group_embeddingsZ)
      distances = result[1]
      result = result[0]
      return torch.transpose(result, -1, 0), torch.transpose(distances, -1, 0)
    result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy))(X, group_embeddingsX))(Z, group_embeddingsZ)
    return torch.transpose(result, -1, 0)


class batched_Brownian(BaseKernel):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def covariance(self, x1, x2):
        # Assume x1, x2 are 1D tensors of the same shape (D,)
        # min(x1, x2) applied elementwise, then summed
        minval = torch.minimum(x1, x2)
        return (self.sigma**2) * minval.sum()

    def forward(self, X, Z, diag=False):
        """
        X: (N, D) - e.g., points where we're evaluating
        Z: (M, D) - e.g., inducing points
        Returns: (M, N) kernel matrix
        """
        if diag:
            return (self.sigma**2).reshape(-1, 1).expand(
                len(self.sigma) if self.sigma.dim() > 0 else 1, len(X)
            )

        result = torch.vmap(lambda z: torch.vmap(lambda x: self.covariance(z, x))(X))(Z)
        return result.transpose(-1, 0)
    

    
class batched_RBF(BaseKernel):
  def __init__(self, sigma=1.0, lengthscale=2.0):
    super().__init__()

    self.sigma = nn.Parameter(torch.tensor(sigma))
    self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
    


  def covariance(self, x1, x2, return_distance=False):

    diff = x1-x2
    dist = (diff**2).sum()

    if return_distance:
      return (self.sigma**2)*torch.exp(-0.5*dist/(self.lengthscale**2)), dist

    return (self.sigma**2)*torch.exp(-0.5*dist/(self.lengthscale**2))


  def forward(self, X, Z, diag=False, return_distance=False):
    
    if diag:
      # return (self.sigma**2).unsqueeze(-1).expand(-1, X.size(0))
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    
    if return_distance:
      result = torch.vmap(lambda x: torch.vmap(lambda y: self.covariance(x, y, return_distance=True))(X))(Z)
      distances = result[1]
      result = result[0]
      return torch.transpose(result, -1, 0), torch.transpose(distances, -1, 0)

    result = torch.vmap(lambda x: torch.vmap(lambda y: self.covariance(x, y))(X))(Z)
    return torch.transpose(result, -1, 0)
  

class batched_MGGP_RBF(batched_RBF):
  def __init__(self, sigma=1.0, lengthscale=1.0, group_diff_param=1.0, n_groups=10):
    super().__init__(sigma, lengthscale)

    self.group_diff_param = nn.Parameter(torch.tensor(group_diff_param))
    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def set_group_distances(self, group_distances):

    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def covariance(self, x1, x2, group_embedding1, group_embedding2, return_distance=False):

    diff = x1-x2
    dist = (diff**2).sum()

    dist_scaled = dist/(self.lengthscale**2)

    p = x1.unsqueeze(0).shape[-1]

    group_dist = torch.sum((group_embedding1 - group_embedding2) ** 2)

    val = 1/(torch.abs(self.group_diff_param) * group_dist + 1)

    if return_distance:
      return (self.sigma**2)*torch.exp(-0.5*dist_scaled*val)*(val**(0.5*p)), dist

    return (self.sigma**2)*torch.exp(-0.5*dist_scaled*val)*(val**(0.5*p))


  def forward(self, X, Z, groupsX, groupsZ, diag=False, return_distance=False):
    
    if diag:
      # return (self.sigma**2).unsqueeze(-1).expand(-1, X.size(0))
      return (self.sigma**2).reshape(-1, 1).expand(len(self.sigma) if isinstance(self.sigma, torch.Tensor) and self.sigma.dim() > 0 else 1, len(X))
    
    
    group_embeddingsX = self.embedding[groupsX]
    group_embeddingsZ = self.embedding[groupsZ]
    

    if return_distance:
      result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy, return_distance=True))(X, group_embeddingsX))(Z, group_embeddingsZ)
      distances = result[1]
      result = result[0]
      return torch.transpose(result, -1, 0), torch.transpose(distances, -1, 0)

    result = torch.vmap(lambda x, gx: torch.vmap(lambda y, gy: self.covariance(x, y, gx, gy))(X, group_embeddingsX))(Z, group_embeddingsZ)
    return torch.transpose(result, -1, 0)

class RBF(nn.Module):
  def __init__(self, sigma=1.0, lengthscale=2.0):
    super().__init__()

    self.sigma = nn.Parameter(torch.tensor(sigma))
    self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
    self.input_dim = 2
    
  def forward(self, X, Z, diag=False, return_distance=False):
    if diag :
        return (self.sigma**2).expand(X.size(0))
    
    distance = torch.cdist(X, Z)
    distance_squared = distance ** 2

    output = self.forward_distance(distance_squared)
    
    if return_distance:
      return output, distance

    return output
  
  def forward_distance(self, distance_squared):
    
    return (self.sigma**2) * torch.exp(-0.5*distance_squared/(self.lengthscale**2))
    

class NSF_RBF(RBF):
  def __init__(self, sigma=1.0, lengthscale=2.0, L=10):
    super().__init__(sigma=sigma, lengthscale=lengthscale)

    self.L = L
    self.sigma = nn.Parameter(sigma*torch.ones((L, 1, 1)))
    self.lengthscale = nn.Parameter(lengthscale*torch.ones((L, 1, 1)))
  
  def forward(self, X, Z, diag=False, return_distance=False):

    if diag:
      return ((self.sigma**2).squeeze())[:, None].expand(-1, X.size(0))

    distance = torch.cdist(X, Z)
    distance_squared = distance ** 2
    distance_squared = (distance_squared[None, :, :]).expand(self.L, -1, -1)

    output = self.forward_distance(distance_squared)

    if return_distance:
      return output, distance

    return self.forward_distance(distance_squared)
  

class MGGP_RBF(RBF):
  def __init__(self, sigma=1.0, lengthscale=2.0, group_diff_param=1.0, n_groups=2, device='cpu'):
    super().__init__(sigma, lengthscale)

    self.group_diff_param = nn.Parameter(torch.tensor(group_diff_param))

    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = _embed_distance_matrix(group_distances).to(device)

  def set_group_distances(self, group_distances):

    self.embedding = _embed_distance_matrix(group_distances)


  def forward(self, X, Z, groupsX, groupsZ, diag=False):
    if diag:
      return (self.sigma**2).expand(X.size(0))


    group_embeddingsX = self.embedding[groupsX]
    group_embeddingsZ = self.embedding[groupsZ]

    group_r2 = _squared_dist(group_embeddingsX, group_embeddingsZ)

    distance_squared = _squared_dist(X, Z)

    assert distance_squared.shape == group_r2.shape


    scale = 1 / (self.group_diff_param * group_r2 + 1)**(0.5*self.input_dim)

    distance_squared = distance_squared/(self.lengthscale**2)

    return self.sigma**2 * torch.exp(-0.5 * distance_squared/ (self.group_diff_param * group_r2 + 1)) * scale


class MGGP_NSF_RBF(NSF_RBF):
  def __init__(self, sigma=1.0, lengthscale=2.0, group_diff_param=1.0, n_groups=2, L=10, device='cpu'):
    super().__init__(sigma, lengthscale, L)

    self.group_diff_param = nn.Parameter(group_diff_param*torch.ones((L, 1, 1)))

    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = nn.Parameter(_embed_distance_matrix(group_distances), requires_grad=False)


  def forward(self, X, Z, groupsX, groupsZ, diag=False):
    if diag:
      return ((self.sigma**2).squeeze())[:, None].expand(-1, X.size(0))


    group_embeddingsX = self.embedding[groupsX]
    group_embeddingsZ = self.embedding[groupsZ]

    group_r2 = _squared_dist(group_embeddingsX, group_embeddingsZ)

    distance_squared = _squared_dist(X, Z)


    group_r2 = (group_r2[None, :, :]).expand(self.L, -1, -1)
    distance_squared = (distance_squared[None, :, :]).expand(self.L, -1, -1)
    assert distance_squared.shape == group_r2.shape


    denom = torch.square(self.group_diff_param) * group_r2 + 1

    scale = 1 / (denom**(0.5*self.input_dim))

    distance_squared = distance_squared/(self.lengthscale**2)

    return self.sigma**2 * torch.exp(-0.5 * distance_squared/ denom) * scale