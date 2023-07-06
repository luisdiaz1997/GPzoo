import torch
import torch.nn as nn

from .utilities import _squared_dist, _torch_sqrt, _embed_distance_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RBF(nn.Module):
  def __init__(self, sigma=1.0, lengthscale=2.0):
    super().__init__()

    self.sigma = nn.Parameter(torch.tensor(sigma))
    self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
    self.input_dim = 2
    
  def forward(self, X, Z, diag=True):
    if diag :
        return (self.sigma**2).expand(X.size(0))
    

    distance_squared = _squared_dist(X, Z)

    return self.sigma**2 * torch.exp(-0.5*distance_squared/(self.lengthscale**2))
    

class NSF_RBF(nn.Module):
  def __init__(self, sigma=1.0, lengthscale=2.0, L=10):
    super().__init__()

    self.L = L
    self.sigma = nn.Parameter(sigma*torch.ones((L, 1, 1)))
    self.lengthscale = nn.Parameter(lengthscale*torch.ones((L, 1, 1)))
    self.input_dim = 2
  
  def forward(self, X, Z, diag=False):

    if diag:
      return ((self.sigma**2).squeeze())[:, None].expand(-1, X.size(0))

    distance_squared = _squared_dist(X, Z)
    distance_squared = (distance_squared[None, :, :]).expand(self.L, -1, -1)
    return self.sigma**2 * torch.exp(-0.5*distance_squared/(self.lengthscale**2))
  

class MGGP_RBF(RBF):
  def __init__(self, sigma=1.0, lengthscale=2.0, group_diff_param=1.0, n_groups=2):
    super().__init__(sigma, lengthscale)

    self.group_diff_param = nn.Parameter(torch.tensor(group_diff_param))

    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
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
  def __init__(self, sigma=1.0, lengthscale=2.0, group_diff_param=1.0, n_groups=2, L=10):
    super().__init__(sigma, lengthscale, L)

    self.group_diff_param = nn.Parameter(group_diff_param*torch.ones((L, 1, 1)))

    group_distances = torch.ones(n_groups) - torch.eye(n_groups)
    self.embedding = _embed_distance_matrix(group_distances)


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


    scale = 1 / (self.group_diff_param * group_r2 + 1)**(0.5*self.input_dim)

    distance_squared = distance_squared/(self.lengthscale**2)

    return self.sigma**2 * torch.exp(-0.5 * distance_squared/ (self.group_diff_param * group_r2 + 1)) * scale