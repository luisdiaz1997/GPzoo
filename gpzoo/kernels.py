import torch
from torch import cdist
import numpy as np
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RBF(nn.Module):
  def __init__(self, sigma=1.0, lengthscale=2.0):
    super().__init__()

    self.sigma = nn.Parameter(torch.tensor(sigma))
    self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
    
  def forward(self, X, Z, diag=True):
    if diag :
        return (self.sigma**2).expand(X.size(0))
    distance = torch.cdist(X, Z)
    return self.sigma**2 * torch.exp(-0.5*(distance/self.lengthscale)**2)
    

class NSF_RBF(nn.Module):
  def __init__(self, sigma=1.0, lengthscale=2.0, L=10):
    super().__init__()

    self.L = L
    self.sigma = nn.Parameter(sigma*torch.ones((L, 1, 1)))
    self.lengthscale = nn.Parameter(lengthscale*torch.ones((L, 1, 1)))
  
  def forward(self, X, Z, diag=False):

    if diag:
      return ((self.sigma**2).squeeze())[:, None].expand(-1, X.size(0))

    distance = torch.cdist(X, Z)
    distance = (distance[None, :, :]).expand(self.L, -1, -1)
    return self.sigma**2 * torch.exp(-0.5*(distance/self.lengthscale)**2)