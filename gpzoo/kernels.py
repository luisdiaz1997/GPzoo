import torch
from torch import cdist
import numpy as np
from abc import ABC, abstractmethod

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Kernel(ABC):
  def __init__(self, X: torch.Tensor = None, variance=None, lengthscale = None) -> None:
    self.sigma = torch.rand((1,1), dtype=torch.float, device=device, requires_grad= True) if variance is None else torch.tensor([variance**0.5], device=device, requires_grad=True, dtype=torch.float)
    self.lengthscale = torch.rand((1,1), dtype=torch.float, device=device, requires_grad= True) if lengthscale is None else torch.tensor([lengthscale], device=device, requires_grad=True, dtype=torch.float)
    self.X = X
    self.distance = self.build_distance_mat(self.X, self.X) if self.X else None
    self.parameters = [self.sigma, self.lengthscale]
  
  def build_distance_mat(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return cdist(X, Y, p=self.norm)

  @property
  @abstractmethod
  def norm(self)-> int:
    pass

  @abstractmethod
  def build_kernel(self, distance)-> torch.Tensor:
    pass

  def params(self):
    return self.parameters

  def predict(self, X: torch.Tensor = None, Y: torch.Tensor = None):
    '''
    Uses distance matrix with the current data X or with unseen data Y to build kernel matrix.

    Parameters:
      X (torch.Tensor): new training
      Y (torch.Tensor): test data
        
    Returns:
      kernel (torch.Tensor): Kernel of K(X, X) or K(X, Y)
    '''

    if X is None:
      if self.X is None: #If X is still None
        raise RuntimeError('Failed to initialize training data, please initialize self.X or use custom X')
            
      X = self.X
      distance = self.distance if Y is None else self.build_distance_mat(X, Y)
        
    else:
        distance = self.build_distance_mat(X, X) if Y is None else self.build_distance_mat(X, Y)

    return self.build_kernel(distance)

class RBF(Kernel):
  def __init__(self, X: torch.Tensor = None, variance = None, lengthscale = None):
    super(RBF, self).__init__(X, variance, lengthscale)
    self.p = 2

    @property
    def norm(self):
      return self.p

    def build_kernel(self, distance: torch.Tensor) -> torch.Tensor:
        return (self.sigma**2) * torch.exp(-0.5 * (distance/self.lengthscale)**2)
