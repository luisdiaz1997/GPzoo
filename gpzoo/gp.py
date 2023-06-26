import torch
import numpy as np
from torch import distributions, nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SVGP:
  def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
      super().__init__()

      self.kernel = kernel
      self.jitter = jitter
        
      self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
      self.Lu = nn.Parameter(torch.randn((M, M)))
      self.mu = nn.Parameter(torch.zeros((M,)))