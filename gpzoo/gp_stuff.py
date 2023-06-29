import torch
from torch import distributions, nn
from torch.distributions import constraints, transform_to

#to be organized


def add_jitter(K, jitter=1e-3):
    if K.dim()==2:
        N, _ =  K.shape
        K.view(-1)[::N+1] += jitter
        return K

    if K.dim()==3:
        L, N, _ = K.shape
        K.view(L, -1)[:, ::N+1] += jitter
        
        return K

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

        Kxx = self.kernel(X, X, diag=True)

        if verbose:
            print('calculating Kzx')

        Kzx = self.kernel(self.Z, X)

        if verbose:
            print('calculating kzz')

        Kzz = self.kernel(self.Z, self.Z)

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
        
        if verbose:
            print('calculating predictive mean')

        mean = W@ (self.mu.unsqueeze(-1))
        mean = torch.squeeze(mean)

        if verbose:
            print('calculating predictive covariance')


        cov_diag = Kxx + torch.sum((W@ (S-Kzz))* W, dim=-1)


        qF = distributions.Normal(mean, cov_diag ** 0.5)

        qU = distributions.MultivariateNormal(self.mu, scale_tril=Lu)

        pU = distributions.MultivariateNormal(torch.zeros_like(self.mu), scale_tril=L)

        return qF, qU, pU


class NSF(nn.Module):
    def __init__(self, X, y, kernel, M=50, L=10, jitter=1e-4):
        super().__init__()

    
        D, N = y.shape


        self.kernel = kernel
        self.svgp = SVGP(self.kernel, dim=2, M=M, jitter=jitter)
        self.svgp.Lu = nn.Parameter(5e-2*torch.rand((L, M, M)))
        self.svgp.mu = nn.Parameter(torch.zeros((L, M)))

        self.W = nn.Parameter(torch.rand((D, L)))

        self.V = nn.Parameter(torch.ones((N,)))


    def forward(self, X, E=10, verbose=False):


        qF, qU, pU = self.svgp(X, verbose)
        
        F = qF.rsample((E,)) #shape ExLxN

        F = torch.exp(F)

        Z = torch.matmul(torch.abs(self.W), F) #shape ExDxN
        
        pY = distributions.Poisson(torch.abs(self.V)*Z)


        
        return pY, qF, qU, pU


