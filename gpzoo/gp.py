import torch
from torch import distributions
from torch.distributions import constraints, transform_to
import torch.nn as nn
from .utilities import add_jitter, svgp_forward, reshape_param, whitened_KL, is_lower_triangular
from .modules import CholeskyParameter
import faiss



class BaseVGP(nn.Module):
    def __init__(self, kernel, dim=1, M=50, jitter=1e-4):
        super().__init__()
        self.kernel = kernel
        self.jitter = jitter
        self.return_distance = False # whether to return the distance matrix in forward kernels
        self.dim = dim
        
        self.Z = nn.Parameter(torch.randn((M, dim))) #choose inducing points
        self.Lu = nn.Parameter(torch.randn(M))
        self.mu = nn.Parameter(torch.zeros((M,)))


    def reshape_input_data(self, **kwargs):
        """
        Ensure each input is 2D: (N, D) or (N, 1)
        """

        X = kwargs.get("X")
        Z = kwargs.get("Z")

        if X.dim() == 1:
            X = X.unsqueeze(-1)
        if Z.dim() == 1:
            Z = Z.unsqueeze(-1)

        return X, Z


    def reshape_parameters(self, mu, Lu, covariances, verbose=False):
        """Reshape the parameters to ensure they are compatible for solving."""

        Kxx, Kzx, Kzz = covariances
        kzz = add_jitter(Kzz, self.jitter)  # Add jitter to Kzz for numerical stability

        return mu, Lu, Kxx, Kzx, Kzz


    def transform_variables(self, mu, Lu, L, verbose=False):
        """Transform the parameters into whitened space."""

        return mu, Lu

    def flatten_to_solve(self, x, keepdim=-1):

        if keepdim >= 0:
            raise ValueError("keepdim must be negative")
        
        shape = x.shape
        preserved = shape[keepdim:]
        batch_dims = shape[:keepdim]

        flattened = int(torch.prod(torch.tensor(batch_dims)))

        x_flattened = x.reshape(flattened, *preserved) #(Batch, *preserved)
        x_flattened = x_flattened.transpose(0, 1)  # Transpose to have preserved dimensions first
        x_flattened = x_flattened.reshape(preserved[0], -1)  # Flatten the first dimension

        return x_flattened, shape
    
    def unflatten_from_solve(self, x_flattened, shape, keepdim=-1):

        preserved = shape[keepdim:]
        batch_dims = shape[:keepdim]
        flattened = int(torch.prod(torch.tensor(batch_dims)))

        x = x_flattened.reshape(preserved[0], flattened, *preserved[1:])  # Reshape to have preserved dimensions first
        x = x.transpose(0, 1)  # (flattened, preserved[0], preserved[1:])
        x = x.reshape(*batch_dims, *preserved)  # Reshape to original dimensions
        return x


    def Lu_is_diagonal(self):

        return self.Lu.ndim == 1 or (self.Lu.ndim > 1 and (self.Lu.shape[-1] != self.Lu.shape[-2]))

    

    def apply_constraints(self):
        """Apply constraints to all parameters that require it."""

        if self.Lu_is_diagonal():
            Lu = torch.nn.functional.softplus(self.Lu)
        else:
            Lu = transform_to(constraints.lower_cholesky)(self.Lu)

        mu = self.mu
        return mu, Lu
    

    def forward_kernels(self, X, diag=True, **kwargs):

        X, Z = self.reshape_input_data(X=X, Z=self.Z, **kwargs)
        batch_shape = X.shape[:-2]  # B0, ..., Bn

        X_flat = X.reshape(-1, *X.shape[-2:])  # Flatten batch dimensions
        Z_flat = Z.reshape(-1, *Z.shape[-2:])  # Flatten batch dimensions

        Kxx_flat = torch.vmap(lambda x: self.kernel(x, x, diag=diag))(X_flat)
        Kzx_flat = torch.vmap(
                lambda z, x: self.kernel(z, x, return_distance=self.return_distance)
        )(Z_flat, X_flat)
        Kzz_flat = torch.vmap(lambda z: self.kernel(z, z))(Z_flat)

        # Reshape back to original batch shape
        Kxx = Kxx_flat.reshape(*batch_shape, *Kxx_flat.shape[1:]).contiguous()  # Shape: B0, ..., Bn, N, N
        Kzx = Kzx_flat.reshape(*batch_shape, *Kzx_flat.shape[1:]).contiguous()  # Shape: B0, ..., Bn, M, N
        Kzz = Kzz_flat.reshape(*batch_shape, *Kzz_flat.shape[1:]).contiguous()  # Shape: B0, ..., Bn, M, M


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



class original_VNNGP(nn.Module):
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

        qF = distributions.Normal(mean, torch.clamp(cov, min=5e-2, max=100.0) ** 0.5)
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
    

class WSVGP(BaseVGP):
    """Whitened Sparse Variational GP
    
    Uses CholeskyParameter for clean constraint handling.
    
    Args:
            kernel: GP kernel function
            dim: Input dimension
            M: Number of inducing points
            jitter: Jitter for numerical stability
            diagonal_only: If True, use diagonal covariance (default: False)
            cholesky_mode: 'exp' or 'softplus' for diagonal constraint (default: 'exp')
    
    The Lu attribute is a CholeskyParameter. Access patterns:
            - model.Lu.data                  -> constrained Cholesky factor (for computation)
            - model.Lu.requires_grad = True  -> control gradient
            - model.Lu.freeze() / unfreeze() -> convenience methods
    """
    def __init__(self, kernel, dim=1, M=50, jitter=1e-4, diagonal_only=False, cholesky_mode='exp'):
        super().__init__(kernel, dim, M, jitter)
        
        self.diagonal_only = diagonal_only
        self.cholesky_mode = cholesky_mode
        
        # Replace manual Lu parameter with CholeskyParameter
        # Remove the inherited Lu from BaseVGP and replace with CholeskyParameter
        del self.Lu
        self.Lu = CholeskyParameter(M, mode=cholesky_mode, diagonal_only=diagonal_only)

    def Lu_is_diagonal(self):
        """Check if Lu is diagonal."""
        return self.diagonal_only

    def apply_constraints(self):
        """Apply constraints to all parameters that require it.
        
        CholeskyParameter handles the constraint internally via the .data property.
        """
        Lu = self.Lu.data  # Already constrained
        mu = self.mu
        return mu, Lu

    
    def forward(self, X, diag=True, verbose=False, **kwargs):
        if verbose:
                print('→ Computing kernels')

        covariances = self.forward_kernels(X, diag=diag, **kwargs)
        if verbose:
                print('→kernels computed')

        if verbose: print('applying constraints')

        mu, Lu = self.apply_constraints()

        if verbose: print('done with constraints')
        mu, Lu, Kxx, Kzx, Kzz = self.reshape_parameters(mu, Lu, covariances, verbose=verbose)

        if verbose:
                print('→ Cholesky(Kzz)')
        L = torch.linalg.cholesky(Kzz) # Shape: L x M x M

        if verbose:
                print('→ Solving for W')
        Wt = torch.linalg.solve_triangular(L, Kzx, upper=False)  # (L)^-1 @ Kzx, Shape: L x M x N
        W = Wt.transpose(-2, -1)  # Kxz @ (L)^-T, Shape: L x N x M
        if verbose:
                print('W shape:', W.shape)
        
        if verbose:
                print('Lu lower_cholesky:', is_lower_triangular(Lu))

        mu, Lu = self.transform_variables(mu, Lu, L, verbose=verbose)
        if verbose:
            print('Lu lower_cholesky:', is_lower_triangular(Lu))

        if verbose:
                print('→ Computing mean')
        mean = W @ mu.unsqueeze(-1)  # Shape: N x 1
        mean = mean.squeeze(-1)           # Shape: N
        if verbose:
                print('→ Mean shape:', mean.shape)

        if diag:
                if verbose:
                        print('→ Diagonal covariance')
                cov_diag = Kxx
                if verbose:
                    print('cov_diag shape:', cov_diag.shape)
                cov_diag = cov_diag - torch.sum(W**2, dim=-1)
                if verbose:
                    print('cov_diag shape:', cov_diag.shape)
                cov_diag = torch.clamp(cov_diag, min=0.0)

                cov_diag = cov_diag + torch.sum((W @ Lu) ** 2, dim=-1)

                if verbose:
                    print('cov_diag shape:', cov_diag.shape)
                # Clamp covariance to prevent numerical explosion
                cov_diag = torch.clamp(cov_diag, min=0.0, max=100.0)
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

        qZ = distributions.MultivariateNormal(mu, scale_tril=Lu)
        pZ = None
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

        Lu = self.Lu.data  # Get constrained value
        
        cov_diag = (self.kernel.sigma**2)[:, None] - torch.sum(W**2, dim=-1)
        cov_diag = torch.clamp(cov_diag, min=0.0)
        cov_diag = cov_diag + torch.sum(((W@Lu)**2), dim=-1)
        # Clamp covariance to prevent numerical explosion
        cov_diag = torch.clamp(cov_diag, min=0.0, max=100.0)

        mean = W @ (self.mu.unsqueeze(-1))
        mean = torch.squeeze(mean)
        qF = distributions.Normal(mean, cov_diag ** 0.5)
        qZ = distributions.MultivariateNormal(self.mu, scale_tril=Lu)
        pZ = None
        return qF, qZ, pZ
    

    def forward_train(self, X, diag=True, verbose=False, **kwargs):
        return self.forward(X, diag=diag, verbose=verbose, **kwargs)
    
    def set_Lu_from_cholesky(self, L_target: torch.Tensor):
        """Set Lu from a target Cholesky factor (convenience method)."""
        self.Lu.data = L_target


class SVGP(WSVGP):
    """Standard (non-whitened) Sparse Variational GP
    
    Inherits CholeskyParameter handling from WSVGP.
    """
    def __init__(self, kernel, dim=1, M=50, jitter=1e-4, diagonal_only=False, cholesky_mode='exp'):
        super().__init__(kernel, dim, M, jitter, diagonal_only, cholesky_mode)


    def transform_variables(self, mu, Lu, L, verbose=False):
        """Transform the parameters to ensure the correct constraints."""

        if L.dim() == 2:
            mu_stacked, mu_shape = self.flatten_to_solve(mu, keepdim=-1)
            if verbose:
                print('mu_stacked shape:', mu_stacked.shape)
                print('mu_shape:', mu_shape)

            Lu_stacked, Lu_shape = self.flatten_to_solve(Lu, keepdim=-2)
            if verbose:
                print('Lu_stacked shape:', Lu_stacked.shape)
                print('Lu_shape:', Lu_shape)

            stacked = torch.cat([mu_stacked, Lu_stacked], dim=-1)  # [M, B1 + B2*M]
            if verbose:
                print('Stacked shape:', stacked.shape)
            X = torch.linalg.solve_triangular(L, stacked, upper=False) #L^-1 @ mu, Lu

            if verbose:
                print('X shape:', X.shape)
            # Unstack the result
            B1 = mu_stacked.shape[-1]

            mu_X = X[:, :B1]
            Lu_X = X[:, B1:]

            if verbose:
                print('mu_X shape:', mu_X.shape)
                print('Lu_X shape:', Lu_X.shape)

            mu = self.unflatten_from_solve(mu_X, mu_shape, keepdim=-1)
            Lu = self.unflatten_from_solve(Lu_X, Lu_shape, keepdim=-2)

        
        else:

            B = mu.shape[:-1]
            M = mu.shape[-1]
            
            if verbose:
                print('mu shape:', mu.shape)
                print('Lu shape:', Lu.shape)
                print('L shape:', L.shape)

            stacked = torch.cat([mu.unsqueeze(-1), Lu], dim=-1)
            X = torch.linalg.solve_triangular(L, stacked, upper=False)

            mu = X[..., 0]  
            Lu = X[..., 1:]

        return mu, Lu


class VNNGP(SVGP):
    def __init__(self, kernel, dim=1, M=50, jitter=1e-4, K=3, diagonal_only=False, cholesky_mode='exp'):
        # Call parent but we'll override Lu
        super().__init__(kernel, dim, M, jitter, diagonal_only, cholesky_mode)
        self.K = K
        self.knn_idx = None  # Store the indices of the K nearest neighbors
        self.knn_idz = None
        self.inducing_points = True

        # Override Lu with a different shape for VNNGP (raw nn.Parameter, not CholeskyParameter)
        del self.Lu
        self.Lu = nn.Parameter(torch.randn((M, K)))

    def apply_constraints(self):

        mu = self.mu
        Lu = self.Lu

        return mu, Lu

    def calculate_knn(self, X):

        """Calculate the K+1 nearest neighbors of X in Z using FAISS."""

        X_np = X.detach().cpu().float().numpy()
        Z = self.Z.detach().cpu().float().numpy()
        index = faiss.IndexFlatL2(Z.shape[-1])
        index.add(Z)
        distances, indices = index.search(X_np, self.K+1)

        return torch.tensor(indices, dtype=torch.long, device='cpu')


    def reshape_input_data(self, **kwargs):
        """
        Ensure each input is 2D: (N, D) or (N, 1)
        """

        X = kwargs.get("X")
        Z = kwargs.get("Z")
        knn_idx = kwargs.get("knn_idx", self.knn_idx)

        X, Z = super(SVGP, self).reshape_input_data(X=X, Z=Z)
        X = X.unsqueeze(-2)

        Z = Z[knn_idx]
        return X, Z

    def reshape_parameters(self, mu, Lu, covariances, verbose=False, knn_idx=None):
        """Reshape the parameters to ensure they are compatible for solving."""

        if knn_idx is None:
            knn_idx = self.knn_idx

        Kxx, Kzx, Kzz = covariances

        if verbose:
            print('adding jitter to Kzz')
        Kzz = add_jitter(Kzz, self.jitter)  # Add jitter to Kzz for numerical stability
        if verbose:
            print('done with jitter')
        mu_shape = mu.shape  # e.g., [*B, M]
        mu_reshaped = mu.reshape(-1, mu_shape[-1])  # [B, M] or [1, M] if unbatched

        # knn_idx.T: [N, K] to index M
        mu_knn = mu_reshaped[..., knn_idx]  # [B, N, K] — broadcasting works
        # Restore shape: [*B, N, K]
        mu = mu_knn.reshape(*mu_shape[:-1], knn_idx.shape[0], self.K)

        Lu_shape = Lu.shape  # [*B, M, M]
        Lu_reshaped = Lu.reshape(-1, Lu_shape[-2], Lu_shape[-1])  # [B, M] or [1, M, M]
        Lu_knn = Lu_reshaped[:, knn_idx]  # [B, N, K, M]

        Su_knn = Lu_knn @ Lu_knn.transpose(-2, -1)  # Shape: B x N x K x K

        Su_knn = Su_knn.reshape(*Lu_shape[:-2], knn_idx.shape[0], self.K, self.K)  # [*B, N, K, K]

        #   # Add jitter for numerical stability
        Su_knn = add_jitter(Su_knn, self.jitter)
        Lu = torch.linalg.cholesky(Su_knn)  # Cholesky decomposition, Shape: B x N x K x K

        if verbose:
            print('computing little Lu')
            print('Lu_knn shape:', Lu_knn.shape)
            print('Lu.shape:', Lu.shape)

        Kxx=Kxx.squeeze(-1)

        if verbose:
            print('Reshaped mu:', mu.shape)
            print('Reshaped Lu:', Lu.shape)
            print('Reshaped Kzx:', Kzx.shape)
            print('Reshaped Kzz:', Kzz.shape)
            print('Reshaped Kxx:', Kxx.shape)


        return mu, Lu, Kxx, Kzx, Kzz
    
    def forward_train(self, X, idx=None, diag=True, verbose=False, **kwargs):

        if idx is not None:

            mean = self.mu[:, idx]
            cov = torch.sum(self.Lu[:, idx]**2, dim=-1)

        else:
            mean = self.mu
            cov = torch.sum(self.Lu**2, dim=-1)


        cov = torch.clamp(cov, min=1e-3, max=100.0)
        scale = cov**0.5

        qF = distributions.Normal(mean, scale)

        return qF, None, None
    
    def forward(self, X, diag=True, verbose=False, **kwargs):
        
        qF, qZ, pZ = super(SVGP, self).forward(X, diag=diag, verbose=verbose, **kwargs)
        qF = distributions.Normal(qF.mean.squeeze(-1), qF.scale.squeeze(-1))

        return qF, qZ, pZ
    
    

    def kl_divergence_full(self, qZ, pZ=None, verbose=False, idx=None, **kwargs):

        if idx is not None:
            Z = self.Z[idx]
            if hasattr(self, 'groupsZ'):
                groupsZ = self.groupsZ[idx]

            knn_idx = self.knn_idz[idx]
        else:
            Z = self.Z
            if hasattr(self, 'groupsZ'):
                groupsZ = self.groupsZ

            knn_idx = self.knn_idz

        

        if hasattr(self, 'groupsZ'):
            covariances = self.forward_kernels(X=Z, groupsX=groupsZ, diag=True, knn_idx=knn_idx)
        else:
            covariances = self.forward_kernels(X=Z, diag=True, knn_idx=knn_idx)


        mu, Lu = self.apply_constraints()

        if idx is not None:
            # fix this for single GPs later
            mu_diag = mu[:, idx]
            Lu_diag = Lu[:, idx]
        else:
            mu_diag = mu
            Lu_diag = Lu



        mu_knn, Lu_knn, Kxx, Kzx, Kzz = self.reshape_parameters(mu, Lu, covariances, knn_idx=knn_idx, verbose=verbose)

        L = torch.linalg.cholesky(Kzz) # Shape: L x k x k
        Wt = torch.linalg.solve_triangular(L, Kzx, upper=False)  # (L)^-1 @ Kzx, Shape: L x k x k
        W = Wt.transpose(-2, -1)  # Kxz @ (L)^-T, Shape: B x N x 1 x K
        mu_tranformed, Lu_transfomed = self.transform_variables(mu_knn, Lu_knn, L, verbose=verbose) #B x N x k, B x N x K x K


        Lu_knn_2 = Lu[:, knn_idx] # B x N x K x K
        S_knnj = Lu_knn_2 @ (Lu_diag.unsqueeze(-1)) # B x N x K x 1

        W2t = torch.linalg.solve_triangular(Lu_knn, S_knnj, upper=False) # B x N x K x 1
        W2 = W2t.transpose(-2, -1)  #  Sj k @ (L)^-T, Shape: B x N x 1 x K 
        WLu =  W @ Lu_transfomed # B x N x 1 x K


        abterm = torch.sum(W2 * WLu, dim=-1)



        mean = W @ mu_tranformed.unsqueeze(-1)  # Shape: B x N x 1 x 1
        mean = mean.squeeze(-1)           # Shape: B x N x 1

        cov_diff = Kxx - torch.sum(W**2, dim=-1) #shape Bx N x 1
        cov_diff = torch.clamp(cov_diff, min=1e-6, max=100.0) # ensure positive variance




        mean_diff = mu_diag.unsqueeze(-1) - mean #shape Bx N x 1
        cov_knn = torch.sum(WLu**2, dim=-1)

        S_diag = torch.sum(Lu_diag**2, dim=-1).unsqueeze(-1) # shape B x N x 1
        S_term = torch.sum(W2**2, dim=-1) # shape B x N x 1
        S_diff = S_diag-S_term

        S_diff = torch.clamp(S_diff, min=1e-6, max=100.0)  # Ensure positive variance

        if verbose:
            print('mean_diff shape:', mean_diff.shape)
            print('cov_diff shape:', cov_diff.shape)
            print('S_diag shape:', S_diag.shape)
            print('S_term shape:', S_term.shape)
            print('cov_knn shape:', cov_knn.shape)
            print('abterm shape:', abterm.shape)
            print('S_diff shape:', S_diff.shape)


        if verbose:
            print('mean_diff min:', mean_diff.min())
            print('cov_diff min:', cov_diff.min())
            print('S_diag min:', S_diag.min())
            print('S_term min:', S_term.min())
            print('cov_knn min:', cov_knn.min())
            print('abterm min:', abterm.min())
            print('S_diff min:', S_diff.min())


        KL =  torch.log(cov_diff)-torch.log(S_diff) -1 +  (mean_diff**2  + S_diag + cov_knn - 2*abterm)/cov_diff

        KL = 0.5*KL

        return torch.sum(KL)
        


        
class LCGP(SVGP):
    """Locally Conditioned Gaussian Process
    
    Uses LowRankPlusDiagonal covariance parameterization S = D + VV^T
    with efficient KL computation via Woodbury identity.
    
    Args:
        kernel: GP kernel function
        dim: Input dimension
        M: Number of inducing points
        jitter: Jitter for numerical stability
        K: Number of nearest neighbors for local conditioning
        rank: Rank R of low-rank component V ∈ ℝ^{M × R}
        diag_mode: 'softplus' or 'exp' for diagonal constraint
    """
    
    def __init__(
        self, 
        kernel, 
        dim=1, 
        M=50, 
        jitter=1e-4, 
        K=3, 
        rank=None,
        diag_mode='softplus'
    ):
        # Call SVGP but we'll override Lu
        super().__init__(kernel, dim, M, jitter, diagonal_only=False, cholesky_mode='exp')
        self.K = K
        self.knn_idx = None  # (N, K) neighbors for data points
        self.knn_idz = None  # (M, K) neighbors for inducing points
        self.inducing_points = True
        
        # Override Lu with LowRankPlusDiagonal
        # For L latent GPs, batch_size = L
        del self.Lu
        self.Lu = LowRankPlusDiagonal(m=M, rank=rank, batch_size=None, diag_mode=diag_mode)
    
    def apply_constraints(self):
        """Lu constraints handled internally by LowRankPlusDiagonal."""
        mu = self.mu
        return mu, self.Lu
    
    def calculate_knn(self, X):
        """Calculate the K nearest neighbors of X in Z using FAISS."""
        X_np = X.detach().cpu().float().numpy()
        Z = self.Z.detach().cpu().float().numpy()
        index = faiss.IndexFlatL2(Z.shape[-1])
        index.add(Z)
        distances, indices = index.search(X_np, self.K + 1)
        return torch.tensor(indices, dtype=torch.long, device='cpu')
    
    def forward_train(self, X, idx=None, diag=True, verbose=False, **kwargs):
        """Fast forward pass using local parameterization directly."""
        if idx is not None:
            mean = self.mu[idx]
            # S_jj = D_j + ||V_j||^2
            cov = self.Lu.D[idx] + self.Lu.V.diag()[idx]
        else:
            mean = self.mu
            cov = self.Lu.D + self.Lu.V.diag()
        
        cov = torch.clamp(cov, min=1e-3, max=100.0)
        scale = cov ** 0.5
        
        qF = distributions.Normal(mean, scale)
        return qF, None, None
    
    def kl_divergence_full(self, qZ=None, pZ=None, verbose=False, idx=None, **kwargs):
        """
        Compute locally conditioned KL divergence.
        
        KL(q(U) || p(U)) ≈ Σ_j E_{q(U_{n(j)})}[KL(q(U_j|U_{n(j)}) || p(U_j|U_{n(j)}))]
        
        Using the simplified form:
        = Σ_j 1/2 [log(τ_j²/τ̃_j²) + s_jj/τ_j² + (m_j - β_j^T m_{n(j)})²/τ_j² 
                   + β_j^T S_{n(j)n(j)} β_j/τ_j² - 2α_j^T S_{n(j)n(j)} β_j/τ_j² - 1]
        """
        if idx is not None:
            Z = self.Z[idx]
            knn_idx = self.knn_idz[idx]
            if hasattr(self, 'groupsZ'):
                groupsZ = self.groupsZ[idx]
        else:
            Z = self.Z
            knn_idx = self.knn_idz
            if hasattr(self, 'groupsZ'):
                groupsZ = self.groupsZ
        
        # Get kernel covariances for prior
        if hasattr(self, 'groupsZ'):
            covariances = self.forward_kernels(X=Z, groupsX=groupsZ, diag=True, knn_idx=knn_idx)
        else:
            covariances = self.forward_kernels(X=Z, diag=True, knn_idx=knn_idx)
        
        Kxx, Kzx, Kzz = covariances
        
        # Add jitter and compute Cholesky for prior
        Kzz = add_jitter(Kzz, self.jitter)
        L_prior = torch.linalg.cholesky(Kzz)  # (M', K, K)
        
        # Prior: β_j = K_{jn(j)} K_{n(j)n(j)}^{-1}, τ_j² = k_jj - K_{jn(j)} K_{n(j)n(j)}^{-1} K_{n(j)j}
        # W_prior = K_{jn(j)} L_prior^{-T} so β_j = W_prior @ L_prior^{-1}
        # and τ_j² = k_jj - ||W_prior||²
        Kzx_t = Kzx.transpose(-2, -1)  # (M', 1, K)
        W_prior_t = torch.linalg.solve_triangular(L_prior, Kzx_t.transpose(-2, -1), upper=False)
        W_prior = W_prior_t.transpose(-2, -1)  # (M', 1, K)
        
        # τ_j² (prior conditional variance)
        Kxx = Kxx.squeeze(-1) if Kxx.dim() > 1 else Kxx
        tau_sq = Kxx - (W_prior ** 2).sum(dim=-1).squeeze(-1)  # (M',)
        tau_sq = torch.clamp(tau_sq, min=1e-6)
        
        # Get variational parameters
        mu = self.mu
        if idx is not None:
            mu_j = mu[idx]
        else:
            mu_j = mu
        
        # Get neighbors' mu
        mu_neighbors = mu[knn_idx]  # (M', K)
        
        # β_j^T m_{n(j)} via whitened prior
        # First whiten mu_neighbors
        mu_neighbors_whitened = torch.linalg.solve_triangular(
            L_prior, 
            mu_neighbors.unsqueeze(-1),  # (M', K, 1)
            upper=False
        ).squeeze(-1)  # (M', K)
        
        # β_j^T m_{n(j)} = W_prior @ L_prior^{-1} @ m_{n(j)} = W_prior @ mu_neighbors_whitened
        beta_m = (W_prior.squeeze(-2) * mu_neighbors_whitened).sum(dim=-1)  # (M',)
        
        # Mean difference: (m_j - β_j^T m_{n(j)})²
        mean_diff_sq = (mu_j - beta_m) ** 2  # (M',)
        
        # Get variational covariance quantities from LowRankPlusDiagonal
        D_inv, Psi_cholesky = self.Lu.get_precision_components()
        tau_tilde_sq, alpha = self.Lu.get_conditional_params(
            knn_idx=knn_idx, 
            D_inv=D_inv, 
            Psi_cholesky=Psi_cholesky,
            idx=idx
        )  # tau_tilde_sq: (M',), alpha: (M', K)
        
        # s_jj = D_j + ||V_j||²
        if idx is not None:
            s_jj = self.Lu.D[idx] + (self.Lu.V.data[idx] ** 2).sum(dim=-1)
        else:
            s_jj = self.Lu.D + (self.Lu.V.data ** 2).sum(dim=-1)
        
        # S_{n(j)n(j)} block for each j
        S_neighbors = self.Lu.get_block(knn_idx)  # (M', K, K)
        
        # β_j^T S_{n(j)n(j)} β_j
        # β_j = W_prior @ L_prior^{-1}, so in whitened space β_whitened = W_prior.squeeze(-2)
        # β^T S β = β_whitened^T (L_prior^{-1} S L_prior^{-T}) β_whitened
        # But we need to be careful - let's compute directly
        
        # Whiten S_neighbors: L_prior^{-1} S_{n(j)n(j)} L_prior^{-T}
        S_whitened = torch.linalg.solve_triangular(L_prior, S_neighbors, upper=False)
        S_whitened = torch.linalg.solve_triangular(
            L_prior, 
            S_whitened.transpose(-2, -1), 
            upper=False
        ).transpose(-2, -1)  # (M', K, K)
        
        # β_j in whitened space is just W_prior.squeeze(-2)
        beta_whitened = W_prior.squeeze(-2)  # (M', K)
        
        # β^T S β
        beta_S_beta = (beta_whitened.unsqueeze(-2) @ S_whitened @ beta_whitened.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # α_j^T S_{n(j)n(j)} β_j
        # α is already in original space, need to compute α^T S β
        # α^T S_{n(j)n(j)} β = α^T S_{n(j)n(j)} K_{n(j)n(j)}^{-1} K_{n(j)j}
        # In whitened space: α^T L_{n(j)} L_{n(j)}^T L^{-1} L^{-T} K_{n(j)j}
        #                  = α^T L_{n(j)} @ (L^{-1} K_{n(j)j})  -- but K_{n(j)j} = Kzx
        
        # Simpler: α^T S β where S = S_neighbors, β comes from prior
        # First compute S @ β (in original space)
        # β = K_{jn(j)} K_{n(j)n(j)}^{-1} = (L^{-1} K_{n(j)j})^T @ L^{-1}
        # Actually let's just compute it directly
        
        # K_{n(j)n(j)}^{-1} K_{n(j)j} = L^{-T} L^{-1} Kzx^T
        Kzx_vec = Kzx.squeeze(-2)  # (M', K)
        beta_orig = torch.cholesky_solve(Kzx_vec.unsqueeze(-1), L_prior).squeeze(-1)  # (M', K)
        
        # S @ β
        S_beta = (S_neighbors @ beta_orig.unsqueeze(-1)).squeeze(-1)  # (M', K)
        
        # α^T S β
        alpha_S_beta = (alpha * S_beta).sum(dim=-1)  # (M',)
        
        # KL formula (simplified form from derivation):
        # KL = 1/2 [log(τ²/τ̃²) + s_jj/τ² + (m_j - β^T m)²/τ² + β^T S β/τ² - 2 α^T S β/τ² - 1]
        
        KL = 0.5 * (
            torch.log(tau_sq) - torch.log(tau_tilde_sq)
            + s_jj / tau_sq
            + mean_diff_sq / tau_sq
            + beta_S_beta / tau_sq
            - 2 * alpha_S_beta / tau_sq
            - 1
        )
        
        if verbose:
            print(f'tau_sq: {tau_sq.min():.4f} to {tau_sq.max():.4f}')
            print(f'tau_tilde_sq: {tau_tilde_sq.min():.4f} to {tau_tilde_sq.max():.4f}')
            print(f's_jj: {s_jj.min():.4f} to {s_jj.max():.4f}')
            print(f'mean_diff_sq: {mean_diff_sq.min():.4f} to {mean_diff_sq.max():.4f}')
            print(f'beta_S_beta: {beta_S_beta.min():.4f} to {beta_S_beta.max():.4f}')
            print(f'alpha_S_beta: {alpha_S_beta.min():.4f} to {alpha_S_beta.max():.4f}')
            print(f'KL per point: {KL.min():.4f} to {KL.max():.4f}')
        
        return torch.sum(KL)
  
     
class MGGP:
    def __init__(self, n_groups, M):
        self.groupsZ = nn.Parameter(
                torch.randint(0, n_groups, (M,)).type(torch.LongTensor), 
                requires_grad=False
        )

    def reshape_input_data(self, **kwargs):

        X, Z = kwargs.pop("X"), kwargs.pop("Z")
        groupsX, groupsZ = kwargs.pop("groupsX"), kwargs.pop("groupsZ")

        X, Z = super().reshape_input_data(X=X, Z=Z, **kwargs)

        groupsX, groupsZ = super().reshape_input_data(X=groupsX, Z=groupsZ, **kwargs)
        groupsX = groupsX.squeeze(-1)
        groupsZ = groupsZ.squeeze(-1)

        return X, Z, groupsX, groupsZ



    def forward_kernels(self, X, groupsX, diag=True, **kwargs):
        groupsZ = self.groupsZ
        X, Z, groupsX, groupsZ = self.reshape_input_data(X=X, Z=self.Z, groupsX=groupsX, groupsZ=groupsZ, **kwargs)
        batch_shape = X.shape[:-2]  # B0, ..., Bn

        # Flatten all inputs
        X_flat = X.reshape(-1, *X.shape[-2:])             # (B, N, D)
        Z_flat = Z.reshape(-1, *Z.shape[-2:])             # (B, M, D)
        groupsX_flat = groupsX.reshape(-1, X.shape[-2])   # (B, N)
        groupsZ_flat = groupsZ.reshape(-1, Z.shape[-2])   # (B, M)

        # Batched kernel calls via vmap
        Kxx_flat = torch.vmap(lambda x, gx: self.kernel(x, x, gx, gx, diag=diag))(
                X_flat, groupsX_flat
        )
        Kzx_flat = torch.vmap(
                lambda z, x, gz, gx: self.kernel(z, x, gz, gx, return_distance=self.return_distance)
        )(Z_flat, X_flat, groupsZ_flat, groupsX_flat)

        Kzz_flat = torch.vmap(lambda z, gz: self.kernel(z, z, gz, gz))(Z_flat, groupsZ_flat)

        # Reshape back to original batch shape
        Kxx = Kxx_flat.reshape(*batch_shape, *Kxx_flat.shape[1:]).contiguous()  # Shape: B0, ..., Bn, N, N
        Kzx = Kzx_flat.reshape(*batch_shape, *Kzx_flat.shape[1:]).contiguous()  # Shape: B0, ..., Bn, M, N
        Kzz = Kzz_flat.reshape(*batch_shape, *Kzz_flat.shape[1:]).contiguous()  # Shape: B0, ..., Bn, M, M

        return Kxx, Kzx, Kzz


def MGGPWrapper(BaseGPClass):
        class MGGPModel(MGGP, BaseGPClass):
                def __init__(self, kernel, dim=1, M=50, jitter=1e-4, n_groups=2, **kwargs):
                        BaseGPClass.__init__(self, kernel, dim, M, jitter, **kwargs)
                        MGGP.__init__(self, n_groups, M)
        MGGPModel.__name__ = f"MGGP_{BaseGPClass.__name__}"
        return MGGPModel


MGGP_WSVGP = MGGPWrapper(WSVGP)
MGGP_SVGP = MGGPWrapper(SVGP)
MGGP_VNNGP = MGGPWrapper(VNNGP)