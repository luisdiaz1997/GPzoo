import torch
from torch import nn
from .utilities import add_jitter

'''
Impliementation of Natural Gradient Descent (NGD) for variational inference borrowed from GPyTorch.
'''

def _triangular_inverse(A, upper=False):
    eye = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return torch.linalg.solve_triangular(A, eye, upper=upper)


def _phi_for_cholesky_(A):
    """Modifies A in-place for Cholesky derivative."""
    A.tril_().diagonal(dim1=-2, dim2=-1).mul_(0.5)
    return A


def _cholesky_backward(dout_dL, L, L_inv):
    """Backward pass through Cholesky decomposition."""
    A = L.transpose(-1, -2) @ dout_dL
    phi = _phi_for_cholesky_(A)
    grad_input = (L_inv.transpose(-1, -2) @ phi) @ L_inv
    return grad_input.add(grad_input.transpose(-1, -2)).mul_(0.5)


class NaturalToMuLu(torch.autograd.Function):
    """
    Custom autograd function that:
    - Forward: converts natural params (θ₁, θ₂) → (μ, Lu)
    - Backward: returns gradients w.r.t. expectation params (η₁, η₂)
    
    This makes the Euclidean gradient (what the optimizer sees) equal to
    the natural gradient.
    """
    
    @staticmethod
    def forward(ctx, theta1, theta2, jitter=1e-6):
        """
        θ₁ = S⁻¹μ
        θ₂ = -½S⁻¹
        
        Returns μ, Lu where S = Lu @ Lu.T
        """
        # S⁻¹ = -2θ₂
        S_inv = -2.0 * theta2
        
        # Cholesky of S⁻¹ to get L_inv where S⁻¹ = L_inv @ L_inv.T
        
        L_inv = torch.linalg.cholesky(add_jitter(S_inv, jitter))

        
        # L = L_inv⁻¹, so S = L @ L.T
        Lu = _triangular_inverse(L_inv, upper=False)
        
        # S = Lu @ Lu.T
        S = Lu @ Lu.transpose(-1, -2)
        
        # μ = S @ θ₁
        mu = (S @ theta1.unsqueeze(-1)).squeeze(-1)
        
        # Save for backward
        ctx.save_for_backward(mu, Lu)
        
        return mu, Lu
    
    @staticmethod
    def backward(ctx, dout_dmu, dout_dLu):
        """
        Returns gradients w.r.t. η₁, η₂ (expectation params),
        NOT θ₁, θ₂ (natural params).
        
        This is the key trick that makes NGD work!
        """
        mu, Lu = ctx.saved_tensors
        L_inv = _triangular_inverse(Lu, upper=False)
        
        # Gradient w.r.t. S from the Cholesky factor
        dout_dS = _cholesky_backward(dout_dLu, Lu, L_inv)
        
        # η₁ = μ
        # η₂ = S + μμᵀ
        # So S = η₂ - μμᵀ
        # 
        # ∂L/∂η₁ = ∂L/∂μ + ∂L/∂S · ∂S/∂μ
        #        = ∂L/∂μ - 2·(∂L/∂S)·μ
        # ∂L/∂η₂ = ∂L/∂S
        
        dout_deta1 = dout_dmu - 2.0 * (dout_dS @ mu.unsqueeze(-1)).squeeze(-1)
        dout_deta2 = dout_dS
        
        return dout_deta1, dout_deta2, None  # None for jitter


class NaturalVariationalDistribution(nn.Module):
    """
    Variational distribution parameterized by natural parameters.
    
    Uses custom backward to enable natural gradient descent.
    """
    
    def __init__(self, M, batch_shape=(), jitter=1e-6, mean_init_std=1e-3):
        super().__init__()
        self.M = M
        self.jitter = jitter
        self.batch_shape = batch_shape
        
        # Natural parameters
        # θ₁ = S⁻¹μ, initialized to 0 (corresponds to μ=0)
        # θ₂ = -½S⁻¹, initialized to -½I (corresponds to S=I)
        theta1_init = torch.zeros(*batch_shape, M)
        theta2_init = -0.5 * torch.eye(M).expand(*batch_shape, M, M).clone()
        
        # Add small noise to break symmetry
        theta1_init = theta1_init + mean_init_std * torch.randn_like(theta1_init)
        
        self.theta1 = nn.Parameter(theta1_init)
        self.theta2 = nn.Parameter(theta2_init)
    
    def forward(self):
        """Returns (μ, Lu) for use in SVGP."""
        mu, Lu = NaturalToMuLu.apply(self.theta1, self.theta2, self.jitter)
        return mu, Lu
    
    @torch.no_grad()
    def initialize_from_prior(self, Kzz):
        """Initialize so that q(u) = N(0, Kzz)."""
        Kzz_inv = torch.inverse(Kzz)
        self.theta1.zero_()
        self.theta2.copy_(-0.5 * Kzz_inv)
    
    @property
    def mu(self):
        """Get current mean (for inspection)."""
        with torch.no_grad():
            S_inv = -2.0 * self.theta2
            S = torch.inverse(S_inv)
            return (S @ self.theta1.unsqueeze(-1)).squeeze(-1)
    
    @property
    def S(self):
        """Get current covariance (for inspection)."""
        with torch.no_grad():
            S_inv = -2.0 * self.theta2
            return torch.inverse(S_inv)


class NGD(torch.optim.Optimizer):
    """
    Natural Gradient Descent optimizer.
    
    Works with NaturalVariationalDistribution which returns gradients
    w.r.t. expectation parameters in its backward pass.
    
    Args:
        params: Parameters to optimize (should be theta1, theta2 from 
                NaturalVariationalDistribution)
        num_data: Total number of data points (for scaling minibatch gradients)
        lr: Learning rate (can often use lr=1.0 with natural gradients)
    """
    
    def __init__(self, params, num_data, lr=0.1):
        self.num_data = num_data
        super().__init__(params, defaults=dict(lr=lr))
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Single optimization step.
        
        Because the backward pass of NaturalVariationalDistribution returns
        ∂L/∂η (not ∂L/∂θ), and the natural gradient ∇̃_θ = ∂L/∂η,
        we can simply do: θ ← θ - lr * grad
        """
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                # The grad is already ∂L/∂η (natural gradient for θ)
                # Scale by num_data because ELBO is typically averaged
                p.add_(p.grad, alpha=-lr * self.num_data)
        
        return None