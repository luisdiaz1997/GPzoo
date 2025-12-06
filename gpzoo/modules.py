from torch import nn
import torch
import torch.nn.functional as F


def _to_unconstrained(L: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert Cholesky factor to unconstrained space."""
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    lower = torch.tril(L, diagonal=-1)
    
    if mode == 'exp':
        unconstrained_diag = torch.log(diag)
    else:  # softplus
        # Inverse softplus: log(exp(x) - 1)
        unconstrained_diag = torch.log(torch.exp(diag) - 1)
    
    return lower + torch.diag_embed(unconstrained_diag)


def _to_constrained(raw: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert unconstrained parameter to valid Cholesky factor."""
    diag = torch.diagonal(raw, dim1=-2, dim2=-1)
    lower = torch.tril(raw, diagonal=-1)
    
    if mode == 'exp':
        constrained_diag = torch.exp(diag)
    else:  # softplus
        constrained_diag = F.softplus(diag)
    
    return lower + torch.diag_embed(constrained_diag)


class CholeskyParameter(nn.Module):
    """
    Parameter constrained to be a valid Cholesky factor.
    
    Supports shapes: (M, M) or (L, M, M) for batched matrices.

    Args:
        size: int or tuple. If int, creates (size, size) matrix.
              If tuple like (L, M), creates (L, M, M) batch of matrices.
        mode: 'softplus' or 'exp' for ensuring positive diagonal.
        diagonal_only: If True, only parameterize the diagonal.
    """
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        def unwrap(x):
            return x.data if isinstance(x, CholeskyParameter) else x
        
        new_args = [unwrap(arg) for arg in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    def __init__(self, size, mode='softplus', diagonal_only=False):
        super().__init__()
        
        # Parse size argument
        if isinstance(size, int):
            self.shape = (size, size)
            self.m = size
            self.batch_size = None
        else:
            self.batch_size, self.m = size
            self.shape = (self.batch_size, self.m, self.m)
        
        self.mode = mode
        self.diagonal_only = diagonal_only
        
        if mode not in ['softplus', 'exp']:
            raise ValueError(f"Unknown mode: {mode}. Choose 'softplus' or 'exp'")
        
        # Initialize unconstrained parameter
        self._raw = nn.Parameter(self._init_raw())
    
    def _init_raw(self) -> torch.Tensor:
        """Initialize raw parameter in unconstrained space."""
        if self.diagonal_only:
            diag_shape = (self.m,) if self.batch_size is None else (self.batch_size, self.m)
            if self.mode == 'softplus':
                return torch.randn(diag_shape) * 0.1
            else:  # exp
                return torch.log(torch.ones(diag_shape) * 0.1)
        else:
            raw = torch.randn(self.shape) * 0.1
            # Set diagonal values
            diag_val = 0.5 if self.mode == 'softplus' else torch.log(torch.tensor(0.1)).item()
            if self.batch_size is None:
                raw.diagonal().fill_(diag_val)
            else:
                for i in range(self.batch_size):
                    raw[i].diagonal().fill_(diag_val)
            return raw
    
    @property
    def data(self) -> torch.Tensor:
        """Returns the constrained Cholesky factor (read-only)."""
        if self.diagonal_only:
            diag = F.softplus(self._raw) if self.mode == 'softplus' else torch.exp(self._raw)
            return torch.diag_embed(diag)
        return _to_constrained(self._raw, self.mode)
    
    @data.setter
    def data(self, L_target: torch.Tensor):
        """Sets parameter from a target Cholesky factor."""
        if self.diagonal_only:
            diag = torch.diagonal(L_target, dim1=-2, dim2=-1)
            if self.mode == 'softplus':
                self._raw.data = torch.log(torch.exp(diag) - 1)
            else:
                self._raw.data = torch.log(diag)
        else:
            self._raw.data = _to_unconstrained(L_target, self.mode)
    
    def set_constrained_value(self, L_target: torch.Tensor):
        """Backward compatible setter."""
        self.data = L_target
    
    def __repr__(self):
        shape_str = f"{self.m}" if self.batch_size is None else f"({self.batch_size}, {self.m})"
        return f"CholeskyParameter(size={shape_str}, mode='{self.mode}', diagonal_only={self.diagonal_only})"


class PositiveParameter(nn.Module):
    """
    Parameter constrained to be positive.
    
    Supports any shape - batching works automatically.

    Args:
        shape: int or tuple for the parameter shape.
        mode: 'softplus', 'exp', or 'projected' for ensuring positivity.
    """
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        def unwrap(x):
            return x.data if isinstance(x, PositiveParameter) else x
        
        new_args = [unwrap(arg) for arg in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    
    def __init__(self, shape, mode='softplus'):
        super().__init__()
        self.mode = mode
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        
        if mode not in ['softplus', 'exp', 'projected']:
            raise ValueError(f"Unknown mode: {mode}. Choose 'softplus', 'exp', or 'projected'")
        
        self._raw = nn.Parameter(self._init_raw())
    
    def _init_raw(self) -> torch.Tensor:
        """Initialize raw parameter in unconstrained space."""
        if self.mode == 'projected':
            return torch.rand(self.shape)
        else:  # exp or softplus
            return torch.randn(self.shape)
    
    @property
    def data(self) -> torch.Tensor:
        """Returns the positive-constrained value."""
        if self.mode == 'softplus':
            return F.softplus(self._raw)
        elif self.mode == 'exp':
            return torch.exp(self._raw)
        elif self.mode == 'projected':
            # During forward pass, use raw value (should already be non-negative after projection)
            return self._raw
    
    @data.setter
    def data(self, target: torch.Tensor):
        """Sets parameter from a target positive tensor."""
        if self.mode == 'softplus':
            # Inverse softplus: log(exp(x) - 1)
            self._raw.data = torch.log(torch.exp(target) - 1)
        elif self.mode == 'exp':
            self._raw.data = torch.log(target)
        elif self.mode == 'projected':
            self._raw.data = target.clamp(min=0.0)
        


    
    def project(self):
        """Project parameters to satisfy constraints. Call after optimizer.step()."""
        if self.mode == 'projected':
            with torch.no_grad():
                self._raw.data.clamp_(min=0.0)
    
    def __repr__(self):
        return f"PositiveParameter(shape={self.shape}, mode='{self.mode}')"