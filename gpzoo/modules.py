from torch import nn
import torch
import torch.nn.functional as F
from abc import abstractmethod
from typing import Union, Tuple


class ConstrainedParameter(nn.Module):
    """
    Base class for parameters with constraints.
    
    Subclasses implement _to_constrained() and _to_unconstrained() to define
    the bijective mapping between raw (unconstrained) and constrained spaces.
    
    The .data property returns the constrained value, and the underlying
    unconstrained parameter is stored in ._raw (an nn.Parameter).
    
    Args:
        shape: Shape of the constrained parameter.
        mode: Constraint mode (subclass-specific).
    """
    
    def __init__(self, shape: Union[int, Tuple[int, ...]], mode: str):
        super().__init__()
        self.mode = mode
        self._shape = (shape,) if isinstance(shape, int) else tuple(shape)
        
        # Subclass should set self._raw in its __init__ after calling super().__init__
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the constrained parameter."""
        return self._shape
    
    @abstractmethod
    def _to_constrained(self, raw: torch.Tensor) -> torch.Tensor:
        """Convert raw parameter to constrained space."""
        pass
    
    @abstractmethod
    def _to_unconstrained(self, constrained: torch.Tensor) -> torch.Tensor:
        """Convert constrained parameter to raw space."""
        pass
    
    @abstractmethod
    def _init_raw(self) -> torch.Tensor:
        """Initialize raw parameter."""
        pass
    
    # ==================== Common interface ====================
    
    @property
    def data(self) -> torch.Tensor:
        """Returns the constrained value."""
        return self._to_constrained(self._raw)
    
    @data.setter
    def data(self, target: torch.Tensor):
        """Sets parameter from a target constrained tensor."""
        self._raw.data = self._to_unconstrained(target)
    
    @property
    def requires_grad(self) -> bool:
        """Returns requires_grad status of the underlying parameter."""
        return self._raw.requires_grad
    
    @requires_grad.setter
    def requires_grad(self, value: bool):
        """Sets requires_grad on the underlying parameter."""
        self._raw.requires_grad = value
    
    @property
    def device(self) -> torch.device:
        """Returns the device of the underlying parameter."""
        return self._raw.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the underlying parameter."""
        return self._raw.dtype
    
    @property
    def raw(self) -> nn.Parameter:
        """Direct access to the underlying unconstrained parameter."""
        return self._raw
    
    def freeze(self):
        """Freeze the parameter (disable gradient computation)."""
        self._raw.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze the parameter (enable gradient computation)."""
        self._raw.requires_grad = True
    
    def project(self):
        """Project parameters to satisfy constraints. Override for projected modes."""
        pass
    
    # ==================== Tensor-like interface ====================
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        def unwrap(x):
            return x.data if isinstance(x, ConstrainedParameter) else x
        
        new_args = [unwrap(arg) for arg in args]
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)
    
    def __getitem__(self, idx):
        """Allow subscripting to index into the constrained value."""
        return self.data[idx]
    
    def __len__(self):
        return self._shape[0] if self._shape else 1
    
    def dim(self):
        return len(self._shape)
    
    def size(self, dim=None):
        if dim is None:
            return torch.Size(self._shape)
        return self._shape[dim]
    
    def numel(self):
        return self._raw.numel()
    
    def detach(self):
        """Returns a detached copy of the constrained value."""
        return self.data.detach()
    
    def clone(self):
        """Returns a cloned copy of the constrained value."""
        return self.data.clone()
    
    def cpu(self):
        """Move to CPU."""
        self._raw.data = self._raw.data.cpu()
        return self
    
    def cuda(self, device=None):
        """Move to CUDA."""
        self._raw.data = self._raw.data.cuda(device)
        return self
    
    def to(self, *args, **kwargs):
        """Move to device/dtype."""
        self._raw.data = self._raw.data.to(*args, **kwargs)
        return self
    
    def float(self):
        """Convert to float32."""
        self._raw.data = self._raw.data.float()
        return self
    
    def double(self):
        """Convert to float64."""
        self._raw.data = self._raw.data.double()
        return self
    
    def half(self):
        """Convert to float16."""
        self._raw.data = self._raw.data.half()
        return self
    
    def contiguous(self):
        """Returns a contiguous copy of the constrained value."""
        return self.data.contiguous()
    
    def numpy(self):
        """Returns numpy array of the constrained value."""
        return self.data.detach().cpu().numpy()
    
    @property
    def grad(self):
        """Returns the gradient of the underlying parameter."""
        return self._raw.grad
    
    @property 
    def is_cuda(self):
        """Check if on CUDA."""
        return self._raw.is_cuda
    
    @property
    def is_leaf(self):
        """Check if leaf tensor."""
        return self._raw.is_leaf




class CholeskyParameter(ConstrainedParameter):
    """
    Parameter constrained to be a valid Cholesky factor (lower triangular with positive diagonal).
    
    Supports shapes: (M, M) or (L, M, M) for batched matrices.

    Args:
        size: int or tuple. If int, creates (size, size) matrix.
              If tuple like (L, M), creates (L, M, M) batch of matrices.
        mode: 'softplus' or 'exp' for ensuring positive diagonal.
        diagonal_only: If True, only parameterize the diagonal.
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], mode: str = 'softplus', diagonal_only: bool = False):
        if mode not in ['softplus', 'exp']:
            raise ValueError(f"Unknown mode: {mode}. Choose 'softplus' or 'exp'")
        
        # Parse size argument
        if isinstance(size, int):
            self.m = size
            self.batch_size = None
            shape = (size, size)
        else:
            self.batch_size, self.m = size
            shape = (self.batch_size, self.m, self.m)
        
        self.diagonal_only = diagonal_only
        
        super().__init__(shape, mode)
        self._raw = nn.Parameter(self._init_raw())
    
    def _init_raw(self) -> torch.Tensor:
        if self.diagonal_only:
            diag_shape = (self.m,) if self.batch_size is None else (self.batch_size, self.m)
            if self.mode == 'softplus':
                return torch.randn(diag_shape) * 0.1
            else:  # exp
                return torch.log(torch.ones(diag_shape) * 0.1)
        else:
            raw = torch.randn(self._shape) * 0.1
            diag_val = 0.5 if self.mode == 'softplus' else torch.log(torch.tensor(0.1)).item()
            if self.batch_size is None:
                raw.diagonal().fill_(diag_val)
            else:
                for i in range(self.batch_size):
                    raw[i].diagonal().fill_(diag_val)
            return raw
    
    def _to_constrained(self, raw: torch.Tensor) -> torch.Tensor:
        if self.diagonal_only:
            diag = F.softplus(raw) if self.mode == 'softplus' else torch.exp(raw)
            return torch.diag_embed(diag)
        
        diag = torch.diagonal(raw, dim1=-2, dim2=-1)
        lower = torch.tril(raw, diagonal=-1)
        
        if self.mode == 'exp':
            constrained_diag = torch.exp(diag)
        else:  # softplus
            constrained_diag = F.softplus(diag)
        
        return lower + torch.diag_embed(constrained_diag)
    
    def _to_unconstrained(self, constrained: torch.Tensor) -> torch.Tensor:
        if self.diagonal_only:
            diag = torch.diagonal(constrained, dim1=-2, dim2=-1)
            if self.mode == 'softplus':
                return torch.log(torch.exp(diag) - 1)
            else:
                return torch.log(diag)
        
        diag = torch.diagonal(constrained, dim1=-2, dim2=-1)
        lower = torch.tril(constrained, diagonal=-1)
        
        if self.mode == 'exp':
            unconstrained_diag = torch.log(diag)
        else:  # softplus
            unconstrained_diag = torch.log(torch.exp(diag) - 1)
        
        return lower + torch.diag_embed(unconstrained_diag)
    
    def set_constrained_value(self, L_target: torch.Tensor):
        """Backward compatible setter."""
        self.data = L_target
    
    def __repr__(self):
        shape_str = f"{self.m}" if self.batch_size is None else f"({self.batch_size}, {self.m})"
        grad_str = ", requires_grad=True" if self._raw.requires_grad else ""
        return f"CholeskyParameter(size={shape_str}, mode='{self.mode}', diagonal_only={self.diagonal_only}{grad_str})"
    

class PositiveParameter(ConstrainedParameter):
    """
    Parameter constrained to be positive.
    
    Supports any shape - batching works automatically.

    Args:
        shape: int or tuple for the parameter shape.
        mode: 'softplus', 'exp', or 'projected' for ensuring positivity.
    """
    
    def __init__(self, shape: Union[int, Tuple[int, ...]], mode: str = 'softplus'):
        if mode not in ['softplus', 'exp', 'projected']:
            raise ValueError(f"Unknown mode: {mode}. Choose 'softplus', 'exp', or 'projected'")
        
        super().__init__(shape, mode)
        self._raw = nn.Parameter(self._init_raw())
    
    def _init_raw(self) -> torch.Tensor:
        if self.mode == 'projected':
            return torch.rand(self._shape)
        else:
            return torch.randn(self._shape)
    
    def _to_constrained(self, raw: torch.Tensor) -> torch.Tensor:
        if self.mode == 'softplus':
            return F.softplus(raw)
        elif self.mode == 'exp':
            return torch.exp(raw)
        else:  # projected
            return raw
    
    def _to_unconstrained(self, constrained: torch.Tensor) -> torch.Tensor:
        if self.mode == 'softplus':
            # Inverse softplus: log(exp(x) - 1)
            return torch.log(torch.exp(constrained) - 1)
        elif self.mode == 'exp':
            return torch.log(constrained)
        else:  # projected
            return constrained.clamp(min=0.0)
    
    def project(self):
        """Project parameters to satisfy constraints. Call after optimizer.step()."""
        if self.mode == 'projected':
            with torch.no_grad():
                self._raw.data.clamp_(min=0.0)
    
    def __repr__(self):
        grad_str = ", requires_grad=True" if self._raw.requires_grad else ""
        return f"PositiveParameter(shape={self._shape}, mode='{self.mode}'{grad_str})"
