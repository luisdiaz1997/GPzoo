from torch import nn
import torch
import math
import torch.nn.functional as F

class PositiveParameter(nn.Module):
    """
    Parameter constrained to be positive via one of three methods:
    - 'softplus': Store unconstrained, apply softplus to get positive values
    - 'exp': Store unconstrained, apply exp to get positive values  
    - 'projected': Store raw values, project to non-negative after optimizer step
    """
    
    def __init__(self, shape, init_value=1.0, mode='softplus'):
        super().__init__()
        self.mode = mode
        
        if isinstance(shape, int):
            shape = (shape,)
        
        if mode == 'softplus':
            # Inverse softplus for initialization
            init_raw = math.log(math.exp(init_value) - 1) if init_value > 0 else 0.0
            self._raw = nn.Parameter(torch.full(shape, init_raw))
        elif mode == 'exp':
            # Inverse exp (log) for initialization
            init_raw = math.log(init_value) if init_value > 0 else -10.0
            self._raw = nn.Parameter(torch.full(shape, init_raw))
        elif mode == 'projected':
            # Direct initialization, will be projected after steps
            self._raw = nn.Parameter(torch.full(shape, float(init_value)))
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'softplus', 'exp', or 'projected'")
    
    @property
    def value(self):
        """Get the positive-constrained value."""
        if self.mode == 'softplus':
            return F.softplus(self._raw)
        elif self.mode == 'exp':
            return torch.exp(self._raw)
        elif self.mode == 'projected':
            # During forward pass, use raw value (should already be non-negative after projection)
            return self._raw
    
    def project(self):
        """Project parameters to satisfy constraints. Call after optimizer.step()."""
        if self.mode == 'projected':
            with torch.no_grad():
                self._raw.data.clamp_(min=0.0)
    
    def __repr__(self):
        return f"PositiveParameter(shape={tuple(self._raw.shape)}, mode='{self.mode}')"