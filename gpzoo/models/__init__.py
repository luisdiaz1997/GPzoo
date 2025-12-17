"""
GPzoo Models Package

This package provides both:
1. Convenience classes for common model configurations (SVGP_NSF, VNNGP_NSF, etc.)
2. The model registry and build_model utility for dataset-specific configurations

Convenience Classes (recommended for most users):
    from gpzoo.models import SVGP_NSF, VNNGP_NSF, MGGP_SVGP_NSF, MGGP_VNNGP_NSF

    model = SVGP_NSF(X, Y, L=12, lengthscale=8.0)
    model = MGGP_VNNGP_NSF(X, Y, groupsX, L=12, K=50, lengthscale=8.0)

Registry (for dataset-specific configurations):
    from gpzoo.models import build_model

    model, _ = build_model("slideseq/svgp_nsf", X=X, Y=Y, L=12, lengthscale=8.0)
"""

from .nsf import (
    SVGP_NSF,
    VNNGP_NSF,
    MGGP_SVGP_NSF,
    MGGP_VNNGP_NSF,
    LCGP_NSF,
    MGGP_LCGP_NSF,
)

from .registry import (
    build_model,
    load_checkpoint,
    register_model,
    available_models,
)

__all__ = [
    # Convenience classes
    "SVGP_NSF",
    "VNNGP_NSF",
    "MGGP_SVGP_NSF",
    "MGGP_VNNGP_NSF",
    "LCGP_NSF",
    "MGGP_LCGP_NSF",
    # Registry utilities
    "build_model",
    "load_checkpoint",
    "register_model",
    "available_models",
]
