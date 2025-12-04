"""
GPzoo: Gaussian Process models for spatial transcriptomics.

Quick Start:
    from gpzoo.models import SVGP_NSF, VNNGP_NSF, MGGP_SVGP_NSF, MGGP_VNNGP_NSF

    # Build a model
    model = SVGP_NSF(X, Y, L=12, lengthscale=8.0)

    # Or for multi-group data
    model = MGGP_VNNGP_NSF(X, Y, groupsX, L=12, K=50, lengthscale=8.0)

Modular Components:
    from gpzoo.kernels import batched_Matern32, batched_MGGP_RBF
    from gpzoo.gp import SVGP, VNNGP, MGGP_SVGP, MGGP_VNNGP
    from gpzoo.likelihoods import NSF2, GaussianLikelihood

    # Compose your own model
    kernel = batched_Matern32(sigma=1.0, lengthscale=8.0)
    gp = SVGP(kernel, M=1000)
    model = NSF2(gp, Y, L=12)
"""

from .models import (
    SVGP_NSF,
    VNNGP_NSF,
    MGGP_SVGP_NSF,
    MGGP_VNNGP_NSF,
    build_model,
)

__version__ = "0.1.0"

__all__ = [
    "SVGP_NSF",
    "VNNGP_NSF",
    "MGGP_SVGP_NSF",
    "MGGP_VNNGP_NSF",
    "build_model",
]
