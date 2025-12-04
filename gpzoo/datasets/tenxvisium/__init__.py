"""10x Visium dataset utilities and training entrypoints."""

# Only import torch-dependent modules if torch is available
# This allows benchmark.py to run in SDMBench environment without torch
try:
    from .common import load_visium_with_regions, run_training
    __all__ = ["load_visium_with_regions", "run_training"]
except ImportError:
    # Running in environment without torch (e.g., SDMBench for benchmarking)
    __all__ = []
