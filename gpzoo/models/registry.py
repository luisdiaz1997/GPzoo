"""Model registry and checkpoint utilities.

This module centralizes a light-weight registry so callers can request a model
by name (for example, the Slideseq MGGP SVGP) and optionally provide a
checkpoint path to continue training. Each registered builder accepts the
usual tensors (`X`, `Y`, etc.) plus an optional ``state_dict`` that is supplied
automatically when a checkpoint is given. The builder determines how to use
that state (e.g. reusing inducing points) while this module takes care of
actually loading the weights into the returned model.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional, Tuple

import torch
from torch import nn

_ModelBuilder = Callable[..., nn.Module]


_MODEL_REGISTRY: Dict[str, str] = {
    "slideseq/svgp_nsf": "gpzoo.datasets.slideseq.svgp_nsf:create_model",
    "slideseq/vnngp_nsf": "gpzoo.datasets.slideseq.vnngp_nsf:create_model",
    "slideseq/svgp_mggp_nsf": "gpzoo.datasets.slideseq.svgp_mggp_nsf:create_model",
    "slideseq/vnngp_mggp_nsf": "gpzoo.datasets.slideseq.vnngp_mggp_nsf:create_model",
    "tenxvisium/vnngp_nsf": "gpzoo.datasets.tenxvisium.vnngp_nsf:create_model",
    "tenxvisium/vnngp_mggp_nsf": "gpzoo.datasets.tenxvisium.vnngp_mggp_nsf:create_model",
    "liver/vnngp_nsf": "gpzoo.datasets.liver.vnngp_nsf:create_model",
    "liver/vnngp_mggp_nsf": "gpzoo.datasets.liver.vnngp_mggp_nsf:create_model",
    "slideseq/lcgp_nsf": "gpzoo.datasets.slideseq.lcgp_nsf:create_model",
    "slideseq/lcgp_mggp_nsf": "gpzoo.datasets.slideseq.lcgp_mggp_nsf:create_model",
}


def register_model(name: str, target: str) -> None:
    """Register a new model factory under ``name``.

    Parameters
    ----------
    name:
        Registry key used with :func:`build_model`.
    target:
        Import string in the form ``"module_path:function"`` that resolves to a
        callable returning an ``nn.Module``.
    """

    if not name or ":" not in target:
        raise ValueError("target must be in 'module:function' format")
    if name in _MODEL_REGISTRY and _MODEL_REGISTRY[name] != target:
        raise ValueError(f"Model '{name}' is already registered.")
    _MODEL_REGISTRY[name] = target


def available_models(prefix: Optional[str] = None) -> Iterable[str]:
    """Return the sorted list of registered model identifiers."""

    keys = sorted(_MODEL_REGISTRY)
    if prefix is None:
        return keys
    return [k for k in keys if k.startswith(prefix)]


def _resolve_builder(name: str) -> _ModelBuilder:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(available_models())}")
    module_path, func_name = _MODEL_REGISTRY[name].split(":", maxsplit=1)
    module = import_module(module_path)
    builder = getattr(module, func_name, None)
    if builder is None:
        raise AttributeError(f"Builder '{func_name}' not found in module '{module_path}'.")
    return builder


def build_model(
    name: str,
    *,
    checkpoint_path: Optional[str] = None,
    map_location: Optional[str] = None,
    strict: bool = True,
    **kwargs: Any,
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """Instantiate ``name`` and optionally load a checkpoint."""

    builder = _resolve_builder(name)
    state_dict: Optional[MutableMapping[str, torch.Tensor]] = None
    checkpoint_metadata: Optional[Dict[str, Any]] = None
    loaded_checkpoint_path: Optional[str] = None

    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        state_dict = torch.load(str(resolved), map_location=map_location or "cpu")
        kwargs = dict(kwargs)
        kwargs.setdefault("state_dict", state_dict)
        loaded_checkpoint_path = str(resolved)

    model = builder(**kwargs)

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if loaded_checkpoint_path:
            print(f"Loaded checkpoint from {loaded_checkpoint_path}")
        checkpoint_metadata = {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "checkpoint_path": loaded_checkpoint_path or str(Path(checkpoint_path).expanduser().resolve()),
        }

    return model, checkpoint_metadata


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    *,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load ``checkpoint_path`` into ``model`` and return the load summary."""

    resolved = Path(checkpoint_path).expanduser().resolve()
    state_dict = torch.load(str(resolved), map_location=map_location or "cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_path": str(resolved),
    }


__all__ = ["available_models", "build_model", "load_checkpoint", "register_model"]
