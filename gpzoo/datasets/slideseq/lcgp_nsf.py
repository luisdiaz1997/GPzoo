from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import optim

from gpzoo.models import LCGP_NSF
from torch.utils.tensorboard import SummaryWriter


def _infer_k(state_dict: Mapping[str, torch.Tensor]) -> Optional[int]:
    # LCGP stores K in prior.K but primarily it's a hyperparam
    # We can try to infer from Lu if it was plain VNNGP, but for LCGP Lu is (L, M, R).
    # K is needed for KNN indices.
    # LCGP generally keeps K in model.prior.K (if stored), lets assume passed or default.
    return None


def create_model(
    *,
    X: torch.Tensor,
    Y: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    L: int = 12,
    lengthscale: float = 20.0,
    kernel_kwargs: Optional[Mapping[str, float]] = None,
    jitter: float = 1e-5,
    K: Optional[int] = None,
    rank: Optional[int] = None,
    diag_mode: str = "softplus",
    inducing_points: Optional[torch.Tensor] = None,
    # LCGP specific args
    scale_multiplier: float = 1e-6,
    precompute_knn: bool = True,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    loadings_mode: str = "softplus",
):
    """Build or resume an LCGP NSF model for Slideseq."""

    sigma = 1.0
    if kernel_kwargs and "sigma" in kernel_kwargs:
        sigma = kernel_kwargs["sigma"]

    if state_dict is not None and inducing_points is None:
        try:
            inducing_points = state_dict["prior.Z"].detach().clone()
        except KeyError as exc:
            raise KeyError("Checkpoint missing prior.Z for LCGP resume.") from exc
        # We don't strictly infer K from state_dict for LCGP easily without custom logic
        pass

    if K is None:
        K = 50

    return LCGP_NSF(
        X=X,
        Y=Y,
        V=V,
        L=L,
        K=K,
        rank=rank,
        lengthscale=lengthscale,
        sigma=sigma,
        jitter=jitter,
        inducing_points=inducing_points,
        scale_multiplier=scale_multiplier,
        diag_mode=diag_mode,
        precompute_knn=precompute_knn,
        device=device,
        seed=seed,
        loadings_mode=loadings_mode,
    )


def _train_main() -> None:
    from gpzoo.models import build_model
    from gpzoo.training_utilities import (
        freeze_svgp_kernel_and_inputs,
        train_lcgp_batched_with_tracking,
        create_sequential_lr_scheduler,
    )
    from gpzoo.datasets.slideseq.common import (
        DEVICE,
        OUTPUT_DIR,
        load_slideseq_with_groups,
        run_training,
    )
    from gpzoo.datasets.slideseq.config import (
        LCGP_CHECKPOINT,
        IMAGE_LOG_EVERY,
        JITTER,
        L_FACTORS,
        LENGTHSCALE,
        LENGTHSCALE_TRAIN_AFTER,
        LOADINGS_MODE,
        LR,
        LR_LENGTHSCALE,
        LR_LOADING,
        LR_MEAN,
        LR_SCALE,
        SCALE_MULTIPLIER,
        SCALE_TRAIN_AFTER,
        SEED,
        STEPS,
        USE_SCHEDULER,
        LCGP_K,
        LCGP_RANK,
        LCGP_DIAG_MODE,
        X_BATCH,
        Y_BATCH,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, Y, _, _, V_init = load_slideseq_with_groups()

    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tb" / f"slideseq_lcgp_k={LCGP_K}")

    checkpoint_path = LCGP_CHECKPOINT
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    start_step = 0

    model, _ = build_model(
        "slideseq/lcgp_nsf",
        checkpoint_path=checkpoint_arg,
        X=X,
        Y=Y,
        V=V_init,
        L=L_FACTORS,
        lengthscale=LENGTHSCALE,
        kernel_kwargs={"sigma": 1.0},
        jitter=JITTER,
        K=LCGP_K,
        rank=LCGP_RANK,
        diag_mode=LCGP_DIAG_MODE,
        scale_multiplier=SCALE_MULTIPLIER,
        device=DEVICE,
        seed=SEED,
        precompute_knn=True,
        loadings_mode=LOADINGS_MODE,
    )


    freeze_svgp_kernel_and_inputs(model)

    # Prepare parameter groups with separate learning rates
    from gpzoo.training_utilities import prepare_frozen_params
    param_dict = prepare_frozen_params(model)

    # Build optimizer param groups with specific learning rates
    param_groups = []

    # Mean parameters (mu) - LR_MEAN
    if param_dict['mean_params']:
        param_groups.append({"params": param_dict['mean_params'], "lr": LR_MEAN})

    # Loading parameters (W) - LR_LOADING
    if param_dict['loading_params']:
        param_groups.append({"params": param_dict['loading_params'], "lr": LR_LOADING})

    # Other parameters - base LR
    if param_dict['other_params']:
        param_groups.append({"params": param_dict['other_params'], "lr": LR})

    # Scale parameters (Lu) - start frozen with lr=0
    if param_dict['scale_params']:
        param_groups.append({"params": param_dict['scale_params'], "lr": 0.0})

    # Use Adam optimizer
    optimizer = optim.Adam(param_groups)

    # Create SequentialLR scheduler: Linear warmup + Cosine decay (only if USE_SCHEDULER is True)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = create_sequential_lr_scheduler(
            optimizer=optimizer,
            total_steps=STEPS,
            base_lr=LR,
            warmup_fraction=0.2,
            start_factor=0.1,
            min_lr=3e-5,
        )

    save_path = checkpoint_path
    result = run_training(
        label=f"slideseq_lcgp_k={LCGP_K}",
        model=model,
        optimizer=optimizer,
        train_fn=train_lcgp_batched_with_tracking,
        save_path=save_path,
        X=X,
        y=Y,
        device=DEVICE,
        steps=STEPS,
        x_batch_size=X_BATCH,
        y_batch_size=Y_BATCH,
        lengthscale_unfreeze_step=LENGTHSCALE_TRAIN_AFTER,
        lengthscale_param=param_dict['lengthscale_param'],
        lengthscale_lr=LR_LENGTHSCALE,
        scale_unfreeze_step=SCALE_TRAIN_AFTER,
        scale_params=param_dict['scale_params'],
        scale_lr=LR_SCALE,
        writer=writer,
        progress_desc=f"LCGP (slideseq)",
        image_log_every=IMAGE_LOG_EVERY,
        start_step=start_step,
        scheduler=scheduler,
    )

    writer.close()

    print("Training finished. Artifacts written to:")
    for key, value in result.items():
        if key.endswith("path") or key.endswith("json"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    _train_main()
