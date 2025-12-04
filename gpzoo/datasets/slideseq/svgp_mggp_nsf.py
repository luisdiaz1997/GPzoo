from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import optim

from gpzoo.kernels import batched_MGGP_RBF
from gpzoo.model_utilities import build_mggp_nsf_svgp, init_mggp_inducing_points
from torch.utils.tensorboard import SummaryWriter


def _infer_groups(groupsX: torch.Tensor) -> int:
    if groupsX.numel() == 0:
        raise ValueError("groupsX must contain at least one value.")
    return int(groupsX.detach().max().item()) + 1


def create_model(
    *,
    X: torch.Tensor,
    Y: torch.Tensor,
    groupsX: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    L: int = 12,
    lengthscale: float = 20.0,
    kernel_kwargs: Optional[Mapping[str, float]] = None,
    jitter: float = 1e-5,
    num_inducing: Optional[int] = None,
    inducing_points: Optional[torch.Tensor] = None,
    inducing_groups: Optional[torch.Tensor] = None,
    inducing_init_method: str = "kmeans",
    inducing_allocation: str = "equal",
    inducing_seed: int = 123,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
):
    """Build or resume an MGGP SVGP NSF model for Slideseq."""

    if groupsX is None:
        raise ValueError("groupsX is required for MGGP models.")

    n_groups = _infer_groups(groupsX)
    kernel_cfg = {"sigma": 1.0, "lengthscale": lengthscale, "group_diff_param": 10.0}
    if kernel_kwargs:
        kernel_cfg.update(dict(kernel_kwargs))
    kernel_cfg["lengthscale"] = lengthscale
    kernel_cfg.setdefault("n_groups", n_groups)
    kernel = batched_MGGP_RBF(**kernel_cfg)

    if state_dict is not None:
        if inducing_points is None:
            try:
                inducing_points = state_dict["prior.Z"].detach().clone()
            except KeyError as exc:
                raise KeyError("Checkpoint is missing prior.Z needed for SVGP resume.") from exc
        if inducing_groups is None:
            try:
                inducing_groups = state_dict["prior.groupsZ"].detach().clone()
            except KeyError as exc:
                raise KeyError("Checkpoint missing prior.groupsZ for MGGP SVGP.") from exc

    if inducing_points is None or inducing_groups is None:
        if num_inducing is None:
            num_inducing = min(4000, X.shape[0])
        inducing_points, inducing_groups = init_mggp_inducing_points(
            X,
            groupsX,
            num_inducing,
            method=inducing_init_method,
            seed=inducing_seed,
            allocation=inducing_allocation,
        )

    return build_mggp_nsf_svgp(
        X=X,
        groupsX=groupsX,
        Y=Y,
        V=V,
        L=L,
        kernel=kernel,
        jitter=jitter,
        inducing_points=inducing_points,
        inducing_groups=inducing_groups,
        device=device,
        seed=seed,
    )


def _train_main() -> None:
    from gpzoo.models import build_model
    from gpzoo.training_utilities import freeze_mggp_kernel_and_inputs, train_mggp_svgp_with_tracking
    from gpzoo.datasets.slideseq.common import (
        DEVICE,
        OUTPUT_DIR,
        load_slideseq_with_groups,
        run_training,
    )
    from gpzoo.datasets.slideseq.config import (
        SVGP_MGGP_CHECKPOINT,
        GROUP_DIFF_PARAM,
        IMAGE_LOG_EVERY,
        INDUCING_ALLOCATION,
        JITTER,
        L_FACTORS,
        LENGTHSCALE,
        LENGTHSCALE_TRAIN_AFTER,
        LR,
        LR_LENGTHSCALE,
        LR_SCALE,
        SEED,
        STEPS,
        SVGP_INDUCING,
        X_BATCH,
        Y_BATCH,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, Y, groupsX, _, V_init = load_slideseq_with_groups()
    num_inducing = min(SVGP_INDUCING, X.shape[0])

    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tb" / "slideseq_mggp_svgp")

    checkpoint_path = OUTPUT_DIR / "slideseq_mggp_svgp.pth"
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    start_step = 0

    model, _ = build_model(
        "slideseq/svgp_mggp_nsf",
        checkpoint_path=checkpoint_arg,
        X=X,
        groupsX=groupsX,
        Y=Y,
        V=V_init,
        L=L_FACTORS,
        lengthscale=LENGTHSCALE,
        kernel_kwargs={
            "sigma": 1.0,
            "group_diff_param": GROUP_DIFF_PARAM,
        },
        jitter=JITTER,
        num_inducing=num_inducing,
        inducing_allocation=INDUCING_ALLOCATION,
        inducing_seed=SEED,
        device=DEVICE,
        seed=SEED,
    )


    freeze_mggp_kernel_and_inputs(model)
    lengthscale_param = getattr(model.prior.kernel, "lengthscale", None)
    if lengthscale_param is not None:
        lengthscale_param.requires_grad = False

    # Build param groups with separate LR for scale (Lu)
    scale_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "Lu" in name:
            scale_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": LR}]
    if scale_params:
        param_groups.append({"params": scale_params, "lr": LR_SCALE})

    optimizer = optim.Adam(param_groups)

    save_path = checkpoint_path
    result = run_training(
        label="slideseq_mggp_svgp",
        model=model,
        optimizer=optimizer,
        train_fn=train_mggp_svgp_with_tracking,
        save_path=save_path,
        X=X,
        groupsX=groupsX,
        y=Y,
        device=DEVICE,
        steps=STEPS,
        x_batch_size=X_BATCH,
        y_batch_size=Y_BATCH,
        lengthscale_unfreeze_step=LENGTHSCALE_TRAIN_AFTER,
        lengthscale_param=lengthscale_param,
        lengthscale_lr=LR_LENGTHSCALE,
        writer=writer,
        progress_desc="MGGP-SVGP (slideseq)",
        image_log_every=IMAGE_LOG_EVERY,
        start_step=start_step,
    )

    writer.close()

    print("Training finished. Artifacts written to:")
    for key, value in result.items():
        if key.endswith("path") or key.endswith("json"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    _train_main()
