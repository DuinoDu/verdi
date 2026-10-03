"""Prior-Depth-Anything node: metric depth completion from an RGB image + depth prior."""
from __future__ import annotations

import sys
import time

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

MIN_DEPTH = 1e-4  # upstream SparseSampler.min_depth


def complete_depth(ctx: Context) -> None:
    import torch

    if ctx.device != "cuda":
        raise NodeError("Prior-Depth-Anything needs a CUDA GPU (torch_cluster knn)")
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    prior = np.load(ctx.input("prior")).astype(np.float32)
    if prior.ndim != 2:
        raise NodeError(f"prior must be HxW, got {prior.shape}")
    prior[~np.isfinite(prior) | (prior < MIN_DEPTH)] = 0
    geometric = None
    if ctx.has_input("geometric"):
        geometric = np.load(ctx.input("geometric")).astype(np.float32)
        if geometric.shape != (h, w):
            raise NodeError(f"geometric depth {geometric.shape} must match the "
                            f"image size {(h, w)}")
        geometric[~np.isfinite(geometric) | (geometric < 0)] = 0
    pattern = str(ctx.param("pattern", "") or "") or None
    lowres = prior.shape != (h, w)
    if lowres:
        if pattern:
            raise NodeError("pattern must be empty when the prior is lower "
                            "resolution than the image (upstream rule)")
        if prior.shape[0] > h or prior.shape[1] > w:
            raise NodeError(f"prior {prior.shape} is larger than the image {(h, w)}",
                            hint="resize the prior to the image size")
    K = int(ctx.param("K", 5))
    n_known = int((prior > 0).sum())
    if n_known < max(K, 1):
        raise NodeError(f"prior has only {n_known} valid pixels (need >= K={K})")

    sys.path.insert(0, str(ctx.repo))
    from prior_depth_anything import PriorDepthAnything
    from prior_depth_anything.utils import Arguments

    torch.manual_seed(int(ctx.param("seed", 0)))
    np.random.seed(int(ctx.param("seed", 0)))
    args = Arguments()
    args.K = K
    version = str(ctx.param("version", "1.0"))
    wdir = str(ctx.weight("priorda"))
    t0 = time.time()
    model = PriorDepthAnything(
        device="cuda:0", version=version, mde_dir=wdir, ckpt_dir=wdir,
        frozen_model_size="vitl", conditioned_model_size="vitb",
        coarse_only=False, args=args)
    ctx.log(f"loaded Prior-Depth-Anything v{version} in {time.time() - t0:.1f}s")
    t0 = time.time()
    with torch.no_grad():
        out = model.infer_one_sample(
            image=rgb, prior=prior, geometric=geometric, pattern=pattern,
            double_global=bool(ctx.param("double_global", False)),
            visualize=False)
    depth = out.detach().float().cpu().numpy().reshape(h, w)
    infer_sec = time.time() - t0
    keep = bool(ctx.param("keep_prior", True)) and not lowres and not pattern
    if keep:
        # SimFoundry 5_decompose_scene: np.where(prior == 0, output, prior)
        depth = np.where(prior > 0, prior, depth)
    if not np.isfinite(depth).any() or (depth > 0).mean() < 0.5:
        raise NodeError("completion produced no valid depth")
    dst = ctx.output_path("depth", "depth.npy")
    io.write_depth(dst, depth)
    ctx.set_output("depth", dst)
    ctx.metadata.update({
        "version": version, "K": K, "pattern": pattern,
        "prior_valid_ratio": float((prior > 0).mean()),
        "prior_shape": list(prior.shape), "geometric": geometric is not None,
        "kept_prior": keep, "infer_sec": round(infer_sec, 2)})


if __name__ == "__main__":
    raise SystemExit(main({"complete_depth": complete_depth}))
