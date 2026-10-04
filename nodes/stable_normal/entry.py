"""StableNormal node: estimate_normal (OpenCV camera-frame normals)."""
from __future__ import annotations

import sys

import numpy as np

from verdi.sdk import Context, NodeError, main

# Upstream StableNormal output: x LEFT, y up, z towards the viewer
# (verified against normals derived from metric depth, see NOTES.md).
# OpenCV camera frame: x right, y down, z forward  ->  negate all axes.
TO_OPENCV = np.array([-1.0, -1.0, -1.0], np.float32)


def _load(ctx: Context, model: str):
    import torch

    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import hubconf

    weights = str(ctx.weights)
    device = "cuda:0" if ctx.device == "cuda" else "cpu"
    if device == "cpu":
        raise NodeError("StableNormal needs a GPU (fp16 pipelines)",
                        kind="request")
    torch.manual_seed(int(ctx.param("seed", 0)))
    if model == "turbo":
        return hubconf.StableNormal_turbo(local_cache_dir=weights,
                                          device=device,
                                          yoso_version="yoso-normal-v1-5")
    return hubconf.StableNormal(local_cache_dir=weights, device=device,
                                yoso_version="yoso-normal-v1-5",
                                diffusion_version="stable-normal-v0-1")


def estimate_normal(ctx: Context) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    model = ctx.param("model", "stable_normal")
    predictor = _load(ctx, model)
    import hubconf

    img = Image.open(ctx.input("image"))
    orig_w, orig_h = img.size
    alpha = None
    if img.mode == "RGBA":
        img, alpha = predictor._process_rgba_image(img)
    img = img.convert("RGB")
    proc = hubconf.resize_image(img, int(ctx.param("resolution", 1024)))
    gen = torch.Generator(device="cuda").manual_seed(
        int(ctx.param("seed", 0)))
    import inspect

    kwargs = {}
    if "generator" in inspect.signature(predictor.model.__call__).parameters:
        kwargs["generator"] = gen
    with torch.no_grad():
        out = predictor.model(proc, match_input_resolution=True, **kwargs)
    pred = np.asarray(out.prediction[0], np.float32)  # HxWx3 in [-1, 1]
    if pred.ndim != 3 or pred.shape[2] != 3 or not np.isfinite(pred).all():
        raise NodeError(f"unexpected StableNormal output {pred.shape}")
    t = torch.from_numpy(pred).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(orig_h, orig_w), mode="bilinear",
                      align_corners=False)
    normal = t[0].permute(1, 2, 0).numpy()
    normal = normal / np.clip(np.linalg.norm(normal, axis=-1,
                                             keepdims=True), 1e-6, None)
    normal = (normal * TO_OPENCV).astype(np.float32)
    if alpha is not None:
        normal[alpha == 0] = 0.0  # background: zero vector = no normal
    out_path = ctx.output_path("normal", "normal.npy")
    np.save(out_path, normal)
    ctx.set_output("normal", out_path)
    ctx.metadata.update({"model": model,
                         "processing_size": list(proc.size)})


if __name__ == "__main__":
    raise SystemExit(main({"estimate_normal": estimate_normal}))
