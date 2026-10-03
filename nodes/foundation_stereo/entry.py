"""FoundationStereo node: rectified stereo pair -> disparity (px) -> metric depth."""
from __future__ import annotations

import sys
import time

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def estimate_depth(ctx: Context) -> None:
    import cv2
    import torch
    import torch.nn.functional as F

    if ctx.device != "cuda":
        raise NodeError("FoundationStereo needs a CUDA GPU")
    left = io.read_image(ctx.input("left"))
    right = io.read_image(ctx.input("right"))
    if left.shape != right.shape:
        raise NodeError(f"left {left.shape[:2]} and right {right.shape[:2]} "
                        "must have the same size (rectified pair)")
    H, W = left.shape[:2]
    cam = io.read_camera(ctx.input("camera"))
    if (int(cam["width"]), int(cam["height"])) != (W, H):
        raise NodeError(f"camera is {cam['width']}x{cam['height']} but images are {W}x{H}",
                        hint="pass the intrinsics of the rectified LEFT image at this resolution")
    if np.abs(np.asarray(cam["dist"])).max() > 1e-6:
        raise NodeError("camera has distortion; FoundationStereo needs rectified, "
                        "undistorted images", hint="rectify first and pass the "
                        "rectified K with zero distortion")
    baseline = ctx.param("baseline")
    if baseline is None or float(baseline) <= 0:
        raise NodeError("param baseline (metres, > 0) is required",
                        hint="distance between the two camera centres, e.g. 0.063")
    baseline = float(baseline)
    scale = float(ctx.param("scale", 1.0))
    if not 0 < scale <= 1:
        raise NodeError("scale must be in (0, 1]")
    iters = int(ctx.param("valid_iters", 32))
    hiera = bool(ctx.param("hiera", False))

    repo = str(ctx.repo)
    sys.path.insert(0, repo)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder

    torch.manual_seed(0)
    np.random.seed(0)
    torch.autograd.set_grad_enabled(False)
    wdir = ctx.weight("vitl")
    cfg = OmegaConf.load(str(wdir / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = iters
    cfg["hiera"] = int(hiera)
    cfg["scale"] = scale
    args = OmegaConf.create(cfg)
    t0 = time.time()
    model = FoundationStereo(args)
    ckpt = torch.load(str(ctx.weight("vitl", prefetch=True) / "model_best_bp2.pth"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    ctx.log(f"loaded FoundationStereo in {time.time() - t0:.1f}s")

    img0, img1 = left, right
    if scale < 1:
        img0 = cv2.resize(left, None, fx=scale, fy=scale)
        img1 = cv2.resize(right, None, fx=scale, fy=scale)
    h, w = img0.shape[:2]
    t0 = time.time()
    x0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    x1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(x0.shape, divis_by=32, force_square=False)
    x0, x1 = padder.pad(x0, x1)
    with torch.amp.autocast("cuda", enabled=True):
        if not hiera:
            disp = model.forward(x0, x1, iters=iters, test_mode=True)
        else:
            disp = model.run_hierachical(x0, x1, iters=iters, test_mode=True,
                                         small_ratio=0.5)
    disp = padder.unpad(disp.float())  # 1,1,h,w px at inference resolution
    if scale < 1:
        # back to the input resolution: resample and rescale the pixel units
        disp = F.interpolate(disp, size=(H, W), mode="bilinear",
                             align_corners=False) * (W / w)
    disp = disp[0, 0].cpu().numpy().astype(np.float32)
    infer_sec = time.time() - t0

    # pixels whose match falls left of the right image (upstream remove_invisible)
    xx = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    invisible = (xx - disp) < 0
    fx = float(cam["K"][0, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = fx * baseline / disp
    max_depth = float(ctx.param("max_depth", 100.0))
    bad = ~np.isfinite(depth) | (depth <= 0) | (depth > max_depth)
    if bool(ctx.param("remove_invisible", True)):
        bad |= invisible
    depth[bad] = 0
    if (depth > 0).mean() < 0.05:
        raise NodeError("almost no valid depth; check that the pair is rectified "
                        "and that left/right are not swapped")

    dpath = ctx.output_path("depth", "depth.npy")
    io.write_depth(dpath, depth)
    ppath = ctx.output_path("disparity_px", "disparity_px.npy")
    np.save(ppath, disp)
    ctx.set_output("depth", dpath)
    ctx.set_output("disparity_px", ppath)
    ctx.metadata.update({
        "fx": fx, "baseline_m": baseline, "inference_size": [w, h],
        "disparity_px_range": [float(np.nanmin(disp)), float(np.nanmax(disp))],
        "invisible_ratio": float(invisible.mean()),
        "valid_ratio": float((depth > 0).mean()), "infer_sec": round(infer_sec, 2)})


if __name__ == "__main__":
    raise SystemExit(main({"estimate_depth": estimate_depth}))
