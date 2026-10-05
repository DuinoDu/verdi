"""FoundationStereo node: rectified stereo pair -> disparity (px) -> metric z-depth.

Depth conversion (rectified pinhole pair, horizontal baseline):
    z = fx * baseline / (d + doffs)
with d = x_left - x_right in pixels at the resolution of `camera`, fx of the
rectified LEFT K at that resolution and doffs = cx_right - cx_left (0 for
CALIB_ZERO_DISPARITY rectification, e.g. stereo_rectify). When the network
runs on downscaled images (scale < 1) the disparity is resampled back to the
input size and multiplied by W / w (pixel units of the input image); K is
never rescaled because depth is computed at the input resolution. All of
this is recorded in the `info` output.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


# ------------------------------------------------------------------ model
def _load_model(ctx: Context, iters: int, hiera: bool, scale: float):
    import torch
    from omegaconf import OmegaConf

    if ctx.device != "cuda":
        raise NodeError("FoundationStereo needs a CUDA GPU")
    sys.path.insert(0, str(ctx.repo))
    from core.foundation_stereo import FoundationStereo

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
    return model


def _infer(model, left: np.ndarray, right: np.ndarray, scale: float,
           iters: int, hiera: bool) -> np.ndarray:
    """Disparity (px of the INPUT resolution), float32 HxW, unmasked."""
    import cv2
    import torch
    import torch.nn.functional as F
    from core.utils.utils import InputPadder

    H, W = left.shape[:2]
    img0, img1 = left, right
    if scale < 1:
        img0 = cv2.resize(left, None, fx=scale, fy=scale)
        img1 = cv2.resize(right, None, fx=scale, fy=scale)
    h, w = img0.shape[:2]
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
        disp = F.interpolate(disp, size=(H, W), mode="bilinear",
                             align_corners=False) * (W / w)
    return disp[0, 0].cpu().numpy().astype(np.float32)


def _lr_error(disp_l: np.ndarray, disp_r: np.ndarray) -> np.ndarray:
    """|d_L(x) - d_R(x - d_L(x))| per left pixel (inf where x - d_L < 0)."""
    H, W = disp_l.shape
    xx = np.arange(W, dtype=np.float32)[None].repeat(H, 0)
    xr = xx - disp_l
    x0 = np.floor(xr).astype(np.int64)
    a = xr - x0
    inside = (x0 >= 0) & (x0 + 1 <= W - 1)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x0 + 1, 0, W - 1)
    rows = np.arange(H)[:, None].repeat(W, 1)
    dr = (1 - a) * disp_r[rows, x0c] + a * disp_r[rows, x1c]
    err = np.abs(disp_l - dr)
    err[~inside] = np.inf
    return err.astype(np.float32)


# ---------------------------------------------------------------- params
def _common(ctx: Context) -> Dict:
    baseline = ctx.param("baseline")
    if baseline is None or float(baseline) <= 0:
        raise NodeError("param baseline (metres, > 0) is required",
                        hint="distance between the two rectified camera centres, "
                             "e.g. 0.0605 (stereo_rectify: rectification.baseline_rect_m)")
    scale = float(ctx.param("scale", 1.0))
    if not 0 < scale <= 1:
        raise NodeError("scale must be in (0, 1]")
    return {"baseline": float(baseline), "scale": scale,
            "iters": int(ctx.param("valid_iters", 32)),
            "hiera": bool(ctx.param("hiera", False)),
            "doffs": float(ctx.param("doffs", 0.0)),
            "max_depth": float(ctx.param("max_depth", 100.0)),
            "remove_invisible": bool(ctx.param("remove_invisible", True)),
            "lr_check": bool(ctx.param("lr_check", False)),
            "lr_threshold": float(ctx.param("lr_threshold_px", 1.0))}


def _camera(ctx: Context, W: int, H: int) -> Dict:
    cam = io.read_camera(ctx.input("camera"))
    if (int(cam["width"]), int(cam["height"])) != (W, H):
        raise NodeError(f"camera is {cam['width']}x{cam['height']} but images are {W}x{H}",
                        hint="pass the intrinsics of the rectified LEFT image at this resolution")
    if np.abs(np.asarray(cam["dist"])).max() > 1e-6:
        raise NodeError("camera has distortion; FoundationStereo needs rectified, "
                        "undistorted images", hint="rectify first (verdi run "
                        "stereo_rectify) and pass its rectified K with zero distortion")
    K = cam["K"]
    if abs(K[0, 1]) > 1e-9:
        raise NodeError("camera K has skew; expected a rectified pinhole K")
    return cam


def _border(ctx: Context, W: int, H: int) -> Optional[np.ndarray]:
    if not ctx.has_input("valid"):
        return None
    from PIL import Image

    m = np.asarray(Image.open(ctx.input("valid"))) != 0
    if m.shape != (H, W):
        raise NodeError(f"valid mask is {m.shape[1]}x{m.shape[0]}, images are {W}x{H}")
    return m


def _pair(left, right):
    if left.shape != right.shape:
        raise NodeError(f"left {left.shape[:2]} and right {right.shape[:2]} "
                        "must have the same size (rectified pair)")


def _depth(disp, disp_r, cam, p, border):
    """-> depth (0 invalid), valid bool, reasons dict, lr_err or None."""
    H, W = disp.shape
    fx = float(cam["K"][0, 0])
    xx = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
    reasons = {}
    d = disp + p["doffs"]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (fx * p["baseline"] / d).astype(np.float32)
    bad = ~np.isfinite(depth) | (depth <= 0)
    reasons["nonpositive_disparity"] = float(bad.mean())
    far = depth > p["max_depth"]
    reasons["beyond_max_depth"] = float((far & ~bad).mean())
    bad |= far
    if p["remove_invisible"]:
        inv = (xx - disp) < 0
        reasons["match_outside_right_image"] = float((inv & ~bad).mean())
        bad |= inv
    if border is not None:
        reasons["rectification_border"] = float((~border & ~bad).mean())
        bad |= ~border
    lr = None
    if disp_r is not None:
        lr = _lr_error(disp, disp_r)
        fail = lr > p["lr_threshold"]
        reasons["left_right_inconsistent"] = float((fail & ~bad).mean())
        bad |= fail
    depth[bad] = 0
    return depth, ~bad, reasons, lr


def _info(cam, p, W, H, extra) -> Dict:
    s = p["scale"]
    w, h = (round(W * s), round(H * s)) if s < 1 else (W, H)
    K = np.asarray(cam["K"], dtype=float)
    Ki = K.copy()
    Ki[0] *= w / W
    Ki[1] *= h / H
    return {
        "depth_type": "z-depth along the optical axis of the rectified LEFT camera (OpenCV), metres; 0 = invalid",
        "formula": "z = fx * baseline_m / (disparity_px + doffs_px)",
        "fx": float(K[0, 0]), "baseline_m": p["baseline"], "doffs_px": p["doffs"],
        "camera_K": K.tolist(), "input_size": [W, H],
        "inference_size": [w, h], "scale": s,
        "K_at_inference_size": Ki.tolist(),
        "disparity_resample": "none" if s >= 1 else
            f"bilinear {w}x{h} -> {W}x{H}, values x {W / w:.6f}",
        "disparity_px": "x_left - x_right at the INPUT resolution, raw network output (no masking)",
        "valid_iters": p["iters"], "hiera": p["hiera"],
        "max_depth_m": p["max_depth"], "remove_invisible": p["remove_invisible"],
        "lr_check": p["lr_check"], "lr_threshold_px": p["lr_threshold"],
        "confidence": "FoundationStereo has no calibrated confidence output; "
                      "use `valid` (geometric checks) and the optional "
                      "left-right consistency error (lr_check) instead",
        **extra,
    }


def _run_pair(model, L, R, p):
    disp = _infer(model, L, R, p["scale"], p["iters"], p["hiera"])
    disp_r = None
    if p["lr_check"]:
        # right-view disparity from the mirrored pair: d_R(x) = d'(W-1-x)
        dprime = _infer(model, np.ascontiguousarray(R[:, ::-1]),
                        np.ascontiguousarray(L[:, ::-1]), p["scale"],
                        p["iters"], p["hiera"])
        disp_r = np.ascontiguousarray(dprime[:, ::-1])
    return disp, disp_r


# ------------------------------------------------------------------ tasks
def estimate_depth(ctx: Context) -> None:
    left = io.read_image(ctx.input("left"))
    right = io.read_image(ctx.input("right"))
    _pair(left, right)
    H, W = left.shape[:2]
    cam = _camera(ctx, W, H)
    p = _common(ctx)
    border = _border(ctx, W, H)
    model = _load_model(ctx, p["iters"], p["hiera"], p["scale"])
    t0 = time.time()
    disp, disp_r = _run_pair(model, left, right, p)
    infer_sec = time.time() - t0
    depth, valid, reasons, lr = _depth(disp, disp_r, cam, p, border)
    if valid.mean() < 0.05:
        raise NodeError("almost no valid depth; check that the pair is rectified "
                        "and that left/right are not swapped")
    dpath = ctx.output_path("depth", "depth.npy")
    io.write_depth(dpath, depth)
    ctx.set_output("depth", dpath)
    ppath = ctx.output_path("disparity_px", "disparity_px.npy")
    np.save(ppath, disp)
    ctx.set_output("disparity_px", ppath)
    vpath = ctx.output_path("valid", "valid.png")
    io.write_mask(vpath, valid.astype(np.uint16))
    ctx.set_output("valid", vpath)
    if lr is not None:
        lpath = ctx.output_path("lr_error_px", "lr_error_px.npy")
        np.save(lpath, lr)
        ctx.set_output("lr_error_px", lpath)
    stats = {"valid_ratio": float(valid.mean()), "invalid_reasons": reasons,
             "disparity_px_range": [float(np.nanmin(disp)), float(np.nanmax(disp))],
             "depth_m_percentiles_5_50_95": [float(v) for v in np.percentile(
                 depth[valid], [5, 50, 95])] if valid.any() else None,
             "infer_sec": round(infer_sec, 2)}
    ipath = ctx.output_path("info", "info.json")
    io.write_json(ipath, _info(cam, p, W, H, stats))
    ctx.set_output("info", ipath)
    ctx.metadata.update({"fx": float(cam["K"][0, 0]), "baseline_m": p["baseline"],
                         **stats})


def estimate_depth_seq(ctx: Context) -> None:
    fl = io.list_frames(ctx.input("left"))
    fr = io.list_frames(ctx.input("right"))
    if not fl or len(fl) != len(fr):
        raise NodeError(f"left has {len(fl)} frames, right has {len(fr)}; "
                        "need the same non-zero count (matched by sorted order)")
    L0 = io.read_image(fl[0])
    H, W = L0.shape[:2]
    cam = _camera(ctx, W, H)
    p = _common(ctx)
    border = _border(ctx, W, H)
    model = _load_model(ctx, p["iters"], p["hiera"], p["scale"])
    d_dir = ctx.output_path("depth")
    p_dir = ctx.output_path("disparity_px")
    v_dir = ctx.output_path("valid")
    l_dir = ctx.output_path("lr_error_px") if p["lr_check"] else None
    per_frame = []
    t0 = time.time()
    for i, (a, b) in enumerate(zip(fl, fr)):
        L, R = io.read_image(a), io.read_image(b)
        _pair(L, R)
        if L.shape[:2] != (H, W):
            raise NodeError(f"frame {a.name} is {L.shape[1]}x{L.shape[0]}, "
                            f"first frame is {W}x{H}")
        disp, disp_r = _run_pair(model, L, R, p)
        depth, valid, reasons, lr = _depth(disp, disp_r, cam, p, border)
        np.save(d_dir / f"{i:06d}.npy", depth)
        np.save(p_dir / f"{i:06d}.npy", disp)
        io.write_mask(v_dir / io.frame_name(i), valid.astype(np.uint16))
        if lr is not None:
            np.save(l_dir / f"{i:06d}.npy", lr)
        per_frame.append({"index": i, "left": a.name, "right": b.name,
                          "valid_ratio": float(valid.mean()),
                          "invalid_reasons": reasons,
                          "median_depth_m": float(np.median(depth[valid]))
                          if valid.any() else None})
        ctx.log(f"[fs] frame {i + 1}/{len(fl)} valid {valid.mean():.3f}")
    infer_sec = time.time() - t0
    ctx.set_output("depth", d_dir)
    ctx.set_output("disparity_px", p_dir)
    ctx.set_output("valid", v_dir)
    if l_dir is not None:
        ctx.set_output("lr_error_px", l_dir)
    ipath = ctx.output_path("info", "info.json")
    io.write_json(ipath, _info(cam, p, W, H, {
        "frames": per_frame, "infer_sec": round(infer_sec, 2),
        "sec_per_frame": round(infer_sec / len(fl), 3)}))
    ctx.set_output("info", ipath)
    ctx.metadata.update({"frames": len(fl), "sec_per_frame": round(infer_sec / len(fl), 3),
                         "fx": float(cam["K"][0, 0]), "baseline_m": p["baseline"]})


if __name__ == "__main__":
    raise SystemExit(main({"estimate_depth": estimate_depth,
                           "estimate_depth_seq": estimate_depth_seq}))
