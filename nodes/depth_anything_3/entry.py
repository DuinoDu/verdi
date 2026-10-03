"""Depth-Anything-3 node: metric depth, relative depth, multi-view reconstruction."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

PATCH = 14
_SKY: List[np.ndarray] = []  # non-sky masks captured from the nested model


def _check_res(res: int) -> int:
    res = int(res)
    if res < PATCH * 8 or res % PATCH:
        raise NodeError(f"process_res={res} must be a multiple of 14 and >= 112",
                        hint="e.g. 448 (SimFoundry sequences) or 504 (default)")
    return res


def _hook_sky() -> None:
    """Record the sky mask the nested model fills with a fake max depth.

    NestedDepthAnything3Net._handle_sky_regions overwrites sky pixels with the
    99th-percentile non-sky depth (<= 200 m). That value is not a measurement;
    we capture the mask so the node can write 0 (= invalid) there instead.
    """
    import depth_anything_3.model.da3 as da3mod

    if getattr(da3mod, "_pdebug_hooked", False):
        return
    orig = da3mod.set_sky_regions_to_max_depth

    def wrapped(depth, depth_conf, non_sky_mask, max_depth=200.0):
        _SKY.append(non_sky_mask.detach().reshape(depth.shape[-3:]
                                                  if depth.dim() > 3
                                                  else depth.shape)
                    .cpu().numpy().astype(bool))
        return orig(depth, depth_conf, non_sky_mask, max_depth=max_depth)

    da3mod.set_sky_regions_to_max_depth = wrapped
    da3mod._pdebug_hooked = True


def _load(ctx: Context, key: str):
    import torch
    from depth_anything_3.api import DepthAnything3

    if ctx.device != "cuda":
        raise NodeError("Depth-Anything-3 needs a CUDA GPU",
                        hint="run with device=cuda")
    _hook_sky()
    t0 = time.time()
    model = DepthAnything3.from_pretrained(str(ctx.weight(key, prefetch=True)))
    model = model.to(torch.device("cuda")).eval()
    ctx.log(f"loaded {key} in {time.time() - t0:.1f}s")
    return model


def _infer(model, rgbs: List[np.ndarray], res: int, ref_view: str = "saddle_balanced"):
    """One upstream inference call (SimFoundry recipe: upper_bound_resize)."""
    import torch

    _SKY.clear()
    pred = model.inference(image=list(rgbs), process_res=res,
                           process_res_method="upper_bound_resize",
                           ref_view_strategy=ref_view, export_dir=None)
    sky = None
    if _SKY:
        non_sky = _SKY[-1].reshape(pred.depth.shape)
        sky = ~non_sky
    elif pred.sky is not None:
        sky = pred.sky.reshape(pred.depth.shape)
    torch.cuda.empty_cache()
    return pred, sky


def _resize(d: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    if d.shape == (h, w):
        return d.astype(np.float32)
    return cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_mask(m: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    if m.shape == (h, w):
        return m
    return cv2.resize(m.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _scale_K(K: np.ndarray, w: int, h: int, W: int, H: int) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= w / W
    K[1, :] *= h / H
    return K


# --------------------------------------------------------------- tasks
def estimate_depth(ctx: Context) -> None:
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    res = _check_res(ctx.param("process_res", 504))
    model = _load(ctx, "nested")
    pred, sky = _infer(model, [rgb], res)
    if not int(pred.is_metric):
        raise NodeError("nested checkpoint returned non-metric depth")
    d = pred.depth[0]
    Hp, Wp = d.shape
    depth = _resize(d, w, h)
    if sky is not None and bool(ctx.param("sky_invalid", True)):
        depth[_resize_mask(sky[0], w, h)] = 0
    out = ctx.output_path("depth", "depth.npy")
    io.write_depth(out, depth)
    K = _scale_K(pred.intrinsics[0], w, h, Wp, Hp)
    cam = ctx.output_path("camera", "camera.json")
    io.write_camera(cam, K, w, h)
    ctx.set_output("depth", out)
    ctx.set_output("camera", cam)
    ctx.metadata.update({
        "network_resolution": [Wp, Hp], "scale_factor": pred.scale_factor,
        "sky_ratio": float(sky.mean()) if sky is not None else None,
        "units": "metres (DA3NESTED-GIANT-LARGE-1.1 metric branch)"})


def estimate_relative_depth(ctx: Context) -> None:
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    res = _check_res(ctx.param("process_res", 504))
    model = _load(ctx, "mono")
    pred, sky = _infer(model, [rgb], res)
    d = pred.depth[0].astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if sky is not None:
        valid &= ~sky[0]
    if valid.mean() < 0.01:
        raise NodeError("DA3MONO returned no valid depth")
    disp = np.zeros_like(d)
    disp[valid] = 1.0 / d[valid]
    disp = _resize(disp, w, h)
    out = ctx.output_path("disparity", "disparity.npy")
    np.save(out, disp.astype(np.float32))
    ctx.set_output("disparity", out)
    ctx.metadata.update({
        "network_resolution": [d.shape[1], d.shape[0]],
        "sky_ratio": float(sky.mean()) if sky is not None else None,
        "units": "relative inverse depth (1 / DA3MONO depth), unknown scale; "
                 "0 = sky / invalid"})


def _plan_chunks(n: int, chunk_size: int, overlap: int):
    """SimFoundry depth_backends._plan_chunks."""
    if n <= chunk_size:
        return [0], [n]
    step = chunk_size - overlap
    starts = list(range(0, n - chunk_size + 1, step))
    if starts[-1] + chunk_size < n:
        starts.append(n - chunk_size)
    return starts, [chunk_size] * len(starts)


def _umeyama(P: np.ndarray, Q: np.ndarray):
    """Similarity (s, R, t) minimising |s R P + t - Q|^2."""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    var = (X ** 2).sum(1).mean()
    if var < 1e-12:
        raise NodeError("chunk overlap cameras do not move; cannot fit Sim(3)",
                        hint="raise chunk_size so the sequence runs in one "
                        "pass, or increase overlap")
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / var)
    t = mq - s * R @ mp
    return s, R, t


def _w2c_44(E: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = np.asarray(E, dtype=np.float64)[:3, :4]
    return T


def reconstruct(ctx: Context) -> None:
    frames = io.list_frames(ctx.input("frames"))
    n = len(frames)
    if n < 2:
        raise NodeError("reconstruct needs >= 2 frames",
                        hint="use estimate_depth for a single image")
    rgbs = [io.read_image(f) for f in frames]
    h, w = rgbs[0].shape[:2]
    if any(r.shape[:2] != (h, w) for r in rgbs):
        raise NodeError("all frames must have the same resolution")
    res = _check_res(ctx.param("process_res", 448))
    chunk_size = int(ctx.param("chunk_size", 220))
    overlap = int(ctx.param("overlap", 40))
    ref_view = str(ctx.param("ref_view_strategy", "saddle_balanced"))
    if n > chunk_size:
        if overlap < 3 or overlap >= chunk_size:
            raise NodeError(f"overlap={overlap} must be in [3, chunk_size)",
                            hint="SimFoundry uses chunk_size=220, overlap=40")
    starts, sizes = _plan_chunks(n, chunk_size, overlap)
    model = _load(ctx, "nested")

    c2w = [None] * n          # camera-to-world in the chunk-0 world
    Ks = [None] * n           # intrinsics at network resolution
    depth = [None] * n        # network-resolution metric depth
    skym = [None] * n
    net_hw = None
    chunk_info = []
    sky_inv = bool(ctx.param("sky_invalid", True))
    for ci, (s0, cs) in enumerate(zip(starts, sizes)):
        t0 = time.time()
        pred, sky = _infer(model, rgbs[s0:s0 + cs], res, ref_view)
        if not int(pred.is_metric):
            raise NodeError("nested checkpoint returned non-metric depth")
        net_hw = pred.depth.shape[1:]
        c2w_i = [np.linalg.inv(_w2c_44(E)) for E in pred.extrinsics]
        info: Dict = {"start": s0, "size": cs, "sec": round(time.time() - t0, 1),
                      "metric_scale_factor": pred.scale_factor}
        if ci == 0:
            sim = (1.0, np.eye(3), np.zeros(3))
            first = 0
        else:
            prev_end = starts[ci - 1] + sizes[ci - 1] - 1
            ov = min(prev_end, s0 + cs - 1) - s0 + 1
            if ov < 3:
                raise NodeError(f"chunk {ci} overlaps only {ov} frames (< 3)",
                                hint="increase overlap")
            P = np.stack([c2w_i[j][:3, 3] for j in range(ov)])
            Q = np.stack([c2w[s0 + j][:3, 3] for j in range(ov)])
            sim = _umeyama(P, Q)
            sc, R, t = sim
            res_err = np.linalg.norm((sc * (R @ P.T)).T + t - Q, axis=1)
            info.update({"overlap": ov, "sim3_scale": sc,
                         "overlap_residual_mean_m": float(res_err.mean()),
                         "overlap_residual_max_m": float(res_err.max())})
            first = ov
        sc, R, t = sim
        for j in range(first, cs):
            g = s0 + j
            T = np.eye(4)
            T[:3, :3] = R @ c2w_i[j][:3, :3]
            T[:3, 3] = sc * (R @ c2w_i[j][:3, 3]) + t
            c2w[g] = T
            Ks[g] = np.asarray(pred.intrinsics[j], dtype=np.float64)
            depth[g] = pred.depth[j].astype(np.float32) * sc
            skym[g] = sky[j] if (sky is not None and sky_inv) else None
        chunk_info.append(info)
        ctx.log(f"chunk {ci}: frames {s0}..{s0 + cs - 1} {info}")
    missing = [i for i in range(n) if c2w[i] is None]
    if missing:
        raise NodeError(f"frames not reconstructed: {missing}")

    Hp, Wp = net_hw
    depth_dir = ctx.output_path("depth")
    for i in range(n):
        d = _resize(depth[i], w, h)
        if skym[i] is not None:
            d[_resize_mask(skym[i], w, h)] = 0
        io.write_depth(depth_dir / io.frame_name(i, ".npy"), d)
    Kin = [_scale_K(K, w, h, Wp, Hp) for K in Ks]
    cams = ctx.output_path("cameras", "cameras.json")
    io.write_json(cams, {"cameras": [
        {"K": K.tolist(), "width": w, "height": h, "dist": [0, 0, 0, 0, 0]}
        for K in Kin], "frames": [f.name for f in frames]})
    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {"T_world_cam": [T.tolist() for T in c2w],
                         "frames": [f.name for f in frames],
                         "units": "metres"})
    ctx.set_output("depth", depth_dir)
    ctx.set_output("cameras", cams)
    ctx.set_output("trajectory", traj)
    ctx.metadata.update({
        "network_resolution": [Wp, Hp], "chunks": chunk_info,
        "path_length_m": float(sum(np.linalg.norm(c2w[i + 1][:3, 3] - c2w[i][:3, 3])
                                   for i in range(n - 1))),
        "fx_per_frame": [round(float(K[0, 0]), 2) for K in Kin],
        "units": "metres (DA3NESTED metric scaling)"})


if __name__ == "__main__":
    raise SystemExit(main({"estimate_depth": estimate_depth,
                           "estimate_relative_depth": estimate_relative_depth,
                           "reconstruct": reconstruct}))
