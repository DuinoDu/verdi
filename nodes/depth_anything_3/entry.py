"""Depth-Anything-3 node: metric depth, relative depth, multi-view reconstruction."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

PATCH = 14
_SKY: List[np.ndarray] = []  # non-sky masks captured from the nested model
_ALIGN: List[float] = []     # Sim(3) scale of upstream input-pose alignment
# known intrinsics for the nested metric branch: {"K": [3x3 at input res per
# view of the current call], "wh": (w, h)}; empty = upstream (predicted focal)
_KNOWN_K: Dict = {}
_FOCAL: List[Dict] = []      # per call: predicted vs used focal (network px)

# weight key -> (HF repo, scale semantics of the depth / poses it returns)
MODELS = {
    "nested": ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", "metric"),
    "giant": ("depth-anything/DA3-GIANT-1.1", "relative"),
    "large": ("depth-anything/DA3-LARGE-1.1", "relative"),
    "metric": ("depth-anything/DA3METRIC-LARGE", "canonical"),
    "mono": ("depth-anything/DA3MONO-LARGE", "relative"),
}
CONVENTION = {
    "depth": "z-depth along the camera optical axis (not ray distance), "
             "input resolution, 0 = sky / invalid",
    "T_world_cam": "camera-to-world 4x4, OpenCV camera axes; upstream DA3 "
                   "returns world-to-camera 3x4 extrinsics, inverted here",
    "K": "pinhole intrinsics of the INPUT frame resolution (network-resolution "
         "K rescaled by input/network size per axis)",
    "frames": "output index i = i-th input frame in sorted order (names in "
              "`frames`)",
}


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

    if getattr(da3mod, "_verdi_hooked", False):
        return
    orig = da3mod.set_sky_regions_to_max_depth

    def wrapped(depth, depth_conf, non_sky_mask, max_depth=200.0):
        _SKY.append(non_sky_mask.detach().reshape(depth.shape[-3:]
                                                  if depth.dim() > 3
                                                  else depth.shape)
                    .cpu().numpy().astype(bool))
        return orig(depth, depth_conf, non_sky_mask, max_depth=max_depth)

    da3mod.set_sky_regions_to_max_depth = wrapped
    da3mod._verdi_hooked = True
    # record the scale of the input-pose alignment (pose conditioning)
    import depth_anything_3.api as api

    orig_align = api.align_poses_umeyama

    def align(*a, **kw):
        out = orig_align(*a, **kw)
        _ALIGN.append(float(out[2]))
        return out

    api.align_poses_umeyama = align

    # Nested metric scaling: metres = metric_out * mean(fx, fy) / 300 with the
    # PREDICTED focal (any-view camera head). With a known camera we use the
    # known focal instead (same formula, the upstream DA3METRIC conversion).
    import numpy as _np
    import torch as _torch

    orig_ms = da3mod.NestedDepthAnything3Net._apply_metric_scaling

    def metric_scaling(self, output, metric_output):
        pred = float(((output.intrinsics[..., 0, 0] + output.intrinsics[..., 1, 1]) / 2)
                     .mean())
        if not _KNOWN_K.get("K"):
            _FOCAL.append({"predicted_focal_net": pred, "used": "predicted"})
            return orig_ms(self, output, metric_output)
        Hn, Wn = metric_output.depth.shape[-2:]
        w, h = _KNOWN_K["wh"]
        Ks = []
        for K in _KNOWN_K["K"]:
            K = _np.asarray(K, dtype=_np.float64).copy()
            K[0] *= Wn / w
            K[1] *= Hn / h
            Ks.append(K)
        Kt = _torch.as_tensor(_np.stack(Ks)[None], dtype=metric_output.depth.dtype,
                              device=metric_output.depth.device)
        if Kt.shape[1] != metric_output.depth.shape[1]:
            raise NodeError("internal: known-K count does not match the views")
        _FOCAL.append({"predicted_focal_net": pred, "used": "known",
                       "known_focal_net": float(((Kt[..., 0, 0] + Kt[..., 1, 1]) / 2).mean())})
        metric_output.depth = da3mod.apply_metric_scaling(metric_output.depth, Kt)
        return output

    da3mod.NestedDepthAnything3Net._apply_metric_scaling = metric_scaling


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


def _infer(model, rgbs: List[np.ndarray], res: int, ref_view: str = "saddle_balanced",
           extrinsics=None, intrinsics=None, align_scale: bool = True,
           use_ray_pose: bool = False):
    """One upstream inference call (SimFoundry recipe: upper_bound_resize).

    extrinsics: world-to-camera 4x4 per image (DA3 convention), intrinsics:
    3x3 per image at the INPUT resolution (DA3 rescales them itself).
    """
    import torch

    _SKY.clear()
    _ALIGN.clear()
    _FOCAL.clear()
    kw = {}
    if extrinsics is not None:
        kw.update(extrinsics=np.asarray(extrinsics, dtype=np.float32),
                  intrinsics=np.asarray(intrinsics, dtype=np.float32),
                  align_to_input_ext_scale=bool(align_scale))
    pred = model.inference(image=list(rgbs), process_res=res,
                           process_res_method="upper_bound_resize",
                           ref_view_strategy=ref_view, export_dir=None,
                           use_ray_pose=bool(use_ray_pose), **kw)
    sky = None
    if _SKY:
        non_sky = _SKY[-1].reshape(pred.depth.shape)
        sky = ~non_sky
    elif getattr(pred, "sky", None) is not None:
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


# --------------------------------------------------------------- inputs
def _check_K(cam: Dict, w: int, h: int, where: str) -> np.ndarray:
    if (int(cam["width"]), int(cam["height"])) != (w, h):
        raise NodeError(f"{where} is {cam['width']}x{cam['height']} but frames are {w}x{h}",
                        hint="pass intrinsics of the frames at their resolution")
    if np.abs(np.asarray(cam.get("dist", [0]), dtype=float)).max() > 1e-6:
        raise NodeError(f"{where} has distortion; DA3 is a pinhole model",
                        hint="undistort / rectify the frames first and pass the "
                             "new pinhole K (dist = 0)")
    return np.asarray(cam["K"], dtype=np.float64)


def _input_Ks(ctx: Context, n: int, w: int, h: int):
    if ctx.has_input("camera") and ctx.has_input("cameras"):
        raise NodeError("pass camera (shared K) OR cameras (per frame), not both")
    if ctx.has_input("camera"):
        K = _check_K(io.read_camera(ctx.input("camera")), w, h, "camera")
        return [K] * n
    if ctx.has_input("cameras"):
        cams = io.read_json(ctx.input("cameras"))["cameras"]
        if len(cams) != n:
            raise NodeError(f"cameras has {len(cams)} entries for {n} frames")
        return [_check_K(c, w, h, f"cameras[{i}]") for i, c in enumerate(cams)]
    return None


def _read_traj(path, n: int, where: str) -> List[np.ndarray]:
    data = io.read_json(path)
    Ts = [np.asarray(T, dtype=np.float64) for T in data["T_world_cam"]]
    if len(Ts) != n:
        raise NodeError(f"{where} has {len(Ts)} poses for {n} frames",
                        hint="one T_world_cam per input frame, same order")
    for i, T in enumerate(Ts):
        R = T[:3, :3]
        if np.abs(R @ R.T - np.eye(3)).max() > 1e-3 or np.linalg.det(R) < 0:
            raise NodeError(f"{where}[{i}] rotation is not orthonormal")
    return Ts


def _sim3_eval(pred_c2w: List[np.ndarray], ref_c2w: List[np.ndarray],
               metric: bool) -> Dict:
    """Independent comparison of predicted vs reference camera poses."""
    P = np.stack([T[:3, 3] for T in pred_c2w])
    Q = np.stack([T[:3, 3] for T in ref_c2w])
    out: Dict = {"frames": len(P)}
    try:
        s, R, t = _umeyama(P, Q)
    except NodeError:
        return {**out, "note": "reference cameras do not move; translation "
                "alignment undefined"}
    ali = (s * (R @ P.T)).T + t
    err = np.linalg.norm(ali - Q, axis=1)
    rot = []
    for Tp, Tr in zip(pred_c2w, ref_c2w):
        dR = (R @ Tp[:3, :3]).T @ Tr[:3, :3]
        rot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    out.update({
        "sim3_scale_ref_per_pred": s,
        "ate_rmse_after_sim3": float(np.sqrt((err ** 2).mean())),
        "ate_max_after_sim3": float(err.max()),
        "rotation_err_deg_after_sim3": {"mean": float(np.mean(rot)),
                                        "max": float(np.max(rot))},
        "per_frame_translation_err": err.tolist(),
        "per_frame_rotation_err_deg": [float(r) for r in rot]})
    if metric:  # rigid (no scale) alignment: tests the metric scale too
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
        R2 = U @ D @ Vt
        e2 = np.linalg.norm((R2 @ (P - mp).T).T + mq - Q, axis=1)
        out["ate_rmse_after_se3"] = float(np.sqrt((e2 ** 2).mean()))
    # relative motion between consecutive frames (scale ratio per step)
    steps_p = np.linalg.norm(np.diff(P, axis=0), axis=1)
    steps_q = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    ok = steps_q > 1e-4
    if ok.any():
        out["step_length_ratio_pred_over_ref"] = {
            "median": float(np.median(steps_p[ok] / steps_q[ok])),
            "p10": float(np.percentile(steps_p[ok] / steps_q[ok], 10)),
            "p90": float(np.percentile(steps_p[ok] / steps_q[ok], 90))}
    return out


# --------------------------------------------------------------- tasks
def estimate_depth(ctx: Context) -> None:
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    res = _check_res(ctx.param("process_res", 504))
    key = str(ctx.param("model", "nested"))
    if key == "metric" and not ctx.has_input("camera"):
        raise NodeError("model=metric (DA3METRIC-LARGE) needs `camera`: its output "
                        "is canonical depth, metres = focal * output / 300",
                        hint="pass the pinhole K of the image, or use model=nested "
                             "(metric without K)")
    Kin = None
    if ctx.has_input("camera"):
        Kin = _check_K(io.read_camera(ctx.input("camera")), w, h, "camera")
    model = _load(ctx, key)
    _KNOWN_K.clear()
    if key == "nested" and Kin is not None:
        _KNOWN_K.update(K=[Kin], wh=(w, h))
    pred, sky = _infer(model, [rgb], res)
    _KNOWN_K.clear()
    d = pred.depth[0].astype(np.float32)
    Hp, Wp = d.shape
    focal = dict(_FOCAL[-1]) if _FOCAL else {}
    if key == "nested":
        if not int(pred.is_metric):
            raise NodeError("nested checkpoint returned non-metric depth")
        Kpred = _scale_K(pred.intrinsics[0], w, h, Wp, Hp)
        focal["predicted_fx_input_res"] = float(Kpred[0, 0])
        if Kin is None:
            K = Kpred
            conversion = "none: DA3NESTED output is already metres (metric branch " \
                         "scaled with the PREDICTED focal, aligned to the any-view depth); " \
                         "the metric scale is only as good as that focal (see info.focal)"
            k_source = "predicted by DA3NESTED (camera decoder)"
        else:
            K = Kin
            conversion = "DA3NESTED with the KNOWN focal: its metric branch is scaled " \
                         "with mean(fx, fy) of the input K at network resolution / 300 " \
                         "instead of the predicted focal, then aligned as upstream"
            k_source = "input camera (used for the metric scale)"
    else:
        Kn = _scale_K(Kin, Wp, Hp, w, h)        # K at network resolution
        focal = 0.5 * (Kn[0, 0] + Kn[1, 1])
        d = d * (focal / 300.0)
        K = Kin
        conversion = (f"metres = focal_px * output / 300 with focal = mean(fx, fy) "
                      f"of the INPUT K rescaled to the network resolution "
                      f"{Wp}x{Hp} = {focal:.3f} px (upstream FAQ; same as the "
                      f"nested model's internal apply_metric_scaling)")
        k_source = "input camera"
    depth = _resize(d, w, h)
    if sky is not None and bool(ctx.param("sky_invalid", True)):
        depth[_resize_mask(sky[0], w, h)] = 0
    out = ctx.output_path("depth", "depth.npy")
    io.write_depth(out, depth)
    cam = ctx.output_path("camera", "camera.json")
    io.write_camera(cam, K, w, h)
    ctx.set_output("depth", out)
    ctx.set_output("camera", cam)
    info = {"model": MODELS[key][0], "scale_status": "metric",
            "conversion": conversion, "intrinsics_source": k_source,
            "network_resolution": [Wp, Hp], "input_resolution": [w, h],
            "nested_metric_scale_factor": getattr(pred, "scale_factor", None),
            "focal": focal,
            "sky_ratio": float(sky.mean()) if sky is not None else None,
            "convention": {k: CONVENTION[k] for k in ("depth", "K")}}
    ipath = ctx.output_path("info", "info.json")
    io.write_json(ipath, info)
    ctx.set_output("info", ipath)
    ctx.metadata.update({k: info[k] for k in ("model", "scale_status",
                                              "network_resolution", "sky_ratio")})


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
        "scale_status": "relative",
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
    key = str(ctx.param("model", "nested"))
    if key not in ("nested", "giant", "large"):
        raise NodeError(f"model={key!r}: reconstruct takes nested | giant | large")
    res = _check_res(ctx.param("process_res", 448))
    chunk_size = int(ctx.param("chunk_size", 220))
    overlap = int(ctx.param("overlap", 40))
    ref_view = str(ctx.param("ref_view_strategy", "saddle_balanced"))
    use_ray = bool(ctx.param("use_ray_pose", False))
    align_scale = bool(ctx.param("align_to_input_ext_scale", True))

    Kin = _input_Ks(ctx, n, w, h)
    poses = _read_traj(ctx.input("poses"), n, "poses") if ctx.has_input("poses") else None
    if Kin is not None and poses is None and key != "nested":
        raise NodeError("DA3 uses known intrinsics only together with known poses "
                        "(camera token = f(extrinsics, intrinsics)); K alone would "
                        "be silently ignored by upstream for giant / large",
                        hint="also pass poses, use model=nested (K then sets the "
                             "metric scale), or drop camera/cameras")
    known_focal = key == "nested" and Kin is not None
    if poses is not None and Kin is None:
        raise NodeError("pose conditioning needs intrinsics too (camera or cameras)")
    conditioned = poses is not None
    if conditioned and ctx.has_input("reference_poses"):
        raise NodeError("reference_poses evaluates the VISUAL poses; it is "
                        "meaningless when the same run is conditioned on poses",
                        hint="run once without poses (+ reference_poses) to test "
                             "the poses, and separately with poses for geometry")
    if n > chunk_size:
        if overlap < 3 or overlap >= chunk_size:
            raise NodeError(f"overlap={overlap} must be in [3, chunk_size)",
                            hint="SimFoundry uses chunk_size=220, overlap=40")
        if conditioned and not align_scale:
            raise NodeError("align_to_input_ext_scale=false returns each chunk in "
                            "its own prediction frame; use one chunk "
                            "(chunk_size >= frames) or align_to_input_ext_scale=true")
    starts, sizes = _plan_chunks(n, chunk_size, overlap)
    model = _load(ctx, key)
    w2c_in = [np.linalg.inv(T) for T in poses] if conditioned else None

    c2w = [None] * n          # camera-to-world in the output world
    Ks = [None] * n           # intrinsics at network resolution
    depth = [None] * n        # network-resolution depth
    conf = [None] * n
    skym = [None] * n
    net_hw = None
    chunk_info = []
    sky_inv = bool(ctx.param("sky_invalid", True))
    for ci, (s0, cs) in enumerate(zip(starts, sizes)):
        t0 = time.time()
        sl = slice(s0, s0 + cs)
        _KNOWN_K.clear()
        if known_focal:
            _KNOWN_K.update(K=list(Kin[sl]), wh=(w, h))
        pred, sky = _infer(model, rgbs[sl], res, ref_view,
                           extrinsics=w2c_in[sl] if conditioned else None,
                           intrinsics=Kin[sl] if conditioned else None,
                           align_scale=align_scale, use_ray_pose=use_ray)
        if key == "nested" and not int(pred.is_metric):
            raise NodeError("nested checkpoint returned non-metric depth")
        net_hw = pred.depth.shape[1:]
        c2w_i = [np.linalg.inv(_w2c_44(E)) for E in pred.extrinsics]
        info: Dict = {"start": s0, "size": cs, "sec": round(time.time() - t0, 1),
                      "nested_metric_scale_factor": getattr(pred, "scale_factor", None)}
        _KNOWN_K.clear()
        if _FOCAL:
            info["focal"] = dict(_FOCAL[-1])
        if conditioned:
            info["input_pose_alignment_scale_pred_per_input"] = _ALIGN[-1] if _ALIGN else None
        if ci == 0 or conditioned:
            # conditioned chunks are already in the input-pose world
            sim = (1.0, np.eye(3), np.zeros(3))
            first = 0 if ci == 0 else (starts[ci - 1] + sizes[ci - 1] - s0)
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
                         "overlap_residual_mean": float(res_err.mean()),
                         "overlap_residual_max": float(res_err.max())})
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
            conf[g] = None if pred.conf is None else pred.conf[j].astype(np.float32)
            skym[g] = sky[j] if (sky is not None and sky_inv) else None
        chunk_info.append(info)
        ctx.log(f"chunk {ci}: frames {s0}..{s0 + cs - 1} {info}")
    missing = [i for i in range(n) if c2w[i] is None]
    if missing:
        raise NodeError(f"frames not reconstructed: {missing}")

    Hp, Wp = net_hw
    depth_dir = ctx.output_path("depth")
    conf_dir = ctx.output_path("confidence")
    for i in range(n):
        d = _resize(depth[i], w, h)
        if skym[i] is not None:
            d[_resize_mask(skym[i], w, h)] = 0
        io.write_depth(depth_dir / io.frame_name(i, ".npy"), d)
        if conf[i] is not None:
            np.save(conf_dir / io.frame_name(i, ".npy"), _resize(conf[i], w, h))
    Kout = [_scale_K(K, w, h, Wp, Hp) for K in Ks]
    if Kin is not None:  # known cameras: report them (depth was scaled with them)
        Kout = [np.asarray(K, dtype=np.float64) for K in Kin]
    cams = ctx.output_path("cameras", "cameras.json")
    io.write_json(cams, {"cameras": [
        {"K": K.tolist(), "width": w, "height": h, "dist": [0, 0, 0, 0, 0]}
        for K in Kout], "frames": [f.name for f in frames]})

    # ---- scale / source semantics (machine readable)
    metric_model = MODELS[key][1] == "metric"
    if not conditioned:
        scale_status = "metric" if metric_model else "relative"
        units = "metres (DA3NESTED metric branch)" if metric_model else \
            "arbitrary (DA3 any-view normalised scale); NOT metres"
        pose_source = "predicted by DA3 from the images (" + (
            "ray head, use_ray_pose" if use_ray else "camera decoder") + ")"
        k_source = "predicted by DA3"
        if known_focal:
            units = "metres (DA3NESTED metric branch scaled with the KNOWN focal)"
            k_source = "input camera(s): used for the metric scale only (poses are " \
                       "still predicted from the images; not pose conditioning)"
        world = "DA3 world of chunk 0 (first-chunk reference view), Sim(3)-merged chunks"
    elif align_scale:
        scale_status = "input_pose_scale"
        units = "units of the INPUT poses (metres only if the input poses are metric); " \
                "depth was divided by the pose-alignment scale"
        pose_source = "INPUT poses passed through unchanged (conditioning); NOT an " \
                      "independent estimate - do not use this run to validate them"
        k_source = "input intrinsics"
        world = "world frame of the input poses"
    else:
        scale_status = "metric" if metric_model else "relative"
        units = ("metres (DA3NESTED metric branch)" if metric_model else
                 "arbitrary DA3 scale") + "; input poses Sim(3)-mapped into the prediction frame"
        pose_source = "INPUT poses Sim(3)-aligned into the DA3 prediction frame " \
                      "(conditioning); NOT an independent estimate"
        k_source = "input intrinsics"
        world = "DA3 prediction world (not the input-pose world)"
    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {"T_world_cam": [T.tolist() for T in c2w],
                         "frame_index": list(range(n)),
                         "frames": [f.name for f in frames],
                         "units": units, "scale_status": scale_status,
                         "pose_source": pose_source, "world": world})
    ctx.set_output("depth", depth_dir)
    ctx.set_output("confidence", conf_dir)
    ctx.set_output("cameras", cams)
    ctx.set_output("trajectory", traj)

    pose_eval = None
    if ctx.has_input("reference_poses"):
        ref = _read_traj(ctx.input("reference_poses"), n, "reference_poses")
        pose_eval = _sim3_eval(c2w, ref, metric_model)
        pose_eval["note"] = ("DA3 poses were predicted WITHOUT the reference "
                             "poses (independent visual estimate)")
        ppath = ctx.output_path("pose_eval", "pose_eval.json")
        io.write_json(ppath, pose_eval)
        ctx.set_output("pose_eval", ppath)
    info = {"model": MODELS[key][0], "weights_key": key,
            "scale_status": scale_status, "units": units,
            "pose_source": pose_source, "intrinsics_source": k_source,
            "world": world, "conditioned_on_poses": conditioned,
            "metric_focal": ("known (input K)" if known_focal else
                             "predicted" if metric_model else None),
            "align_to_input_ext_scale": align_scale if conditioned else None,
            "use_ray_pose": use_ray, "ref_view_strategy": ref_view,
            "process_res": res, "network_resolution": [Wp, Hp],
            "input_resolution": [w, h], "chunks": chunk_info,
            "confidence": "DA3 depth confidence (network output, uncalibrated, "
                          "higher = more confident; compare within a run only)",
            "convention": CONVENTION, "frames": [f.name for f in frames],
            "path_length": float(sum(np.linalg.norm(c2w[i + 1][:3, 3] - c2w[i][:3, 3])
                                     for i in range(n - 1)))}
    ipath = ctx.output_path("info", "info.json")
    io.write_json(ipath, info)
    ctx.set_output("info", ipath)
    ctx.metadata.update({
        "model": info["model"], "scale_status": scale_status,
        "pose_source": pose_source, "network_resolution": [Wp, Hp],
        "chunks": chunk_info,
        "fx_per_frame": [round(float(K[0, 0]), 2) for K in Kout]})


if __name__ == "__main__":
    raise SystemExit(main({"estimate_depth": estimate_depth,
                           "estimate_relative_depth": estimate_relative_depth,
                           "reconstruct": reconstruct}))
