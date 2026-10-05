"""MapAnything node: images (+ optional K / poses / depth) -> metric multi-view
geometry (z-depth, T_world_cam, K, confidence, fused point cloud).

Conventions (upstream -> verdi):
* MapAnything camera_poses are OpenCV cam2world in the world of view 0 (or
  of the input poses when given) == verdi T_world_cam; no conversion.
* The network runs on a resized + centre-cropped copy of every frame
  (`resolution_set` aspect-ratio table, e.g. 518x294 for 16:9). The exact
  input -> network pixel map u_net = sx * u_in + ox, v_net = sy * v_in + oy
  is computed with upstream's own crop code and recorded in info.crop;
  outputs are resampled back to the INPUT resolution (pixels outside the
  crop are 0 = invalid) unless output_resolution = "network".
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

MODELS = {"apache": "facebook/map-anything-apache",
          "nc": "facebook/map-anything"}
LAZY = {"nc": "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"}  # = manifest revision
DINOV2_DIR = "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"


# ------------------------------------------------------------------ setup
def _offline_hub(ctx: Context) -> None:
    """uniception loads the DINOv2 code (and ImageNet weights, which the
    MapAnything checkpoint overwrites) with torch.hub from GitHub. Use the
    pinned local dinov2 copy and skip the throw-away weight download."""
    import torch

    repo = ctx.weight("dinov2_repo") / DINOV2_DIR
    if not (repo / "hubconf.py").exists():
        raise NodeError(f"pinned dinov2 repo missing at {repo}", kind="setup",
                        hint="verdi setup mapanything")
    orig = torch.hub.load
    if getattr(torch.hub, "_verdi_patched", False):
        return

    def load(repo_or_dir, model, *args, **kw):
        if repo_or_dir == "facebookresearch/dinov2":
            kw.pop("force_reload", None)
            kw.pop("source", None)
            kw["pretrained"] = False
            return orig(str(repo), model, *args, source="local", **kw)
        return orig(repo_or_dir, model, *args, **kw)

    torch.hub.load = load
    torch.hub._verdi_patched = True


def _load(ctx: Context, key: str):
    import torch

    if ctx.device != "cuda":
        raise NodeError("MapAnything needs a CUDA GPU")
    _offline_hub(ctx)
    from mapanything.models import MapAnything

    t0 = time.time()
    if key in LAZY:  # optional checkpoint, fetched into HF_HOME on first use
        model = MapAnything.from_pretrained(MODELS[key], revision=LAZY[key])
    else:
        model = MapAnything.from_pretrained(str(ctx.weight(key, prefetch=True)))
    model = model.to("cuda").eval()
    ctx.log(f"[mapanything] loaded {MODELS[key]} in {time.time() - t0:.1f}s")
    return model


# ----------------------------------------------------------------- inputs
def _check_K(cam: Dict, w: int, h: int, where: str) -> np.ndarray:
    if (int(cam["width"]), int(cam["height"])) != (w, h):
        raise NodeError(f"{where} is {cam['width']}x{cam['height']} but frames are {w}x{h}")
    if np.abs(np.asarray(cam.get("dist", [0]), dtype=float)).max() > 1e-6:
        raise NodeError(f"{where} has distortion; MapAnything takes pinhole K",
                        hint="undistort / rectify first (stereo_rectify for the "
                             "fisheye rig) and pass the new K with dist = 0")
    return np.asarray(cam["K"], dtype=np.float64)


def _Ks(ctx: Context, n, w, h) -> Optional[List[np.ndarray]]:
    if ctx.has_input("camera") and ctx.has_input("cameras"):
        raise NodeError("pass camera (shared) OR cameras (per frame)")
    if ctx.has_input("camera"):
        return [_check_K(io.read_camera(ctx.input("camera")), w, h, "camera")] * n
    if ctx.has_input("cameras"):
        cams = io.read_json(ctx.input("cameras"))["cameras"]
        if len(cams) != n:
            raise NodeError(f"cameras has {len(cams)} entries for {n} frames")
        return [_check_K(c, w, h, f"cameras[{i}]") for i, c in enumerate(cams)]
    return None


def _traj(path, n, where) -> List[np.ndarray]:
    Ts = [np.asarray(T, dtype=np.float64) for T in io.read_json(path)["T_world_cam"]]
    if len(Ts) != n:
        raise NodeError(f"{where} has {len(Ts)} poses for {n} frames")
    return Ts


def _depths(ctx: Context, n, w, h) -> Optional[List[np.ndarray]]:
    if not ctx.has_input("depths"):
        return None
    files = io.list_frames(ctx.input("depths"), (".npy",))
    if len(files) != n:
        raise NodeError(f"depths has {len(files)} maps for {n} frames")
    out = []
    for f in files:
        d = np.load(f).astype(np.float32)
        if d.shape != (h, w):
            raise NodeError(f"depth {f.name} is {d.shape}, frames are {(h, w)}")
        d[~np.isfinite(d) | (d < 0)] = 0
        out.append(d)
    return out


# ------------------------------------------------------------ crop mapping
def _crop_map(w: int, h: int, target) -> Dict:
    """Affine input -> network pixel map of upstream crop_resize (no-K path)."""
    from PIL import Image
    from mapanything.utils.cropping import rescale_image_and_other_optional_info

    K0 = np.array([[1000.0, 0, w / 2], [0, 1000.0, h / 2], [0, 0, 1]])
    img = Image.new("RGB", (w, h))
    im2, _, K1, _ = rescale_image_and_other_optional_info(
        image=img, output_resolution=np.array(target), camera_intrinsics=K0.copy())
    ws, hs = im2.size
    left, top = (ws - target[0]) // 2, (hs - target[1]) // 2
    K1 = np.asarray(K1, dtype=np.float64).copy()
    K1[0, 2] -= left
    K1[1, 2] -= top
    return _affine(K0, K1)


def _affine(Kin: np.ndarray, Knet: np.ndarray) -> Dict:
    sx, sy = Knet[0, 0] / Kin[0, 0], Knet[1, 1] / Kin[1, 1]
    return {"sx": float(sx), "sy": float(sy),
            "ox": float(Knet[0, 2] - sx * Kin[0, 2]),
            "oy": float(Knet[1, 2] - sy * Kin[1, 2])}


def _to_input(arr: np.ndarray, m: Dict, w: int, h: int, nearest=False):
    """Resample a network-resolution map to the input grid (0 outside)."""
    import cv2

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = (m["sx"] * uu + m["ox"]).astype(np.float32)
    my = (m["sy"] * vv + m["oy"]).astype(np.float32)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(arr.astype(np.float32), mx, my, interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _K_to_input(Knet: np.ndarray, m: Dict) -> np.ndarray:
    A = np.array([[m["sx"], 0, m["ox"]], [0, m["sy"], m["oy"]], [0, 0, 1]])
    return np.linalg.inv(A) @ Knet


# ---------------------------------------------------------------- pose eval
def _umeyama(P, Q, with_scale=True):
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    var = (X ** 2).sum(1).mean()
    if var < 1e-12:
        return None
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / var) if with_scale else 1.0
    return s, R, mq - s * R @ mp


def _pose_eval(pred, ref) -> Dict:
    P = np.stack([T[:3, 3] for T in pred])
    Q = np.stack([T[:3, 3] for T in ref])
    out = {"frames": len(P), "note": "MapAnything poses predicted WITHOUT the "
           "reference poses (independent visual estimate)"}
    for name, ws in (("sim3", True), ("se3", False)):
        fit = _umeyama(P, Q, ws)
        if fit is None:
            out["note"] += "; reference cameras do not move"
            return out
        s, R, t = fit
        err = np.linalg.norm((s * (R @ P.T)).T + t - Q, axis=1)
        out[f"ate_rmse_after_{name}"] = float(np.sqrt((err ** 2).mean()))
        if ws:
            out["sim3_scale_ref_per_pred"] = s
            rot = [np.degrees(np.arccos(np.clip(
                (np.trace((R @ a[:3, :3]).T @ b[:3, :3]) - 1) / 2, -1, 1)))
                for a, b in zip(pred, ref)]
            out["rotation_err_deg_after_sim3"] = {"mean": float(np.mean(rot)),
                                                  "max": float(np.max(rot))}
            out["per_frame_translation_err"] = err.tolist()
    return out


# ------------------------------------------------------------------- task
def reconstruct(ctx: Context) -> None:
    import torch
    from PIL import Image

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    frames = io.list_frames(ctx.input("frames"))
    n = len(frames)
    if n < 1:
        raise NodeError("no frames found")
    rgbs = [io.read_image(f) for f in frames]
    h, w = rgbs[0].shape[:2]
    if any(r.shape[:2] != (h, w) for r in rgbs):
        raise NodeError("all frames must have the same resolution")
    key = str(ctx.param("model", "apache"))
    Ks = _Ks(ctx, n, w, h)
    poses = _traj(ctx.input("poses"), n, "poses") if ctx.has_input("poses") else None
    depths = _depths(ctx, n, w, h)
    if depths is not None and Ks is None:
        raise NodeError("depth inputs need intrinsics (camera or cameras)",
                        hint="MapAnything unprojects input depth with the given K")
    if poses is not None and ctx.has_input("reference_poses"):
        raise NodeError("reference_poses evaluates VISUAL poses; it is "
                        "meaningless when the same run is conditioned on poses")
    pose_metric = bool(ctx.param("poses_metric", True))
    depth_metric = bool(ctx.param("depths_metric", True))
    res_set = int(ctx.param("resolution_set", 518))
    if res_set not in (518, 512):
        raise NodeError("resolution_set must be 518 or 512")

    from mapanything.utils.image import preprocess_inputs

    views = []
    for i in range(n):
        v = {"img": torch.from_numpy(rgbs[i])}
        if Ks is not None:
            v["intrinsics"] = torch.from_numpy(Ks[i].astype(np.float32))
        if depths is not None:
            v["depth_z"] = torch.from_numpy(depths[i])
        if poses is not None:
            v["camera_poses"] = torch.from_numpy(poses[i].astype(np.float32))
        if poses is not None or depths is not None:
            v["is_metric_scale"] = torch.tensor(
                [pose_metric if poses is not None else depth_metric])
        views.append(v)
    t0 = time.time()
    pviews = preprocess_inputs(views, resize_mode="fixed_mapping",
                               resolution_set=res_set)
    Hn, Wn = pviews[0]["img"].shape[-2:]
    if Ks is not None:
        crops = [_affine(Ks[i], pviews[i]["intrinsics"][0].numpy().astype(np.float64))
                 for i in range(n)]
    else:
        crops = [_crop_map(w, h, (Wn, Hn))] * n
    model = _load(ctx, key)
    with torch.no_grad():
        preds = model.infer(
            pviews, memory_efficient_inference=bool(ctx.param("memory_efficient", True)),
            use_amp=True, amp_dtype="bf16", apply_mask=True,
            mask_edges=bool(ctx.param("mask_edges", True)),
            apply_confidence_mask=bool(ctx.param("apply_confidence_mask", False)),
            confidence_percentile=float(ctx.param("confidence_percentile", 10)),
            use_multiview_confidence=bool(ctx.param("multiview_confidence", False)))
    infer_sec = time.time() - t0
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    out_res = str(ctx.param("output_resolution", "input"))
    d_dir = ctx.output_path("depth")
    c_dir = ctx.output_path("confidence")
    v_dir = ctx.output_path("valid")
    T_out, K_out, msf = [], [], []
    pts_all, col_all = [], []
    for i, p in enumerate(preds):
        dz = p["depth_z"][0, ..., 0].float().cpu().numpy()
        conf = p["conf"][0].float().cpu().numpy()
        mask = p["mask"][0, ..., 0].cpu().numpy().astype(bool)
        Kn = p["intrinsics"][0].float().cpu().numpy().astype(np.float64)
        T = p["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        msf.append(float(p["metric_scaling_factor"].reshape(-1)[0]))
        dz = np.where(mask & np.isfinite(dz) & (dz > 0), dz, 0).astype(np.float32)
        # fused cloud from network-resolution world points
        pts = p["pts3d"][0].float().cpu().numpy()[dz > 0]
        img = p["img_no_norm"][0].float().cpu().numpy()[dz > 0]
        pts_all.append(pts)
        col_all.append(np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8))
        if out_res == "input":
            dz_o = _to_input(dz, crops[i], w, h, nearest=True)
            conf_o = _to_input(conf, crops[i], w, h)
            K = _K_to_input(Kn, crops[i])
            W_, H_ = w, h
        else:
            dz_o, conf_o, K, W_, H_ = dz, conf, Kn, Wn, Hn
        np.save(d_dir / io.frame_name(i, ".npy"), dz_o.astype(np.float32))
        np.save(c_dir / io.frame_name(i, ".npy"), conf_o.astype(np.float32))
        io.write_mask(v_dir / io.frame_name(i), (dz_o > 0).astype(np.uint16))
        T_out.append(T)
        K_out.append(K)

    cams = ctx.output_path("cameras", "cameras.json")
    io.write_json(cams, {"cameras": [
        {"K": K.tolist(), "width": W_, "height": H_, "dist": [0, 0, 0, 0, 0]}
        for K in K_out], "frames": [f.name for f in frames]})
    if poses is None:
        pose_source = "predicted by MapAnything from the inputs (" + (
            "images + intrinsics" if Ks is not None else "images only") + (
            " + depth" if depths is not None else "") + ")"
        world = "camera frame of view 0 (MapAnything convention)"
    else:
        pose_source = "INPUT poses used as conditioning; outputs are the model's " \
                      "re-estimate in the input-pose world, NOT an independent check"
        world = "world frame of the input poses"
    if depths is not None:
        from verdi.types.registry import METRIC_SCALE_STATES, read_scale

        st = read_scale(ctx.input("depths"))["scale_status"]
        if depth_metric and st not in METRIC_SCALE_STATES + ("unspecified",):
            raise NodeError(f"depths are declared scale_status={st!r} but "
                            "depths_metric=true", hint="pass depths_metric=false "
                            "or metric depths")
    if poses is not None and not pose_metric:
        scale_status = "input_pose_scale"
    elif depths is not None and not depth_metric and poses is None:
        scale_status = "input_depth_scale"
    else:
        scale_status = "metric"
    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {"T_world_cam": [T.tolist() for T in T_out],
                         "frame_index": list(range(n)),
                         "frames": [f.name for f in frames],
                         "units": "metres" if scale_status == "metric" else scale_status,
                         "scale_status": scale_status, "pose_source": pose_source,
                         "world": world})
    ctx.set_output("depth", d_dir)
    ctx.set_output("confidence", c_dir)
    ctx.set_output("valid", v_dir)
    ctx.set_output("cameras", cams)
    ctx.set_output("trajectory", traj)

    pts = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
    col = np.concatenate(col_all) if col_all else np.zeros((0, 3), np.uint8)
    mx = int(ctx.param("max_points", 1_000_000))
    if len(pts) > mx:
        sel = np.random.default_rng(0).choice(len(pts), mx, replace=False)
        pts, col = pts[sel], col[sel]
    ply = ctx.output_path("pointcloud", "pointcloud.ply")
    _write_ply(ply, pts, col)
    ctx.set_output("pointcloud", ply)
    for p in (d_dir, ply):
        io.write_scale(p, scale_status, "metres" if scale_status == "metric" else scale_status,
                       f"mapanything {MODELS[key]}", pose_source)

    if ctx.has_input("reference_poses"):
        ev = _pose_eval(T_out, _traj(ctx.input("reference_poses"), n, "reference_poses"))
        ep = ctx.output_path("pose_eval", "pose_eval.json")
        io.write_json(ep, ev)
        ctx.set_output("pose_eval", ep)
    info = {
        "model": MODELS[key], "scale_status": scale_status,
        "pose_source": pose_source, "world": world,
        "inputs_used": {"intrinsics": Ks is not None, "poses": poses is not None,
                        "depths": depths is not None,
                        "poses_metric": pose_metric if poses is not None else None,
                        "depths_metric": depth_metric if depths is not None else None},
        "intrinsics_source": "model output (conditioned on input K)" if Ks is not None
                             else "predicted",
        "metric_scaling_factor": msf,
        "network_resolution": [int(Wn), int(Hn)], "input_resolution": [w, h],
        "output_resolution": [W_, H_],
        "crop": {"map": "u_net = sx * u_in + ox, v_net = sy * v_in + oy "
                        "(upstream resize + centre crop)", "per_frame": crops[:1]
                 if Ks is None else crops},
        "convention": {
            "depth": "z-depth (camera optical axis), metres unless scale_status "
                     "says otherwise, 0 = invalid (outside the network crop, "
                     "masked by MapAnything's non-ambiguous / edge mask)",
            "T_world_cam": "OpenCV camera-to-world (MapAnything camera_poses, unchanged)",
            "confidence": "MapAnything per-pixel confidence (learned, uncalibrated)"},
        "params": {"resolution_set": res_set, "mask_edges": bool(ctx.param("mask_edges", True)),
                   "apply_confidence_mask": bool(ctx.param("apply_confidence_mask", False)),
                   "multiview_confidence": bool(ctx.param("multiview_confidence", False))},
        "infer_sec": round(infer_sec, 2), "peak_vram_gib": round(peak, 2),
        "frames": [f.name for f in frames]}
    ip = ctx.output_path("info", "info.json")
    io.write_json(ip, info)
    ctx.set_output("info", ip)
    ctx.metadata.update({k: info[k] for k in ("model", "scale_status", "pose_source",
                                              "network_resolution", "infer_sec",
                                              "peak_vram_gib")})


def _write_ply(path: Path, pts: np.ndarray, col: np.ndarray) -> None:
    n = len(pts)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\n"
              "property uchar blue\nend_header\n")
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = col[:, 0], col[:, 1], col[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


if __name__ == "__main__":
    raise SystemExit(main({"reconstruct": reconstruct}))
