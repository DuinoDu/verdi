"""VGGT node: feed-forward multi-view reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

PATCH = 14
SIDE = 518  # VGGT training resolution (longest side)


def _net_size(w: int, h: int):
    """Upstream "crop" sizing without the crop: long side 518, other side
    rounded to a multiple of 14 (keeps the full field of view)."""
    if w >= h:
        W = SIDE
        H = max(PATCH, int(round(h * SIDE / w / PATCH)) * PATCH)
    else:
        H = SIDE
        W = max(PATCH, int(round(w * SIDE / h / PATCH)) * PATCH)
    return W, H


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def reconstruct(ctx: Context) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    frames = io.list_frames(ctx.input("frames"))
    max_frames = int(ctx.param("max_frames", 64))
    if len(frames) < 1:
        raise NodeError("no frames")
    if len(frames) > max_frames:
        raise NodeError(
            f"{len(frames)} frames > max_frames={max_frames}",
            hint="subsample the sequence (e.g. every k-th frame) or raise "
            "max_frames (VRAM grows ~linearly, ~1.5 GB per 10 frames)")
    rgbs = [io.read_image(f) for f in frames]
    h, w = rgbs[0].shape[:2]
    if any(r.shape[:2] != (h, w) for r in rgbs):
        raise NodeError("all frames must have the same resolution")
    W, H = _net_size(w, h)
    x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float() / 255.
    x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False,
                      antialias=True).clamp(0, 1).to(ctx.device)

    model = VGGT.from_pretrained(str(ctx.weight("vggt_1b"))).to(ctx.device)
    model.eval()
    dtype = torch.bfloat16 if ctx.device == "cuda" else torch.float32
    # same call as upstream demo: whole model under bf16 autocast (heads
    # switch autocast off internally)
    with torch.no_grad(), torch.autocast(ctx.device, dtype=dtype,
                                         enabled=ctx.device == "cuda"):
        pred = model(x)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    depth, conf = pred["depth"], pred["depth_conf"]
    extr = extr[0].double().cpu().numpy()          # S,3,4 world->cam (OpenCV)
    intr = intr[0].double().cpu().numpy()          # S,3,3 at H x W
    depth = depth[0, ..., 0].float()               # S,H,W
    conf = conf[0].float()                         # S,H,W

    # ---- cameras at the input resolution
    sx, sy = w / W, h / H
    Ks = intr.copy()
    Ks[:, 0, :] *= sx
    Ks[:, 1, :] *= sy
    T_wc = []
    for E in extr:
        T = np.eye(4)
        T[:3, :4] = E
        T_wc.append(np.linalg.inv(T))
    T_wc = np.stack(T_wc)

    # ---- depth at the input resolution
    conf_min = float(ctx.param("depth_conf_min", 0.0))
    d_full = F.interpolate(depth[:, None], size=(h, w), mode="bilinear",
                           align_corners=False)[:, 0]
    c_full = F.interpolate(conf[:, None], size=(h, w), mode="bilinear",
                           align_corners=False)[:, 0]
    d_full = d_full.cpu().numpy()
    c_full = c_full.cpu().numpy()
    depth_dir = ctx.output_path("depth")
    for i in range(len(frames)):
        d = d_full[i].copy()
        d[(c_full[i] < conf_min) | (d <= 0)] = 0
        io.write_depth(depth_dir / io.frame_name(i, ".npy"), d)

    # ---- point cloud: unproject network-resolution depth with its cameras
    d_net = depth.cpu().numpy()
    c_net = conf.cpu().numpy()
    small = np.stack([np.asarray(Image.fromarray(r).resize((W, H),
                                                           Image.BILINEAR))
                      for r in rgbs])
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    pts, cols, cfs = [], [], []
    for i in range(len(frames)):
        K = intr[i]
        z = d_net[i]
        xc = (uu - K[0, 2]) / K[0, 0] * z
        yc = (vv - K[1, 2]) / K[1, 1] * z
        pc = np.stack([xc, yc, z, np.ones_like(z)], -1).reshape(-1, 4)
        pw = (T_wc[i] @ pc.T).T[:, :3]
        valid = (z.reshape(-1) > 0)
        pts.append(pw[valid])
        cols.append(small[i].reshape(-1, 3)[valid])
        cfs.append(c_net[i].reshape(-1)[valid])
    pts, cols, cfs = np.concatenate(pts), np.concatenate(cols), \
        np.concatenate(cfs)
    pct = float(ctx.param("conf_percentile", 50.0))
    keep = cfs >= max(np.percentile(cfs, pct), 1.0 + 1e-5)
    pts, cols = pts[keep], cols[keep]
    max_points = int(ctx.param("max_points", 1_000_000))
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points,
                                              replace=False)
        pts, cols = pts[sel], cols[sel]
    if len(pts) == 0:
        raise NodeError("no confident points", hint="lower conf_percentile")
    ply = ctx.output_path("pointcloud", "pointcloud.ply")
    _write_ply(ply, pts.astype(np.float32), cols.astype(np.uint8))

    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {"T_world_cam": [T.tolist() for T in T_wc],
                         "frames": [f.name for f in frames],
                         "scale": "up to scale (not metres)"})
    cam = ctx.output_path("camera", "camera.json")
    io.write_camera(cam, Ks.mean(0), w, h)
    intr_path = ctx.output_path("intrinsics", "intrinsics.json")
    io.write_json(intr_path, {"width": w, "height": h,
                              "K": [K.tolist() for K in Ks],
                              "frames": [f.name for f in frames]})
    ctx.set_output("trajectory", traj)
    ctx.set_output("camera", cam)
    ctx.set_output("intrinsics", intr_path)
    ctx.set_output("depth", depth_dir)
    ctx.set_output("pointcloud", ply)
    ctx.metadata.update({
        "network_resolution": [W, H],
        "fx_per_frame": [round(float(K[0, 0]), 2) for K in Ks],
        "baseline_first_last": float(np.linalg.norm(T_wc[-1][:3, 3]
                                                    - T_wc[0][:3, 3])),
        "units": "arbitrary (VGGT normalises scene scale); not metres",
    })


if __name__ == "__main__":
    raise SystemExit(main({"reconstruct": reconstruct}))
