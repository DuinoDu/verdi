"""MoGe-2 node: metric point map / depth / normal / intrinsics from one image."""
from __future__ import annotations

import math

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

CHECKPOINTS = {"vitl": "moge2_vitl", "vitb": "moge2_vitb", "vits": "moge2_vits"}


def _load(ctx: Context):
    import torch
    from moge.model.v2 import MoGeModel

    key = CHECKPOINTS.get(ctx.param("checkpoint", "vitl"))
    if key is None:
        raise NodeError(f"unknown checkpoint {ctx.param('checkpoint')!r}",
                        kind="request")
    ckpt = ctx.weight(key, prefetch=True) / "model.pt"
    model = MoGeModel.from_pretrained(str(ckpt)).to(ctx.device).eval()
    if ctx.device == "cuda" and ctx.param("precision", "fp16") == "fp16":
        model.half()      # upstream infer.py --fp16 (model weights in fp16)
    torch.set_grad_enabled(False)
    return model


def _infer(ctx: Context, model, rgb: np.ndarray, fov_x_deg=None):
    """Run MoGe-2; return numpy dict (points/depth/normal/mask/K px)."""
    import torch

    h, w = rgb.shape[:2]
    t = torch.tensor(rgb / 255.0, dtype=torch.float32,
                     device=ctx.device).permute(2, 0, 1)
    num_tokens = int(ctx.param("num_tokens", 0)) or None
    out = model.infer(t, num_tokens=num_tokens,
                      resolution_level=int(ctx.param("resolution_level", 9)),
                      fov_x=fov_x_deg,
                      use_fp16=ctx.device == "cuda"
                      and ctx.param("precision", "fp16") == "fp16")
    res = {k: v.float().cpu().numpy() for k, v in out.items()}
    if "points" not in res or "mask" not in res:
        raise NodeError("MoGe returned no point map / mask",
                        hint="checkpoint without metric point head?")
    mask = res["mask"].astype(bool) & np.isfinite(res["depth"]) \
        & (res["depth"] > 0)
    if mask.mean() < 0.01:
        raise NodeError("MoGe marked <1% of the pixels valid",
                        hint="input is not a perspective photo of a scene?")
    Kn = res["intrinsics"]            # normalised (fx/W, fy/H, cx/W, cy/H)
    K = np.array([[Kn[0, 0] * w, 0, Kn[0, 2] * w],
                  [0, Kn[1, 1] * h, Kn[1, 2] * h], [0, 0, 1]], dtype=float)
    points = np.where(mask[..., None], res["points"], 0).astype(np.float32)
    depth = np.where(mask, res["depth"], 0).astype(np.float32)
    normal = res.get("normal")
    if normal is not None:
        normal = np.where(mask[..., None], normal, 0).astype(np.float32)
    return {"points": points, "depth": depth, "normal": normal,
            "mask": mask, "K": K}


def _fov_from_camera(ctx: Context, w: int, h: int):
    if not ctx.has_input("camera"):
        return None, None
    cam = io.read_camera(ctx.input("camera"))
    if int(cam["width"]) != w or int(cam["height"]) != h:
        raise NodeError(
            f"camera is {cam['width']}x{cam['height']} but image is {w}x{h}",
            hint="scale K to the image resolution first", kind="request")
    fx = float(cam["K"][0, 0])
    return math.degrees(2 * math.atan(0.5 * w / fx)), cam


def _fov_param(ctx: Context):
    fov = float(ctx.param("fov_x", 0.0))
    return fov if fov > 0 else None


def _write_ply(path, pts: np.ndarray, rgb: np.ndarray) -> None:
    n = len(pts)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\n"
              "property uchar blue\nend_header\n").encode()
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def _edge_mask(depth: np.ndarray, valid: np.ndarray, rtol: float):
    """Upstream utils3d depth_map_edge: pixels whose 3x3 depth range is
    > rtol * depth (flying pixels at occlusion boundaries)."""
    import utils3d

    if rtol <= 0:
        return np.zeros_like(valid)
    d = np.where(valid, depth, 1.0).astype(np.float32)
    return np.asarray(utils3d.np.depth_map_edge(d, rtol=rtol, mask=valid),
                      dtype=bool)


def estimate(ctx: Context) -> None:
    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    fov, cam = _fov_from_camera(ctx, w, h)
    fov = fov if fov is not None else _fov_param(ctx)
    model = _load(ctx)
    r = _infer(ctx, model, rgb, fov)

    ctx.set_output("pointmap", _save_npy(ctx.output_path(
        "pointmap", "pointmap.npy"), r["points"]))
    ctx.set_output("depth", io.write_depth(
        ctx.output_path("depth", "depth.npy"), r["depth"]))
    if r["normal"] is None:
        raise NodeError("checkpoint has no normal head",
                        hint="use a *-normal MoGe-2 checkpoint")
    ctx.set_output("normal", _save_npy(ctx.output_path(
        "normal", "normal.npy"), r["normal"]))
    ctx.set_output("mask", io.write_mask(
        ctx.output_path("mask", "mask.png"), r["mask"].astype(np.uint8),
        {1: {"label": "valid"}}))
    ctx.set_output("camera", io.write_camera(
        ctx.output_path("camera", "camera.json"), r["K"], w, h))

    keep = r["mask"] & ~_edge_mask(r["depth"], r["mask"],
                                   float(ctx.param("edge_rtol", 0.04)))
    _write_ply(ctx.output_path("pointcloud", "pointcloud.ply"),
               r["points"][keep], rgb[keep])
    ctx.set_output("pointcloud", ctx.output_dir / "pointcloud.ply")
    K = r["K"]
    ctx.metadata.update({
        "fx_px": float(K[0, 0]), "fy_px": float(K[1, 1]),
        "fov_x_deg": float(math.degrees(2 * math.atan(0.5 * w / K[0, 0]))),
        "fov_given": fov is not None, "valid_ratio": float(r["mask"].mean()),
        "checkpoint": ctx.param("checkpoint", "vitl")})


def _save_npy(path, arr):
    np.save(path, np.ascontiguousarray(arr, dtype=np.float32))
    return path


def estimate_seq(ctx: Context) -> None:
    frames = io.list_frames(ctx.input("frames"))
    if not frames:
        raise NodeError("no frames found", kind="request")
    rgb0 = io.read_image(frames[0])
    h, w = rgb0.shape[:2]
    fov, cam = _fov_from_camera(ctx, w, h)
    fov = fov if fov is not None else _fov_param(ctx)
    model = _load(ctx)
    mode = ctx.param("fov", "median")
    per_frame_fov = []
    if fov is None and mode == "median":
        # pass 1: per-frame FOV estimates, pass 2 with the shared median FOV
        for f in frames:
            r = _infer(ctx, model, io.read_image(f), None)
            per_frame_fov.append(math.degrees(
                2 * math.atan(0.5 * w / r["K"][0, 0])))
        fov = float(np.median(per_frame_fov))
        ctx.log(f"median fov_x {fov:.2f} deg (per frame {per_frame_fov})")
    d_dir, p_dir = ctx.output_path("depth"), ctx.output_path("pointmap")
    cams = []
    for i, f in enumerate(frames):
        rgb = io.read_image(f)
        if rgb.shape[:2] != (h, w):
            raise NodeError(f"frame {f.name} has size {rgb.shape[:2]} != "
                            f"{(h, w)}", kind="request")
        r = _infer(ctx, model, rgb, fov)
        io.write_depth(d_dir / io.frame_name(i, ".npy"), r["depth"])
        _save_npy(p_dir / io.frame_name(i, ".npy"), r["points"])
        cams.append({"K": r["K"].tolist(), "width": w, "height": h,
                     "dist": [0, 0, 0, 0, 0]})
    ctx.set_output("depth", d_dir)
    ctx.set_output("pointmap", p_dir)
    ctx.set_output("camera", io.write_json(
        ctx.output_path("camera", "cameras.json"), {"cameras": cams}))
    ctx.metadata.update({"fov_x_deg": fov, "per_frame_fov_x_deg":
                         per_frame_fov, "fov_mode": mode if not ctx.has_input(
                             "camera") else "camera"})


if __name__ == "__main__":
    raise SystemExit(main({"estimate": estimate, "estimate_seq": estimate_seq}))
