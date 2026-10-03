"""SpatialTrackerV2 node: track (2D + 3D tracks, camera trajectory, depth)."""
from __future__ import annotations

import sys

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io


def _frames(ctx: Context):
    frames = io.list_frames(ctx.input("frames"))
    max_frames = int(ctx.param("max_frames", 0) or 0)
    if max_frames > 0:
        frames = frames[:max_frames]
    if len(frames) < 2:
        raise NodeError(f"need at least 2 frames, got {len(frames)}",
                        kind="request")
    imgs = [io.read_image(f) for f in frames]
    if len({a.shape for a in imgs}) != 1:
        raise NodeError("frames differ in size", kind="request")
    arr = np.stack(imgs)
    if arr.shape[1] > arr.shape[2]:
        raise NodeError(
            f"portrait frames ({arr.shape[2]}x{arr.shape[1]}) are not "
            "supported: VGGT4Track center-crops them",
            hint="rotate the frames to landscape first", kind="request")
    return arr


def _resize_nearest(a: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    return cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST)


def _rgbd_inputs(ctx: Context, t: int, mh: int, mw: int):
    """User depth/camera/trajectory resized to the model resolution."""
    if not ctx.has_input("depth"):
        if ctx.has_input("camera") or ctx.has_input("trajectory"):
            raise NodeError("camera/trajectory are only used together with "
                            "depth", kind="request",
                            hint="also pass -i depth=<depth_seq dir>")
        return None
    if not ctx.has_input("camera"):
        raise NodeError("depth input needs camera intrinsics",
                        hint="pass -i camera=camera.json", kind="request")
    files = io.list_frames(ctx.input("depth"), (".npy",))[:t]
    if len(files) != t:
        raise NodeError(f"depth has {len(files)} frames, frames has {t}",
                        kind="request")
    depth = np.stack([_resize_nearest(np.load(f).astype(np.float32), mh, mw)
                      for f in files])
    cam = io.read_camera(ctx.input("camera"))
    K = cam["K"].copy()
    K[0] *= mw / float(cam["width"])
    K[1] *= mh / float(cam["height"])
    intrs = np.repeat(K[None].astype(np.float32), t, 0)
    extrs = None
    if ctx.has_input("trajectory"):
        Ts = np.asarray(io.read_json(ctx.input("trajectory"))["T_world_cam"],
                        np.float32)
        if len(Ts) < t:
            raise NodeError(f"trajectory has {len(Ts)} poses, need {t}",
                            kind="request")
        extrs = Ts[:t]
    return depth, intrs, extrs


def _queries(ctx: Context, t: int, h: int, w: int, mh: int, mw: int):
    import torch
    from models.SpaTrackV2.models.utils import get_points_on_a_grid

    if ctx.has_input("queries"):
        pts = io.read_json(ctx.input("queries")).get("points", [])
        if not pts:
            raise NodeError("prompt set has no points", kind="request")
        q = []
        for i, p in enumerate(pts):
            f = int(p.get("frame", 0))
            x, y = float(p["xy"][0]), float(p["xy"][1])
            if not 0 <= f < t or not (0 <= x < w and 0 <= y < h):
                raise NodeError(f"points[{i}] (frame {f}, xy {p['xy']}) is "
                                f"outside {t} frames of {w}x{h}",
                                kind="request")
            q.append([f, x * mw / w, y * mh / h])
        return np.asarray(q, np.float32)
    grid = int(ctx.param("grid_size", 20))
    if grid < 1:
        raise NodeError("grid_size must be >= 1", kind="request")
    pts = get_points_on_a_grid(grid, (mh, mw), device="cpu")  # 1 N 2 (x, y)
    if ctx.has_input("mask"):
        from PIL import Image

        m = np.asarray(Image.open(ctx.input("mask"))) != 0
        if m.shape != (h, w):
            raise NodeError(f"mask {m.shape} != frame size {(h, w)}",
                            kind="request")
        m = _resize_nearest(m.astype(np.uint8), mh, mw) > 0
        ij = pts[0].long()
        keep = torch.from_numpy(m[ij[:, 1].clamp(0, mh - 1).numpy(),
                                  ij[:, 0].clamp(0, mw - 1).numpy()])
        pts = pts[:, keep]
        if pts.shape[1] == 0:
            raise NodeError("no grid point falls inside the mask",
                            hint="raise grid_size or use a larger mask")
    return torch.cat([torch.zeros_like(pts[:, :, :1]), pts],
                     dim=2)[0].numpy()


def track(ctx: Context) -> None:
    import torch
    import torch.nn.functional as F

    if ctx.device != "cuda":
        raise NodeError("SpatialTrackerV2 needs a CUDA GPU", kind="request")
    sys.path.insert(0, str(ctx.repo))
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import \
        VGGT4Track
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import \
        preprocess_image

    arr = _frames(ctx)
    t, h, w = arr.shape[:3]
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
    video = preprocess_image(video).clamp(0, 255)  # width 518, h % 14 == 0
    mh, mw = video.shape[2:]
    ctx.log(f"[spatracker] {t} frames {w}x{h} -> model {mw}x{mh}")
    rgbd = _rgbd_inputs(ctx, t, mh, mw)

    extrs_vggt = None
    if rgbd is None or rgbd[2] is None:
        vggt = VGGT4Track.from_pretrained(str(ctx.weight("front")))
        vggt.eval().to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda",
                                                 dtype=torch.bfloat16):
            pred = vggt(video[None].cuda() / 255)
        extrs_vggt = pred["poses_pred"][0].float().cpu().numpy()
        if rgbd is None:
            depth = pred["points_map"][..., 2].reshape(t, mh, mw)
            depth = depth.float().cpu().numpy()
            intrs = pred["intrs"][0].float().cpu().numpy()
            unc = pred["unc_metric"].reshape(t, mh, mw).float().cpu() \
                .numpy() > 0.5
        del vggt, pred
        torch.cuda.empty_cache()
    if rgbd is not None:
        depth, intrs, extrs = rgbd
        unc = depth > 0
        if extrs is None:
            extrs = extrs_vggt
            ctx.log("[spatracker] no trajectory given: VGGT4Track poses "
                    "initialise the camera")
    else:
        extrs = extrs_vggt

    queries = _queries(ctx, t, h, w, mh, mw)
    model = Predictor.from_pretrained(str(ctx.weight("offline")))
    model.spatrack.track_num = int(ctx.param("vo_points", 756))
    model.eval()
    model.to("cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (c2w, intrs_o, point_map, conf_depth, track3d, track2d, vis, conf,
         video_o) = model.forward(
            video, depth=depth, intrs=intrs, extrs=extrs, queries=queries,
            fps=1, full_point=False,
            iters_track=int(ctx.param("iters_track", 4)),
            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc,
            support_frame=t - 1, replace_ratio=0.2)

    oh, ow = video_o.shape[2:]
    sx, sy = w / ow, h / oh
    c2w = c2w.float().cpu()
    xy = track2d[..., :2].float().cpu().numpy() * np.array([sx, sy])
    vis_prob = vis[..., 0].float().cpu().numpy()
    ctx.log(f"[spatracker] vis range {vis_prob.min():.3f}..{vis_prob.max():.3f}")
    # camera-to-world, re-expressed relative to frame 0 and re-orthonormalised
    # (bf16 output); world = camera of frame 0
    Ts = c2w.numpy().astype(np.float64)
    Ts = np.linalg.inv(Ts[0]) @ Ts
    for T in Ts:
        u, _, vt = np.linalg.svd(T[:3, :3])
        T[:3, :3] = u @ vt
        T[3] = [0, 0, 0, 1]
    xyz_cam = track3d[..., :3].float().cpu().numpy().astype(np.float64)
    xyz_world = (np.einsum("tij,tnj->tni", Ts[:, :3, :3], xyz_cam)
                 + Ts[:, :3, 3][:, None, :])
    K = intrs_o.float().cpu().numpy().copy()
    K[:, 0] *= sx
    K[:, 1] *= sy
    if not (np.isfinite(xy).all() and np.isfinite(xyz_world).all()
            and np.isfinite(Ts).all()):
        raise NodeError("model produced non-finite tracks/poses")

    out = ctx.output_path("tracks", "tracks.npz")
    np.savez_compressed(
        out, xy=xy.astype(np.float32), visible=vis_prob > 0.5,
        vis_prob=vis_prob.astype(np.float32),
        conf=conf[..., 0].float().cpu().numpy().astype(np.float32),
        xyz_cam=xyz_cam.astype(np.float32),
        xyz_world=xyz_world.astype(np.float32),
        query_frame=queries[:, 0].astype(np.int32), K=K.astype(np.float32))
    ctx.set_output("tracks", out)

    traj = ctx.output_path("trajectory", "trajectory.json")
    io.write_json(traj, {"T_world_cam": Ts.tolist(),
                         "frames": [int(i) for i in range(t)]})
    ctx.set_output("trajectory", traj)

    z = point_map[:, 2].float()
    z[conf_depth.float() < 0.5] = 0
    z = F.interpolate(z[:, None], size=(h, w), mode="nearest")[:, 0]
    ddir = ctx.output_path("depth")
    for i, d in enumerate(z.cpu().numpy()):
        io.write_depth(ddir / io.frame_name(i, ".npy"), d)
    ctx.set_output("depth", ddir)

    cam = ctx.output_path("camera", "camera.json")
    io.write_camera(cam, K[0], w, h)
    ctx.set_output("camera", cam)
    ctx.metadata.update({"model_size": [int(mw), int(mh)],
                         "points": int(len(queries)),
                         "depth_source": "input" if rgbd else "vggt4track"})


if __name__ == "__main__":
    raise SystemExit(main({"track": track}))
