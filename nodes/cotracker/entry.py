"""CoTracker3 node: track_grid / track_points (offline model)."""
from __future__ import annotations

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io


def _video(ctx: Context):
    import torch

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
    arr = np.stack(imgs)  # T H W 3 uint8
    video = torch.from_numpy(arr).permute(0, 3, 1, 2)[None].float()
    return video.to(ctx.device), arr.shape[1], arr.shape[2]


def _model(ctx: Context):
    from cotracker.predictor import CoTrackerPredictor

    ckpt = ctx.weight("cotracker3") / "scaled_offline.pth"
    model = CoTrackerPredictor(checkpoint=str(ckpt), offline=True,
                               window_len=60)
    return model.to(ctx.device).eval()


def _save(ctx: Context, tracks, vis, **extra) -> None:
    xy = tracks[0].float().cpu().numpy().astype(np.float32)   # T N 2
    visible = vis[0].cpu().numpy().astype(bool)               # T N
    if visible.ndim == 3:
        visible = visible[..., 0]
    if not np.isfinite(xy).all():
        raise NodeError("tracker produced non-finite coordinates")
    out = ctx.output_path("tracks", "tracks.npz")
    np.savez_compressed(out, xy=xy, visible=visible, **extra)
    ctx.log(f"[cotracker] tracks {xy.shape}, visible {visible.mean():.3f}")
    ctx.set_output("tracks", out)


def track_grid(ctx: Context) -> None:
    import torch

    video, h, w = _video(ctx)
    t = video.shape[1]
    qf = int(ctx.param("grid_query_frame", 0))
    if not 0 <= qf < t:
        raise NodeError(f"grid_query_frame {qf} outside [0, {t})",
                        kind="request")
    grid = int(ctx.param("grid_size", 20))
    if grid < 1:
        raise NodeError("grid_size must be >= 1", kind="request")
    segm = None
    if ctx.has_input("mask"):
        from PIL import Image

        m = np.asarray(Image.open(ctx.input("mask")))
        if m.shape != (h, w):
            raise NodeError(f"mask {m.shape} != frame size {(h, w)}",
                            kind="request")
        if not (m != 0).any():
            raise NodeError("mask is empty", kind="request")
        segm = torch.from_numpy((m != 0).astype(np.float32))[None, None]
        segm = segm.to(ctx.device)
    model = _model(ctx)
    with torch.inference_mode():
        tracks, vis = model(video, grid_size=grid, grid_query_frame=qf,
                            segm_mask=segm,
                            backward_tracking=bool(
                                ctx.param("backward_tracking", True)))
    n = tracks.shape[2]
    if n == 0:
        raise NodeError("no grid point falls inside the mask",
                        hint="raise grid_size or use a larger mask")
    _save(ctx, tracks, vis, query_frame=np.full(n, qf, np.int32))


def track_points(ctx: Context) -> None:
    import torch

    data = io.read_json(ctx.input("queries"))
    pts = data.get("points", [])
    if not pts:
        raise NodeError("prompt set has no points", kind="request",
                        hint="give points [{xy: [x, y], frame: t}]")
    video, h, w = _video(ctx)
    t = video.shape[1]
    q, obj = [], []
    for i, p in enumerate(pts):
        f = int(p.get("frame", 0))
        x, y = float(p["xy"][0]), float(p["xy"][1])
        if not 0 <= f < t:
            raise NodeError(f"points[{i}].frame {f} outside [0, {t})",
                            kind="request")
        if not (0 <= x < w and 0 <= y < h):
            raise NodeError(f"points[{i}].xy {p['xy']} outside the "
                            f"{w}x{h} frame", kind="request")
        q.append([f, x, y])
        obj.append(int(p.get("obj_id", i + 1)))
    queries = torch.tensor(q, dtype=torch.float32, device=ctx.device)[None]
    model = _model(ctx)
    with torch.inference_mode():
        tracks, vis = model(video, queries=queries,
                            backward_tracking=bool(
                                ctx.param("backward_tracking", True)))
    _save(ctx, tracks, vis,
          query_frame=np.asarray([r[0] for r in q], np.int32),
          obj_id=np.asarray(obj, np.int32))


if __name__ == "__main__":
    raise SystemExit(main({"track_grid": track_grid,
                           "track_points": track_points}))
