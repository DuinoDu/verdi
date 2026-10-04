"""FoundationPose node: estimate (register) / track (register + track_one)."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

MIN_VALID_PIXELS = 4  # upstream silently returns a guessed pose below this


def _import_upstream(ctx: Context):
    repo = str(ctx.repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        ckpt = ctx.weight("fp") / run / "model_best.pth"
        if not ckpt.exists():
            raise NodeError(f"missing checkpoint {ckpt}", kind="setup",
                            hint="run `verdi setup foundationpose`")
    if not (ctx.repo / "weights").exists():
        raise NodeError(f"{ctx.repo}/weights link missing", kind="setup",
                        hint="run `verdi setup foundationpose`")
    import estimater  # noqa: F401  (upstream module, star-imports Utils)

    if estimater.mycpp is None:
        raise NodeError("FoundationPose mycpp extension not importable",
                        kind="setup",
                        hint="re-run `verdi setup foundationpose` and "
                        "check the cmake/make output")
    return estimater


def _load_mesh(path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise NodeError(f"{path}: could not load a triangle mesh",
                        kind="request")
    vis = mesh.visual
    if isinstance(vis, TextureVisuals):
        mat = vis.material
        if isinstance(mat, PBRMaterial):
            mat = mat.to_simple()  # upstream reads material.image
            vis.material = mat
        if getattr(mat, "image", None) is None or vis.uv is None:
            # texture-less material: upstream falls back to grey colours
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
    extent = float(np.max(mesh.extents))
    if not 0.005 < extent < 5.0:
        raise NodeError(f"mesh extent {extent:.4g} looks not metric",
                        kind="request",
                        hint="the mesh must be in metres (scale it first)")
    return mesh


def _camera(ctx: Context, hw):
    cam = io.read_camera(ctx.input("camera"))
    if (int(cam["height"]), int(cam["width"])) != tuple(hw):
        raise NodeError(
            f"camera is {cam['width']}x{cam['height']} but image is "
            f"{hw[1]}x{hw[0]}", kind="request",
            hint="pass intrinsics of the actual image resolution")
    if np.any(np.abs(cam["dist"]) > 1e-6):
        ctx.log("warning: camera distortion is ignored (undistort first)")
    return cam["K"].astype(np.float64)


def _depth(path, hw):
    d = np.load(path).astype(np.float32)
    if d.shape != tuple(hw):
        raise NodeError(f"depth {path} is {d.shape}, image is {tuple(hw)}",
                        kind="request", hint="depth must be aligned with rgb")
    d[~np.isfinite(d) | (d < 0.001)] = 0
    return d


def _mask(ctx: Context, hw):
    from PIL import Image

    ids = np.asarray(Image.open(ctx.input("mask")))
    if ids.ndim == 3:
        ids = ids[..., 0]
    if ids.shape != tuple(hw):
        raise NodeError(f"mask is {ids.shape}, image is {tuple(hw)}",
                        kind="request")
    mid = int(ctx.param("mask_id", 0))
    m = ids == mid if mid else ids != 0
    if not m.any():
        raise NodeError("object mask is empty", kind="request",
                        hint="check mask / mask_id")
    return m


def _estimator(ctx: Context, fp, mesh):
    import nvdiffrast.torch as dr

    fp.set_seed(int(ctx.param("seed", 0)))
    debug_dir = tempfile.mkdtemp(prefix="fp_debug_")
    return fp.FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
        mesh=mesh, scorer=fp.ScorePredictor(),
        refiner=fp.PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(),
        debug=0, debug_dir=debug_dir)


def _register(ctx, est, K, rgb, depth, mask, iters):
    valid = (depth >= 0.001) & mask
    if valid.sum() < MIN_VALID_PIXELS:
        raise NodeError(
            f"only {int(valid.sum())} masked pixels have valid depth",
            kind="request",
            hint="mask and depth must overlap (depth in metres, 0=invalid)")
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                        iteration=int(iters))
    scores = getattr(est, "scores", None)
    score = float(scores[0]) if scores is not None and len(scores) else None
    return np.asarray(pose, dtype=np.float64).reshape(4, 4), score


def _check_pose(T, where):
    if not np.all(np.isfinite(T)):
        raise NodeError(f"{where}: FoundationPose returned a non-finite pose")


def estimate(ctx: Context) -> None:
    rgb = io.read_image(ctx.input("image"))
    hw = rgb.shape[:2]
    K = _camera(ctx, hw)
    depth = _depth(ctx.input("depth"), hw)
    mask = _mask(ctx, hw)
    mesh = _load_mesh(ctx.input("mesh"))
    fp = _import_upstream(ctx)
    est = _estimator(ctx, fp, mesh)
    T, score = _register(ctx, est, K, rgb, depth, mask,
                         ctx.param("refine_iter", 5))
    _check_pose(T, "estimate")
    out = ctx.output_path("poses", "poses.json")
    io.write_pose_set(out, [{"T_cam_obj": T, "label": ctx.param("label"),
                             "score": score}])
    ctx.set_output("poses", out)


def track(ctx: Context) -> None:
    frames = io.list_frames(ctx.input("frames"))
    depths = io.list_frames(ctx.input("depths"), (".npy",))
    if len(frames) != len(depths):
        raise NodeError(f"{len(frames)} frames but {len(depths)} depth maps",
                        kind="request",
                        hint="frames and depths must pair 1:1 (sorted)")
    rgb0 = io.read_image(frames[0])
    hw = rgb0.shape[:2]
    K = _camera(ctx, hw)
    mask = _mask(ctx, hw)
    mesh = _load_mesh(ctx.input("mesh"))
    fp = _import_upstream(ctx)
    est = _estimator(ctx, fp, mesh)
    label = ctx.param("label")
    out_frames = []
    for i, (f, d) in enumerate(zip(frames, depths)):
        rgb = rgb0 if i == 0 else io.read_image(f)
        if rgb.shape[:2] != hw:
            raise NodeError(f"{f.name}: size differs from first frame",
                            kind="request")
        depth = _depth(d, hw)
        if i == 0:
            T, score = _register(ctx, est, K, rgb, depth, mask,
                                 ctx.param("refine_iter", 5))
            entry = {"T_cam_obj": T.tolist(), "label": label,
                     "score": score}
        else:
            T = np.asarray(est.track_one(
                rgb=rgb, depth=depth, K=K,
                iteration=int(ctx.param("track_iter", 2))),
                dtype=np.float64).reshape(4, 4)
            entry = {"T_cam_obj": T.tolist(), "label": label}
        _check_pose(T, f"frame {i}")
        out_frames.append({"frame": i, "poses": [entry]})
        ctx.log(f"frame {i}: t = {np.round(T[:3, 3], 4).tolist()}")
    out = ctx.output_path("poses", "poses.json")
    io.write_json(out, {"frames": out_frames})
    ctx.set_output("poses", out)


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    raise SystemExit(main({"estimate": estimate, "track": track}))
