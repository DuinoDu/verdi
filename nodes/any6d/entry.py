"""Any6D node: estimate (pose + metric size) / register_candidates."""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

from pdebug.sdk import Context, NodeError, main
from pdebug.types import io

MIN_VALID_PIXELS = 4  # upstream silently returns a guessed pose below this


def _import_upstream(ctx: Context):
    repo = ctx.repo
    for p in (str(repo / "foundationpose"), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        ckpt = ctx.weight("fp") / run / "model_best.pth"
        if not ckpt.exists():
            raise NodeError(f"missing checkpoint {ckpt}", kind="setup",
                            hint="run `otn-cli setup any6d`")
    if not (repo / "foundationpose" / "weights").exists():
        raise NodeError(f"{repo}/foundationpose/weights link missing",
                        kind="setup", hint="run `otn-cli setup any6d`")
    # upstream resolves mycpp relative to foundationpose/ (cwd-style import)
    cwd = os.getcwd()
    os.chdir(repo / "foundationpose")
    try:
        import estimater  # noqa: F401  (builds scorer/refiner as defaults)
    finally:
        os.chdir(cwd)
    if getattr(estimater, "mycpp", None) is None:
        raise NodeError("mycpp extension not importable", kind="setup",
                        hint="re-run `otn-cli setup any6d` and check the "
                        "cmake/make output")
    return estimater


def _load_mesh(path, metric: bool):
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise NodeError(f"{path}: could not load a triangle mesh",
                        kind="request")
    vis = mesh.visual
    if isinstance(vis, TextureVisuals):
        mat = vis.material
        img = getattr(mat, "baseColorTexture", None) or getattr(
            mat, "image", None)
        if img is None or vis.uv is None:
            # texture-less material: render with plain vertex colours
            mesh.visual = trimesh.visual.ColorVisuals(mesh)
    extent = float(np.max(mesh.extents))
    if metric and not 0.005 < extent < 5.0:
        raise NodeError(f"mesh extent {extent:.4g} looks not metric",
                        kind="request",
                        hint="this task needs a mesh in metres; use task "
                        "`estimate` to also estimate the scale")
    return mesh


def _inputs(ctx: Context):
    from PIL import Image

    rgb = io.read_image(ctx.input("image"))
    hw = rgb.shape[:2]
    cam = io.read_camera(ctx.input("camera"))
    if (int(cam["height"]), int(cam["width"])) != tuple(hw):
        raise NodeError(
            f"camera is {cam['width']}x{cam['height']} but image is "
            f"{hw[1]}x{hw[0]}", kind="request",
            hint="pass intrinsics of the actual image resolution")
    if np.any(np.abs(cam["dist"]) > 1e-6):
        ctx.log("warning: camera distortion is ignored (undistort first)")
    depth = np.load(ctx.input("depth")).astype(np.float32)
    if depth.shape != tuple(hw):
        raise NodeError(f"depth is {depth.shape}, image is {tuple(hw)}",
                        kind="request", hint="depth must be aligned with rgb")
    depth[~np.isfinite(depth) | (depth < 0.001)] = 0
    ids = np.asarray(Image.open(ctx.input("mask")))
    if ids.ndim == 3:
        ids = ids[..., 0]
    if ids.shape != tuple(hw):
        raise NodeError(f"mask is {ids.shape}, image is {tuple(hw)}",
                        kind="request")
    mid = int(ctx.param("mask_id", 0))
    mask = ids == mid if mid else ids != 0
    valid = int(((depth >= 0.001) & mask).sum())
    if valid < MIN_VALID_PIXELS:
        raise NodeError(f"only {valid} masked pixels have valid depth",
                        kind="request",
                        hint="mask and depth must overlap (metres, 0=invalid)")
    return rgb, depth, mask, cam["K"].astype(np.float64)


def _estimator(ctx: Context, a6d, mesh):
    import nvdiffrast.torch as dr

    debug_dir = ctx.output_path("_debug")  # upstream always exports here
    est = a6d.Any6D(symmetry_tfs=None, mesh=mesh,
                    scorer=a6d.ScorePredictor(),
                    refiner=a6d.PoseRefinePredictor(),
                    glctx=dr.RasterizeCudaContext(), debug=0,
                    debug_dir=str(debug_dir))
    return est, debug_dir


def estimate(ctx: Context) -> None:
    import trimesh

    a6d = _import_upstream(ctx)
    rgb, depth, mask, K = _inputs(ctx)
    mesh_in = _load_mesh(ctx.input("mesh"), metric=False)
    est, debug_dir = _estimator(ctx, a6d, mesh_in)
    pose = est.register_any6d(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                              iteration=int(ctx.param("iteration", 5)),
                              name="node")
    pose = np.asarray(pose, dtype=float)
    shutil.rmtree(debug_dir, ignore_errors=True)
    out_mesh: trimesh.Trimesh = est.mesh
    v_in = np.asarray(mesh_in.vertices, float)
    v_out = np.asarray(out_mesh.vertices, float)
    if v_in.shape != v_out.shape:
        raise NodeError("upstream changed the mesh topology", kind="node")
    # upstream only scales (per axis) and translates the mesh: fit it
    affine = np.eye(4)
    for k in range(3):
        A = np.stack([v_in[:, k], np.ones(len(v_in))], 1)
        (s, t), *_ = np.linalg.lstsq(A, v_out[:, k], rcond=None)
        affine[k, k], affine[k, 3] = s, t
    resid = float(np.abs(v_in @ affine[:3, :3].T + affine[:3, 3]
                         - v_out).max())
    mesh_path = ctx.output_path("mesh", "mesh.glb")
    out_mesh.export(str(mesh_path))
    ctx.set_output("mesh", mesh_path)
    poses_path = ctx.output_path("poses", "poses.json")
    score = float(est.scores[0]) if getattr(est, "scores", None) is not None \
        else None
    io.write_pose_set(poses_path, [{
        "T_cam_obj": pose, "label": ctx.param("label", "object"),
        "score": score,
        "size": [float(x) for x in out_mesh.extents],
        "scale": [float(affine[k, k]) for k in range(3)],
        "input_to_mesh": affine.tolist(),
    }])
    ctx.set_output("poses", poses_path)
    ctx.metadata.update({"scale": [float(affine[k, k]) for k in range(3)],
                         "scale_fit_residual": resid})


def register_candidates(ctx: Context) -> None:
    import torch

    a6d = _import_upstream(ctx)
    rgb, depth, mask, K = _inputs(ctx)
    mesh = _load_mesh(ctx.input("mesh"), metric=True)
    est, debug_dir = _estimator(ctx, a6d, mesh)
    pose_data, ids = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                  iteration=int(ctx.param("iteration", 5)),
                                  name="node", return_all_poses=True)
    shutil.rmtree(debug_dir, ignore_errors=True)
    ids = ids.cpu().numpy()
    to_mesh = est.get_tf_to_centered_mesh()
    poses = (est.poses @ to_mesh).cpu().numpy()  # sorted by score
    scores = torch.as_tensor(est.scores).float().cpu().numpy()
    k = min(int(ctx.param("top_k", 25)), len(poses))
    label = ctx.param("label", "object")
    poses_path = ctx.output_path("poses", "poses.json")
    io.write_pose_set(poses_path, [
        {"T_cam_obj": poses[i], "label": label, "score": float(scores[i]),
         "rank": i} for i in range(k)])
    ctx.set_output("poses", poses_path)
    rend = ctx.output_path("renders")
    for r in range(k):
        img = pose_data.rgbAs[ids[r]].permute(1, 2, 0).float().cpu().numpy()
        io.write_image(rend / io.frame_name(r),
                       (img * 255).clip(0, 255).astype(np.uint8))
    ctx.set_output("renders", rend)
    ref = pose_data.rgbBs[ids[0]].permute(1, 2, 0).float().cpu().numpy()
    ref_path = ctx.output_path("reference", "reference.png")
    io.write_image(ref_path, (ref * 255).clip(0, 255).astype(np.uint8))
    ctx.set_output("reference", ref_path)
    ctx.metadata["hypotheses"] = int(len(poses))


if __name__ == "__main__":
    raise SystemExit(main({"estimate": estimate,
                           "register_candidates": register_candidates}))
