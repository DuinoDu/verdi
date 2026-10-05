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
        raise NodeError("camera has distortion; FoundationPose renders with a "
                        "pinhole K and would silently ignore it", kind="request",
                        hint="undistort / rectify the RGB-D frames first and pass "
                             "the new pinhole K (dist = 0)")
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


class _Diag:
    """Per-frame evidence for a pose: render the mesh and compare with the
    observed depth (and, if given, with an independent object mask)."""

    def __init__(self, ctx: Context, fp, est, K, hw):
        import nvdiffrast.torch as dr

        self.fp, self.K, self.hw = fp, K, hw
        self.tensors = fp.make_mesh_tensors(est.mesh_ori)
        self.glctx = dr.RasterizeCudaContext()
        self.tol = float(ctx.param("depth_tol_m", 0.01))
        self.min_inl = float(ctx.param("min_depth_inlier_ratio", 0.3))
        self.min_iou = float(ctx.param("min_mask_iou", 0.5))

    def __call__(self, T, depth, mask=None):
        import torch

        H, W = self.hw
        out = {"finite": bool(np.all(np.isfinite(T)))}
        if not out["finite"]:
            return {**out, "valid": False, "reason": "non-finite pose"}
        z = float(T[2, 3])
        uvw = self.K @ T[:3, 3]
        uv = (uvw[:2] / uvw[2]).tolist() if uvw[2] > 1e-9 else None
        out.update({"z_m": z, "center_px": uv,
                    "center_in_image": bool(uv is not None and 0 <= uv[0] < W
                                            and 0 <= uv[1] < H and z > 0)})
        with torch.no_grad():
            _, rd, _ = self.fp.nvdiffrast_render(
                K=self.K, H=H, W=W,
                ob_in_cams=torch.as_tensor(T[None], dtype=torch.float32, device="cuda"),
                glctx=self.glctx, mesh_tensors=self.tensors, output_size=(H, W))
        rd = rd[0].detach().cpu().numpy().astype(np.float32)
        sil = rd > 0
        area = int(sil.sum())
        out["rendered_area_px"] = area
        if area == 0:
            return {**out, "valid": False, "reason": "object not in view at this pose"}
        obs = depth[sil]
        rend = rd[sil]
        have = obs > 0
        res = obs - rend
        inl = have & (np.abs(res) < self.tol)
        occl = have & (res < -self.tol)       # something in front of the object
        out.update({
            "observed_depth_coverage": float(have.mean()),
            "depth_inlier_ratio": float(inl.sum() / max(have.sum(), 1)),
            "occluded_ratio": float(occl.sum() / max(have.sum(), 1)),
            "behind_ratio": float((have & (res > self.tol)).sum() / max(have.sum(), 1)),
            "median_abs_residual_m": float(np.median(np.abs(res[have]))) if have.any() else None,
            "depth_tol_m": self.tol})
        reasons = []
        if not out["center_in_image"]:
            reasons.append("centre outside image / behind camera")
        if out["depth_inlier_ratio"] < self.min_inl:
            reasons.append(f"depth_inlier_ratio {out['depth_inlier_ratio']:.2f} < {self.min_inl}")
        if mask is not None:
            inter = int((sil & mask).sum())
            union = int((sil | mask).sum())
            out["mask_iou"] = inter / union if union else 0.0
            if out["mask_iou"] < self.min_iou:
                reasons.append(f"mask_iou {out['mask_iou']:.2f} < {self.min_iou}")
        out["valid"] = not reasons
        out["reason"] = "; ".join(reasons) or None
        return out


def _seq_masks(ctx: Context, n: int, hw):
    if not ctx.has_input("masks"):
        return None
    from PIL import Image

    files = io.list_frames(ctx.input("masks"), (".png",))
    if len(files) != n:
        raise NodeError(f"masks has {len(files)} frames for {n} RGB frames",
                        kind="request")
    mid = int(ctx.param("mask_id", 0))
    out = []
    for f in files:
        ids = np.asarray(Image.open(f))
        if ids.shape[:2] != tuple(hw):
            raise NodeError(f"mask {f.name} size differs from frames", kind="request")
        out.append(ids == mid if mid else ids != 0)
    return out


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
    d = _Diag(ctx, fp, est, K, hw)(T, depth, mask)
    out = ctx.output_path("poses", "poses.json")
    io.write_pose_set(out, [{"T_cam_obj": T, "label": ctx.param("label"),
                             "score": score, "valid": d["valid"],
                             "diagnostics": d}])
    ctx.set_output("poses", out)
    ctx.metadata.update({"valid": d["valid"], "diagnostics": d})


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
    masks = _seq_masks(ctx, len(frames), hw)
    diag = None
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
        if diag is None:
            diag = _Diag(ctx, fp, est, K, hw)
        dg = diag(T, depth, masks[i] if masks is not None else
                  (mask if i == 0 else None))
        entry.update({"valid": dg["valid"], "diagnostics": dg})
        out_frames.append({"frame": i, "name": f.name, "valid": dg["valid"],
                           "poses": [entry]})
        ctx.log(f"frame {i}: t = {np.round(T[:3, 3], 4).tolist()} valid={dg['valid']} "
                f"inlier={dg.get('depth_inlier_ratio')}")
    out = ctx.output_path("poses", "poses.json")
    io.write_json(out, {"convention": {
        "T_cam_obj": "4x4 mesh frame -> OpenCV camera frame of that frame, metres",
        "valid": "heuristic from evidence, NOT a calibrated confidence: pose "
                 "finite, object centre in front of the camera and inside the "
                 "image, depth_inlier_ratio >= min_depth_inlier_ratio (rendered "
                 "mesh depth vs observed depth within depth_tol_m) and, when "
                 "masks are given, mask_iou >= min_mask_iou. Occlusion lowers "
                 "the inlier ratio (see occluded_ratio). track_one never "
                 "re-initialises: after a loss the pose keeps drifting.",
        "frame": "index into the sorted input frames (name = file name)"},
        "frames": out_frames})
    ctx.set_output("poses", out)
    ctx.metadata["valid_frames"] = int(sum(f["valid"] for f in out_frames))
    ctx.metadata["frames"] = len(out_frames)


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    raise SystemExit(main({"estimate": estimate, "track": track}))
