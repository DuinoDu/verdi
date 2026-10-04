"""GenPose++ (GenPose2) node: category-level 6D pose + size from RGB-D + mask."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from verdi.sdk import Context, NodeError, main
from verdi.types import io

CKPTS = {"score": "ScoreNet/scorenet.pth", "energy": "EnergyNet/energynet.pth",
         "scale": "ScaleNet/scalenet.pth"}


def _load_genpose2(ctx: Context):
    repo = ctx.repo
    for p in (repo / "networks/pts_encoder/pointnet2_utils/pointnet2", repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    paths = {}
    for k, rel in CKPTS.items():
        path = ctx.weight("ckpts") / rel
        if not path.exists():
            raise NodeError(f"GenPose++ checkpoint missing: {path}",
                            hint="run `verdi setup genpose2` (see NOTES.md "
                            "for the Dropbox download)", kind="setup")
        paths[k] = str(path)
    argv, sys.argv = sys.argv, sys.argv[:1]  # upstream config parses argv
    try:
        from runners.infer import GenPose2, visualize_pose
        from datasets.datasets_infer import InferDataset

        model = GenPose2(score_model_path=paths["score"],
                         energy_model_path=paths["energy"],
                         scale_model_path=paths["scale"])
    finally:
        sys.argv = argv
    return model, InferDataset, visualize_pose


def _box_corners(T: np.ndarray, size: np.ndarray) -> np.ndarray:
    """8 corners (camera frame, metres) of the box of side lengths size."""
    signs = np.array([[x, y, z] for x in (1, -1) for y in (1, -1)
                      for z in (1, -1)], float)
    pts = signs * (np.asarray(size, float) / 2.0)
    return pts @ T[:3, :3].T + T[:3, 3]


def estimate(ctx: Context) -> None:
    import torch
    from PIL import Image

    rgb = io.read_image(ctx.input("image"))
    h, w = rgb.shape[:2]
    depth = np.load(ctx.input("depth")).astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    ids = np.asarray(Image.open(ctx.input("mask")))
    if depth.shape != (h, w) or ids.shape != (h, w):
        raise NodeError(f"image {(h, w)}, depth {depth.shape} and mask "
                        f"{ids.shape} must have the same size")
    obj_ids = [int(i) for i in np.unique(ids) if i != 0]
    if not obj_ids:
        raise NodeError("mask has no instance (all 0)",
                        hint="segment the objects first, e.g. with sam2 / "
                        "langsam")
    if max(obj_ids) >= 255:
        raise NodeError("GenPose++ supports instance ids 1..254 "
                        "(255 is its background value)")
    for i in obj_ids:
        valid = (ids == i) & (depth > 0) & (depth <= 4.0)
        if valid.sum() < 10:
            raise NodeError(f"instance {i} has {int(valid.sum())} pixels with "
                            "valid depth in (0, 4] m",
                            hint="GenPose++ ignores depth beyond 4 m; check "
                            "the depth units (metres) and mask alignment")
    cam = io.read_camera(ctx.input("camera"))
    K = cam["K"]
    meta = {"camera": {"intrinsics": {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
        "cy": float(K[1, 2]), "width": int(cam["width"]),
        "height": int(cam["height"])}}}
    mask_gp = np.full((h, w), 255, np.uint8)  # upstream background = 255
    for i in obj_ids:
        mask_gp[ids == i] = i

    model, InferDataset, visualize_pose = _load_genpose2(ctx)
    model.cfg.eval_repeat_num = int(ctx.param("repeat_num", 50))
    data = InferDataset({"color": rgb, "depth": depth, "mask": mask_gp,
                         "meta": meta}, img_size=model.cfg.img_size,
                        device=model.cfg.device, n_pts=model.cfg.num_points)
    with torch.no_grad():
        try:
            pose, length = model.inference(data)
        except AssertionError as exc:
            raise NodeError(f"GenPose++ rejected the input: {exc}") from exc
    poses_np = pose[0].cpu().numpy()
    sizes_np = length[0].cpu().numpy()
    if len(poses_np) != len(obj_ids):
        raise NodeError(f"got {len(poses_np)} poses for {len(obj_ids)} "
                        "instances")

    poses = []
    for oid, T, size in zip(obj_ids, poses_np, sizes_np):
        T = np.asarray(T, float)
        u, _, vt = np.linalg.svd(T[:3, :3])  # float32 -> clean rotation
        T[:3, :3] = u @ vt
        poses.append({"T_cam_obj": T, "label": f"instance_{oid}",
                      "mask_id": oid, "size": [float(s) for s in size],
                      "bbox3d_cam": _box_corners(T, size).tolist()})
    out = ctx.output_path("poses", "poses.json")
    io.write_pose_set(out, poses)
    ctx.set_output("poses", out)

    import cv2

    vis_bgr = visualize_pose(data, pose, length)
    vis = ctx.output_path("vis", "vis.png")
    io.write_image(vis, cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
    ctx.set_output("vis", vis)
    ctx.metadata["instances"] = obj_ids


if __name__ == "__main__":
    raise SystemExit(main({"estimate": estimate}))
